"""Collect exact-reset Stack_D0 or Lift_D0 evaluation episodes with a SpaceMouse.

Each approved successful rollout is written to its own HDF5 file. In addition
to the full action / simulator-state trajectory, every file stores synchronized
observations and the initial, first-grasp, and successful-goal frame indices.
The saved model XML plus ``states[0]`` can be used to reproduce the exact scene
that produced those three frames.

Controls:
  - Move / twist SpaceMouse for XYZ + yaw (roll/pitch disabled)
  - Left button (click) toggles the gripper
  - Right button aborts the current trial
  - Review: y / n, or left / right SpaceMouse button
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.devices import SpaceMouse
from robosuite.utils.input_utils import input2action

# Importing MimicGen registers the D0 environments with robosuite.
import mimicgen  # noqa: F401


ROOT = Path(__file__).resolve().parents[3]
TASKS = {
	"stack": {
		"env_name": "Stack_D0",
		"object_attr": "cubeA",
		"save_dir": ROOT / "data" / "robot" / "eval_raw",
		"instruction": "grasp cube A and stack it on cube B",
	},
	"lift": {
		"env_name": "Lift_D0",
		"object_attr": "cube",
		"save_dir": ROOT / "data" / "robot" / "eval_raw",
		"instruction": "grasp and lift the cube",
	},
}
CAM_ABOVE = "agentview"
CAM_WRIST = "robot0_eye_in_hand"
CAMERAS = (CAM_ABOVE, CAM_WRIST)
IMG_SIZE = 84
VIDEO_SIZE = 256
CONTROL_FREQ = 20
MAX_STEPS = 600
SUCCESS_HOLD = 10

# Training images use robosuite's OpenCV (upright) convention.
macros.IMAGE_CONVENTION = "opencv"


def set_geom_groups(env, collision: bool = False, visual: bool = True) -> None:
	"""Hide collision geoms / show visual meshes (MuJoCo 2.3.x fix)."""
	ctx = env.sim._render_context_offscreen
	if ctx is None:
		return
	address = ctx.vopt.geomgroup.__array_interface__["data"][0]
	groups = (ctypes.c_uint8 * 6).from_address(address)
	for index in range(6):
		groups[index] = 0
	groups[0] = int(collision)
	groups[1] = int(visual)


def hide_overlays(env) -> None:
	"""Hide teleoperation guide sites from saved camera observations."""
	env.sim.model.site_rgba[:, 3] = 0.0


def show_ee_guide(env, object_attr: str) -> None:
	"""Show the end-effector guide in the live teleoperation window only."""
	robot = env.robots[0]
	env.sim.model.site_rgba[robot.eef_cylinder_id] = np.array(
		[0.0, 1.0, 0.0, 0.3]
	)
	env.sim.model.site_rgba[robot.eef_site_id] = np.array(
		[1.0, 0.0, 0.0, 0.5]
	)
	env._visualize_gripper_to_target(
		gripper=robot.gripper,
		target=getattr(env, object_attr),
	)


def is_cube_grasped(env, object_attr: str) -> bool:
	return bool(
		env._check_grasp(
			gripper=env.robots[0].gripper,
			object_geoms=getattr(env, object_attr),
		)
	)


def make_env(max_steps: int, env_name: str):
	controller = load_controller_config(default_controller="OSC_POSE")
	env = suite.make(
		env_name,
		robots="Panda",
		controller_configs=controller,
		has_renderer=True,
		has_offscreen_renderer=True,
		render_collision_mesh=False,
		render_visual_mesh=True,
		render_camera=CAM_ABOVE,
		use_camera_obs=True,
		camera_names=list(CAMERAS),
		camera_heights=IMG_SIZE,
		camera_widths=IMG_SIZE,
		use_object_obs=True,
		control_freq=CONTROL_FREQ,
		horizon=max_steps,
		ignore_done=True,
	)
	set_geom_groups(env)
	env_info = {
		"env_name": env_name,
		"robots": ["Panda"],
		"controller_configs": controller,
	}
	return env, env_info


def exact_reset_for_collection(env):
	"""Reset once, then collect from the exact XML / state pair we save."""
	env.reset()
	set_geom_groups(env)
	hide_overlays(env)
	model_xml = env.sim.model.get_xml()
	initial_state = np.asarray(env.sim.get_state().flatten()).copy()

	# This mirrors robosuite's DataCollectionWrapper and makes action playback
	# deterministic under the exact compiled model saved with the episode.
	env.reset_from_xml_string(model_xml)
	env.sim.reset()
	env.sim.set_state_from_flattened(initial_state)
	env.sim.forward()
	set_geom_groups(env)
	hide_overlays(env)
	obs = env._get_observations(force_update=True)
	return obs, model_xml


def _copy_observation(obs) -> dict[str, np.ndarray]:
	return {
		key: np.asarray(value).copy()
		for key, value in obs.items()
		if isinstance(value, np.ndarray)
	}


def collect_episode(env, device, max_steps: int, object_attr: str):
	"""Collect one candidate with observations aligned to simulator states."""
	obs, model_xml = exact_reset_for_collection(env)
	device.start_control()

	states = [np.asarray(env.sim.get_state().flatten()).copy()]
	actions = []
	rewards = []
	dones = []
	observations = [_copy_observation(obs)]
	grasp_flags = [is_cube_grasped(env, object_attr)]
	success_flags = [bool(env._check_success())]
	review_frames = []

	gripper_closed = False
	previous_button = False
	success_hold = -1
	aborted = False

	show_ee_guide(env, object_attr)
	env.render()

	while len(actions) < max_steps:
		action, _ = input2action(
			device=device,
			robot=env.robots[0],
			active_arm="right",
		)
		if action is None:
			aborted = True
			break

		action = np.asarray(action, dtype=np.float64).copy()
		action[3] = 0.0
		action[4] = 0.0

		button_held = bool(device.single_click_and_hold)
		if button_held and not previous_button:
			gripper_closed = not gripper_closed
		previous_button = button_held
		action[-1] = 1.0 if gripper_closed else -1.0

		# Guides are hidden while env.step creates the camera observations.
		hide_overlays(env)
		obs, reward, done, _ = env.step(action)
		set_geom_groups(env)

		actions.append(action.copy())
		rewards.append(float(reward))
		dones.append(bool(done))
		states.append(np.asarray(env.sim.get_state().flatten()).copy())
		observations.append(_copy_observation(obs))
		grasp_flags.append(is_cube_grasped(env, object_attr))
		success_flags.append(bool(env._check_success()))
		review_frames.append(
			np.asarray(
				env.sim.render(
					height=VIDEO_SIZE,
					width=VIDEO_SIZE,
					camera_name=CAM_ABOVE,
				)
			).copy()
		)

		show_ee_guide(env, object_attr)
		env.render()

		if success_flags[-1]:
			if success_hold > 0:
				success_hold -= 1
			else:
				success_hold = SUCCESS_HOLD
			if success_hold == 0:
				break
		else:
			success_hold = -1

	hide_overlays(env)
	grasp_indices = np.flatnonzero(np.asarray(grasp_flags, dtype=np.bool_))
	grasp_index = int(grasp_indices[0]) if grasp_indices.size else None
	success = (
		not aborted
		and bool(success_flags[-1])
		and 0 < len(actions) < max_steps
		and grasp_index is not None
	)
	return {
		"states": states,
		"actions": actions,
		"rewards": rewards,
		"dones": dones,
		"obs": observations,
		"grasp_flags": grasp_flags,
		"success_flags": success_flags,
		"grasp_index": grasp_index,
		"goal_index": len(states) - 1,
		"review_frames": review_frames,
		"model_xml": model_xml,
		"success": success,
		"aborted": aborted,
	}


def review_video(frames, task: str, device=None) -> bool:
	if not frames:
		return False
	path = Path(tempfile.gettempdir()) / f"{task}_eval_review.mp4"
	with imageio.get_writer(path, fps=CONTROL_FREQ) as writer:
		for frame in frames:
			writer.append_data(frame)
	print(f"\nReview video: {path}")
	try:
		subprocess.run(["open", str(path)], check=False)
	except Exception:
		pass

	print("Approve this eval? [y/n]  |  SpaceMouse: left=yes, right=no")
	if device is not None:
		device.start_control()
		while device.single_click_and_hold or device._reset_state:
			if device._reset_state:
				device.start_control()
			time.sleep(0.05)
		previous_button = False

	while True:
		if select.select([sys.stdin], [], [], 0.05)[0]:
			answer = sys.stdin.readline().strip().lower()
			if answer in ("y", "n"):
				return answer == "y"

		if device is None:
			continue
		button_held = bool(device.single_click_and_hold)
		if button_held and not previous_button:
			print("Approved (left button).")
			return True
		previous_button = button_held
		if device._reset_state:
			print("Rejected (right button).")
			device.start_control()
			return False


def _episode_files(save_dir: Path) -> list[Path]:
	return sorted(save_dir.glob("eval_[0-9][0-9][0-9][0-9].hdf5"))


def next_episode_index(save_dir: Path) -> int:
	files = _episode_files(save_dir)
	return int(files[-1].stem.split("_")[-1]) + 1 if files else 0


def _stack_observations(observations) -> dict[str, np.ndarray]:
	keys = observations[0].keys()
	for index, obs in enumerate(observations[1:], start=1):
		if obs.keys() != observations[0].keys():
			raise ValueError(f"Observation keys changed at frame {index}")
	return {key: np.stack([obs[key] for obs in observations]) for key in keys}


def save_episode(
	save_dir: Path,
	index: int,
	episode,
	env_info,
	env_name: str,
) -> Path:
	"""Atomically save one complete, approved evaluation episode."""
	save_dir.mkdir(parents=True, exist_ok=True)
	path = save_dir / f"eval_{index:04d}.hdf5"
	partial = save_dir / f"eval_{index:04d}.partial.hdf5"
	if path.exists() or partial.exists():
		raise FileExistsError(f"Refusing to overwrite {path} or {partial}")

	states = np.asarray(episode["states"])
	actions = np.asarray(episode["actions"])
	if states.shape[0] != actions.shape[0] + 1:
		raise ValueError("Expected exactly one more state / observation than action")
	stacked_obs = _stack_observations(episode["obs"])
	if any(value.shape[0] != states.shape[0] for value in stacked_obs.values()):
		raise ValueError("Observation trajectory is not aligned with simulator states")

	with h5py.File(partial, "w") as file:
		data = file.create_group("data")
		demo = data.create_group("demo_0")
		demo.attrs["model_file"] = episode["model_xml"]
		demo.attrs["num_samples"] = actions.shape[0]
		demo.attrs["initial_index"] = 0
		demo.attrs["grasp_index"] = int(episode["grasp_index"])
		demo.attrs["goal_index"] = int(episode["goal_index"])
		demo.attrs["success"] = True
		demo.create_dataset("states", data=states)
		demo.create_dataset("actions", data=actions)
		demo.create_dataset("rewards", data=np.asarray(episode["rewards"]))
		demo.create_dataset("dones", data=np.asarray(episode["dones"], dtype=np.bool_))
		demo.create_dataset(
			"grasp_flags",
			data=np.asarray(episode["grasp_flags"], dtype=np.bool_),
		)
		demo.create_dataset(
			"success_flags",
			data=np.asarray(episode["success_flags"], dtype=np.bool_),
		)
		obs_group = demo.create_group("obs")
		for key, values in stacked_obs.items():
			kwargs = {"compression": "lzf"} if values.ndim >= 3 else {}
			obs_group.create_dataset(key, data=values, **kwargs)

		now = datetime.datetime.now()
		data.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
		data.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
		data.attrs["repository_version"] = suite.__version__
		data.attrs["mujoco_version"] = mujoco.__version__
		data.attrs["env"] = env_name
		data.attrs["env_info"] = json.dumps(env_info)
		data.attrs["total"] = actions.shape[0]
		file.flush()

	os.replace(partial, path)
	print(
		f"Saved {path} ({actions.shape[0]} actions, "
		f"grasp frame {episode['grasp_index']}, goal frame {episode['goal_index']})"
	)
	return path


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Collect exact-reset Stack_D0 or Lift_D0 eval episodes with a SpaceMouse"
	)
	parser.add_argument(
		"num_episodes",
		nargs="?",
		type=int,
		default=50,
		help="Target total approved evals in the output folder (default: 50)",
	)
	parser.add_argument(
		"--task",
		choices=TASKS,
		default="lift",
		help="Robosuite task to collect (default: lift)",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Override the task-specific output directory",
	)
	parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
	parser.add_argument("--pos-sensitivity", type=float, default=1.0)
	parser.add_argument("--rot-sensitivity", type=float, default=1.0)
	args = parser.parse_args()
	task = TASKS[args.task]
	output_dir = args.output_dir or task["save_dir"]

	if args.num_episodes < 1:
		raise ValueError("num_episodes must be at least 1")
	if args.max_steps < 2:
		raise ValueError("max_steps must be at least 2")

	output_dir.mkdir(parents=True, exist_ok=True)
	existing = len(_episode_files(output_dir))
	if existing >= args.num_episodes:
		print(
			f"Already have {existing} evals in {output_dir} "
			f"(target {args.num_episodes})."
		)
		return

	env, env_info = make_env(args.max_steps, task["env_name"])
	device = SpaceMouse(
		pos_sensitivity=args.pos_sensitivity,
		rot_sensitivity=args.rot_sensitivity,
	)
	print(f"Task: {task['instruction']}.")
	print("Right button aborts; left click toggles the gripper.")
	print(f"Saving exact-reset evals to {output_dir}")
	print(f"Progress: {existing}/{args.num_episodes}")

	saved = existing
	try:
		while saved < args.num_episodes:
			print(f"\n=== Eval trial {saved + 1}/{args.num_episodes} ===")
			episode = collect_episode(
				env, device, args.max_steps, task["object_attr"]
			)
			if not episode["success"]:
				reason = "aborted" if episode["aborted"] else "not successful"
				if episode["grasp_index"] is None:
					reason += "; no grasp frame detected"
				print(f"Rejected automatically ({reason}, {len(episode['actions'])} steps).")
				continue

			print(
				f"Candidate succeeded in {len(episode['actions'])} steps; "
				"opening review video."
			)
			if not review_video(episode["review_frames"], args.task, device):
				print("Rejected by review.")
				continue

			index = next_episode_index(output_dir)
			save_episode(
				output_dir, index, episode, env_info, task["env_name"]
			)
			saved += 1
			print(f"Progress: {saved}/{args.num_episodes}")
	finally:
		env.close()

	print(
		f"\nDone. Run process_eval.py --task {args.task} to build "
		f"evals.hdf5 from {output_dir}."
	)


if __name__ == "__main__":
	main()
