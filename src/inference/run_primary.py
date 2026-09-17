"""Closed-loop Lift evaluation for the corrected primary LeWM checkpoint."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import tempfile
import time
import zipfile
from collections import deque
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from architectures.predictor import PredictorNetwork
from architectures.robot_encoder import RobotEncoder
from datasets.robot.primary_dataset import (
	build_robot_state,
	normalize_robot_images,
	normalize_robot_state,
)
from inference.mpc_parallel import Planner
from primary_contract import (
	ACTION_ENCODING_ID,
	ARCHITECTURE_ID,
	CHECKPOINT_FORMAT_VERSION,
	IMAGE_NORMALIZATION_ID,
)

import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.lift import Lift
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class Lift_D0(Lift, SingleArmEnv_MG):
	def edit_model_xml(self, xml_str):
		return SingleArmEnv_MG.edit_model_xml(self, xml_str)

	def reward(self, action=None):
		return Lift.reward(self, action=action)


CAM_ABOVE = "agentview"
CAM_WRIST = "robot0_eye_in_hand"
ENV_NAME = "Lift_D0"
IMAGE_SIZE = 84
WINDOW = 3
EMBEDDING_DIM = 192
FRAME_SKIP = 5
PRIMITIVE_ACTION_DIM = 7
PREDICTOR_ACTION_DIM = FRAME_SKIP * PRIMITIVE_ACTION_DIM
HORIZON = 5
PLAN_STEPS = FRAME_SKIP * HORIZON
CONTROL_FREQ = 20
macros.IMAGE_CONVENTION = "opencv"


def select_checkpoint(model_dir: Path) -> Path:
	best = model_dir / "best_model.zip"
	if best.exists():
		return best
	checkpoints = []
	for path in model_dir.glob("epoch_*_model.zip"):
		match = re.fullmatch(r"epoch_(\d+)_model\.zip", path.name)
		if match:
			checkpoints.append((int(match.group(1)), path))
	if not checkpoints:
		raise FileNotFoundError(f"No primary checkpoint found in {model_dir}")
	return max(checkpoints)[1]


def _load_torch(zf, member, tmp, weights_only):
	zf.extract(member, tmp)
	try:
		return torch.load(tmp / member, map_location="cpu", weights_only=weights_only)
	except TypeError:
		return torch.load(tmp / member, map_location="cpu")


def load_models(checkpoint: Path, device):
	with zipfile.ZipFile(checkpoint, "r") as zf, tempfile.TemporaryDirectory() as tmp_name:
		tmp = Path(tmp_name)
		if "state_stats.pth" not in zf.namelist():
			raise ValueError("Checkpoint has no state/action statistics; retrain it")
		stats = _load_torch(zf, "state_stats.pth", tmp, True)
		expected = {
			"checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
			"architecture_id": ARCHITECTURE_ID,
			"action_encoding_id": ACTION_ENCODING_ID,
			"image_normalization_id": IMAGE_NORMALIZATION_ID,
			"frame_skip": FRAME_SKIP,
			"primitive_action_dim": PRIMITIVE_ACTION_DIM,
			"predictor_action_dim": PREDICTOR_ACTION_DIM,
		}
		for key, value in expected.items():
			if stats.get(key) != value:
				raise ValueError(
					f"Incompatible checkpoint: {key}={stats.get(key)!r}, expected {value!r}. "
					"This evaluator requires a newly trained corrected checkpoint."
				)

		encoder = RobotEncoder().to(device)
		predictor = PredictorNetwork(d_action=PREDICTOR_ACTION_DIM, window=WINDOW).to(device)
		encoder.load_state_dict(_load_torch(zf, "encoder_weights.pth", tmp, True))
		predictor.load_state_dict(_load_torch(zf, "predictor_weights.pth", tmp, True))

	encoder.eval()
	predictor.eval()
	return encoder, predictor, {key: torch.as_tensor(stats[key]).float() for key in (
		"state_mean", "state_std", "action_mean", "action_std", "action_low", "action_high"
	)}


def set_geom_groups(env, collision=False, visual=True):
	context = env.sim._render_context_offscreen
	if context is None:
		return
	address = context.vopt.geomgroup.__array_interface__["data"][0]
	groups = (ctypes.c_uint8 * 6).from_address(address)
	for index in range(6):
		groups[index] = 0
	groups[0] = int(collision)
	groups[1] = int(visual)


def observation_tensors(obs, device, stats):
	above = torch.from_numpy(obs[f"{CAM_ABOVE}_image"]).permute(2, 0, 1).float() / 255.0
	wrist = torch.from_numpy(obs[f"{CAM_WRIST}_image"]).permute(2, 0, 1).float() / 255.0
	above = normalize_robot_images(above)
	wrist = normalize_robot_images(wrist)
	state = build_robot_state(
		torch.from_numpy(obs["robot0_eef_pos"]).float(),
		torch.from_numpy(obs["robot0_eef_quat"]).float(),
		torch.from_numpy(obs["robot0_gripper_qpos"]).float(),
	)
	state = normalize_robot_state(state, stats["state_mean"], stats["state_std"])
	return above[None, None].to(device), wrist[None, None].to(device), state[None, None].to(device)


@torch.no_grad()
def encode_observation(encoder, obs, device, stats):
	return encoder(*observation_tensors(obs, device, stats))[0, 0]


@torch.no_grad()
def encode_demo_frames(encoder, obs, indices, device, stats, batch_size=128):
	outputs = []
	for offset in range(0, len(indices), batch_size):
		batch_indices = indices[offset:offset + batch_size]
		above = torch.from_numpy(obs["agentview_image"][batch_indices]).permute(0, 3, 1, 2).float() / 255.0
		wrist = torch.from_numpy(obs["robot0_eye_in_hand_image"][batch_indices]).permute(0, 3, 1, 2).float() / 255.0
		above = normalize_robot_images(above).unsqueeze(1).to(device)
		wrist = normalize_robot_images(wrist).unsqueeze(1).to(device)
		state = build_robot_state(
			torch.from_numpy(obs["robot0_eef_pos"][batch_indices]).float(),
			torch.from_numpy(obs["robot0_eef_quat"][batch_indices]).float(),
			torch.from_numpy(obs["robot0_gripper_qpos"][batch_indices]).float(),
		)
		state = normalize_robot_state(state, stats["state_mean"], stats["state_std"])
		outputs.append(encoder(above, wrist, state[:, None].to(device))[:, 0].cpu())
	return torch.cat(outputs).to(device)


def _text(value):
	return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_subgoal_labels(path):
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(
			f"Manual subgoal labels not found: {path}. Run label_subgoals.py first."
		)
	with Path(path).open("r", encoding="utf-8") as stream:
		data = json.load(stream)
	if data.get("format_version") != 1 or not isinstance(data.get("labels"), dict):
		raise ValueError(f"Unsupported subgoal label file format: {path}")
	return data["labels"]


@torch.no_grad()
def load_eval_bank(encoder, path, labels_path, device, stats, demo_name=None):
	labels = load_subgoal_labels(labels_path)
	entries = []
	with h5py.File(path, "r") as file:
		if _text(file["data"].attrs.get("env", "")) != ENV_NAME:
			raise ValueError(f"{path} does not contain {ENV_NAME} evals")
		version = file["data"].attrs.get("mujoco_version")
		if version is not None and _text(version) != mujoco.__version__:
			print(f"WARNING: evals use MuJoCo {_text(version)}; inference uses {mujoco.__version__}")
		if demo_name is not None:
			if demo_name not in labels:
				raise KeyError(f"{demo_name!r} has no manual grasp label in {labels_path}")
			names = [demo_name]
		elif "mask/eval" in file:
			all_names = [_text(value) for value in file["mask/eval"][:]]
			names = [name for name in all_names if name in labels]
		else:
			all_names = sorted(file["data"].keys())
			names = [name for name in all_names if name in labels]
		if not names:
			raise ValueError(f"No labeled eval demos from {path} were found in {labels_path}")
		if demo_name is None and len(names) != len(all_names):
			print(f"using {len(names)}/{len(all_names)} eval demos with manual grasp labels")
		for name in names:
			demo = file[f"data/{name}"]
			num_frames = int(demo["states"].shape[0])
			label = labels[name]
			grasp_index = int(label["grasp_frame"])
			goal_index = int(label.get("goal_frame", num_frames - 1))
			if "num_frames" in label and int(label["num_frames"]) != num_frames:
				raise ValueError(
					f"{name}: labels describe {label['num_frames']} frames, eval has {num_frames}"
				)
			if not 0 < grasp_index < goal_index < num_frames:
				raise ValueError(
					f"{name}: expected 0 < grasp_frame < goal_frame < {num_frames}, "
					f"got {grasp_index} and {goal_index}"
				)
			if goal_index != num_frames - 1:
				raise ValueError(
					f"{name}: goal_frame must be the final frame ({num_frames - 1}), got {goal_index}"
				)
			embeddings = encode_demo_frames(
				encoder, demo["obs"], [0, grasp_index, goal_index], device, stats
			)
			model_xml = _text(demo.attrs["model_file"])
			entries.append({
				"name": name,
				"start": embeddings[0],
				"stages": (
					{"name": "grasp", "goal": embeddings[1], "frame": grasp_index},
					{"name": "lift", "goal": embeddings[2], "frame": goal_index},
				),
				"initial_state": np.asarray(demo["states"][0]).copy(),
				"model_xml": model_xml,
			})
	return entries


def make_env(horizon):
	controller = load_controller_config(default_controller="OSC_POSE")
	env = suite.make(
		ENV_NAME, robots="Panda", controller_configs=controller,
		has_renderer=False, has_offscreen_renderer=True,
		render_collision_mesh=False, render_visual_mesh=True,
		use_camera_obs=True, camera_names=[CAM_ABOVE, CAM_WRIST],
		camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
		control_freq=CONTROL_FREQ, horizon=horizon, ignore_done=True,
	)
	set_geom_groups(env)
	return env


def reset_to_eval_state(env, entry):
	env.reset()
	env.reset_from_xml_string(env.edit_model_xml(entry["model_xml"]))
	env.sim.reset()
	env.sim.set_state_from_flattened(entry["initial_state"])
	env.sim.forward()
	set_geom_groups(env)
	return env._get_observations(force_update=True)


def render_frame(env, size=256):
	views = [
		env.sim.render(height=size, width=size, camera_name=camera)[::-1]
		for camera in (CAM_ABOVE, CAM_WRIST)
	]
	return np.concatenate(views, axis=1).astype(np.uint8)


def postprocess_action(action):
	action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0).copy()
	action[-1] = 1.0 if action[-1] >= 0 else -1.0
	return action


def held_gripper_schedule(planner, command):
	"""Keep the last executed gripper command fixed throughout a CEM plan."""
	raw = np.full((1, planner.H, planner.block_size), command, dtype=np.float32)
	return (raw - planner.raw_action_mean[-1]) / planner.raw_action_std[-1]


def run_rollout(
	env,
	planner,
	encoder,
	entry,
	device,
	stats,
	writer,
	stage_settings,
):
	for stage in entry["stages"]:
		if stage["name"] not in stage_settings:
			raise KeyError(f"Missing settings for {stage['name']!r} stage")
		settings = stage_settings[stage["name"]]
		threshold = float(settings["threshold"])
		if int(settings["budget"]) < 1 or not np.isfinite(threshold) or threshold < 0:
			raise ValueError(f"Invalid settings for {stage['name']!r} stage")
	obs = reset_to_eval_state(env, entry)
	writer.append_data(render_frame(env))
	z = encode_observation(encoder, obs, device, stats)
	print(f"exact eval {entry['name']} | start embedding MSE: {(z-entry['start']).square().mean().item():.6f}")
	context = deque([z], maxlen=WINDOW)
	action_history = deque(maxlen=WINDOW - 1)
	total_steps = 0
	last_gripper_command = None

	for number, stage in enumerate(entry["stages"], 1):
		goal = stage["goal"]
		budget = int(stage_settings[stage["name"]]["budget"])
		threshold = float(stage_settings[stage["name"]]["threshold"])
		use_threshold = bool(stage_settings[stage["name"]].get("use_threshold", True))
		steps = 0
		current_mse = (context[-1] - goal).square().mean().item()
		reached = use_threshold and current_mse <= threshold
		threshold_text = f"{threshold:.4f}" if use_threshold else "disabled"
		print(
			f"{stage['name']} stage {number}/{len(entry['stages'])}: "
			f"demo frame {stage['frame']}, threshold={threshold_text}"
		)
		if reached:
			print(f"  already within threshold (latent MSE {current_mse:.4f})")
		while steps < budget and not reached:
			zi = torch.stack(tuple(context))
			a = (
				torch.stack(tuple(action_history))
				if action_history else torch.empty(0, PREDICTOR_ACTION_DIM, device=device)
			)
			if stage["name"] == "lift" and last_gripper_command is not None:
				gripper_schedules = held_gripper_schedule(planner, last_gripper_command)
			else:
				gripper_schedules = planner.single_switch_gripper_schedules()
			plan = planner.CEM(zi, a, goal, gripper_schedules=gripper_schedules)
			raw_blocks = planner.denormalize_action_blocks(plan)
			for raw_block in raw_blocks:
				executed = []
				for raw_action in raw_block:
					action = postprocess_action(raw_action)
					obs, _, done, _ = env.step(action)
					set_geom_groups(env)
					writer.append_data(render_frame(env))
					executed.append(action)
					last_gripper_command = float(action[-1])
					steps += 1
					total_steps += 1
					if env._check_success():
						return True, total_steps
					if done or steps >= budget:
						break

				normalized_block = planner.normalize_primitive_actions(np.stack(executed))
				if len(executed) == FRAME_SKIP:
					action_history.append(torch.as_tensor(
						normalized_block.reshape(-1), dtype=z.dtype, device=device
					))
				z = encode_observation(encoder, obs, device, stats)
				context.append(z)
				goal_mse = (z - goal).square().mean().item()
				print(f"  steps {steps:>3}/{budget} | latent MSE {goal_mse:.4f}")
				if use_threshold and goal_mse <= threshold:
					reached = True
					break
				if done or steps >= budget:
					break
		if not use_threshold:
			print(f"  fixed {stage['name']} budget complete ({steps}/{budget} steps)")
			continue
		if not reached:
			print("subgoal budget exhausted without reaching the latent goal")
			return False, total_steps
	return bool(env._check_success()), total_steps


def next_rollout_number(rollout_dir: Path) -> int:
	"""Continue after the largest completed rollout_N_pass/fail video."""
	latest = 0
	# Search anywhere in the name so videos from the earlier
	# <checkpoint>_<demo>_rollout_N_<result> convention are counted too.
	pattern = re.compile(r"(?:^|_)rollout_(\d+)_(?:pass|fail)\.mp4$")
	for path in rollout_dir.glob("*rollout_*_*.mp4"):
		match = pattern.search(path.name)
		if match:
			latest = max(latest, int(match.group(1)))
	return latest + 1


def evaluate(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	checkpoint = args.checkpoint or select_checkpoint(args.model_dir)
	encoder, predictor, stats = load_models(checkpoint, device)
	action_mean = stats["action_mean"].numpy()
	action_std = stats["action_std"].numpy()
	action_low = np.tile(((stats["action_low"] - stats["action_mean"]) / stats["action_std"]).numpy(), FRAME_SKIP)
	action_high = np.tile(((stats["action_high"] - stats["action_mean"]) / stats["action_std"]).numpy(), FRAME_SKIP)
	planner = Planner(
		device, encoder, predictor, H=HORIZON, N=args.samples, K=args.elites, T=args.iterations,
		action_dim=PREDICTOR_ACTION_DIM, primitive_action_dim=PRIMITIVE_ACTION_DIM,
		action_low=action_low, action_high=action_high,
		raw_action_mean=action_mean, raw_action_std=action_std,
		min_std=0.05, momentum=0.1, seed=args.seed,
	)
	entries = load_eval_bank(encoder, args.evals, args.labels, device, stats, args.demo)
	rng = np.random.default_rng(args.seed)
	indices = rng.choice(len(entries), args.rollouts, replace=args.rollouts > len(entries))
	stage_settings = {
		"grasp": {
			"budget": args.grasp_budget,
			"threshold": args.grasp_threshold,
			"use_threshold": not args.disable_thresholds,
		},
		"lift": {
			"budget": args.lift_budget,
			"threshold": args.lift_threshold,
			"use_threshold": not args.disable_thresholds,
		},
	}
	env = make_env(max(200, args.grasp_budget + args.lift_budget))
	args.rollout_dir.mkdir(parents=True, exist_ok=True)
	first_rollout_number = next_rollout_number(args.rollout_dir)
	print(f"completed rollout videos found through #{first_rollout_number - 1}; starting at #{first_rollout_number}")
	successes = 0
	try:
		for batch_index, index in enumerate(indices):
			rollout_number = first_rollout_number + batch_index
			entry = entries[int(index)]
			pending = args.rollout_dir / f"rollout_{rollout_number}_pending.mp4"
			pending.unlink(missing_ok=True)
			writer = imageio.get_writer(str(pending), fps=CONTROL_FREQ, codec="libx264")
			try:
				success, steps = run_rollout(
					env, planner, encoder, entry, device, stats, writer,
					stage_settings,
				)
			finally:
				writer.close()
			result = pending.with_name(pending.name.replace("pending", "pass" if success else "fail"))
			pending.replace(result)
			successes += int(success)
			print(
				f"batch {batch_index + 1}/{args.rollouts} → rollout_{rollout_number}: "
				f"{'pass' if success else 'fail'} ({steps} steps) -> {result}"
			)
	finally:
		env.close()
	print(f"success rate: {successes}/{args.rollouts} = {100*successes/args.rollouts:.1f}%")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
	parser.add_argument("--checkpoint", type=Path)
	parser.add_argument("--evals", type=Path, default=ROOT / "data/robot/mimicgen/evals.hdf5")
	parser.add_argument(
		"--labels", type=Path,
		default=ROOT / "data/robot/mimicgen/subgoal_labels.json",
	)
	parser.add_argument("--rollout-dir", type=Path, default=ROOT / "rollouts")
	parser.add_argument("--demo")
	parser.add_argument("--rollouts", type=int, default=5)
	parser.add_argument("--grasp-budget", type=int, default=100)
	parser.add_argument("--grasp-threshold", type=float, default=0.05)
	parser.add_argument("--lift-budget", type=int, default=100)
	parser.add_argument("--lift-threshold", type=float, default=0.05)
	parser.add_argument(
		"--disable-thresholds", action="store_true",
		help="Run each stage for its complete fixed budget; latent MSE is diagnostic only",
	)
	parser.add_argument("--samples", type=int, default=300)
	parser.add_argument("--elites", type=int, default=30)
	parser.add_argument("--iterations", type=int, default=10)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	if args.rollouts < 1:
		parser.error("--rollouts must be positive")
	if min(args.grasp_budget, args.lift_budget) < 1:
		parser.error("stage budgets must be positive")
	if min(args.grasp_threshold, args.lift_threshold) < 0:
		parser.error("stage thresholds must be nonnegative")
	evaluate(args)


if __name__ == "__main__":
	main()
