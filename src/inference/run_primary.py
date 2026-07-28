"""
Evaluate the primary world model + CEM planner on robosuite Stack.

Loads encoder/predictor from an epoch_*_model.zip checkpoint, runs N rollouts,
and reports success rate. Does not modify the planner — only calls Planner.CEM.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import tempfile
import zipfile
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

# Repo checkout is named `robosuite/`; drop it so the installed package wins.
_repo_rs = str((ROOT / "robosuite").resolve())
sys.path = [p for p in sys.path if str(Path(p).resolve()) != _repo_rs]
sys.path = [p for p in sys.path if p not in ("", str(ROOT), str(ROOT.resolve()))]

from architectures.predictor import PredictorNetwork
from architectures.robot_encoder import RobotEncoder
from inference.mpc import Planner

import robosuite as suite
from robosuite.controllers import load_controller_config

CAM_ABOVE = "agentview"
CAM_WRIST = "robot0_eye_in_hand"
IMG_SIZE = 84
WINDOW = 3
ACTION_DIM = 7
DEFAULT_MAX_STEPS = 400
DEFAULT_HDF5 = ROOT / "data" / "robot" / "mimicgen" / "demo.hdf5"


def set_geom_groups(env, collision: bool = False, visual: bool = True) -> None:
	"""Hide collision geoms / show visual meshes (MuJoCo 2.3.2 + NumPy 2.x fix)."""
	ctx = env.sim._render_context_offscreen
	if ctx is None:
		return
	addr = ctx.vopt.geomgroup.__array_interface__["data"][0]
	groups = (ctypes.c_uint8 * 6).from_address(addr)
	for i in range(6):
		groups[i] = 0
	groups[0] = int(collision)
	groups[1] = int(visual)


def obs_to_tensors(obs, device):
	above = torch.from_numpy(obs[f"{CAM_ABOVE}_image"]).float() / 255.0
	wrist = torch.from_numpy(obs[f"{CAM_WRIST}_image"]).float() / 255.0
	above = above.permute(2, 0, 1)  # (3, H, W)
	wrist = wrist.permute(2, 0, 1)
	state = torch.from_numpy(
		np.concatenate(
			[
				obs["robot0_eef_pos"],
				obs["robot0_eef_quat"],
				obs["robot0_gripper_qpos"],
			],
			axis=-1,
		)
	).float()
	return (
		above.unsqueeze(0).unsqueeze(0).to(device),  # (1, 1, 3, H, W)
		wrist.unsqueeze(0).unsqueeze(0).to(device),
		state.unsqueeze(0).unsqueeze(0).to(device),  # (1, 1, 9)
	)


@torch.no_grad()
def encode_obs(encoder, obs, device):
	above, wrist, state = obs_to_tensors(obs, device)
	return encoder(above, wrist, state).squeeze(0).squeeze(0)  # (192,)


def load_models(zip_path: Path, device: str):
	encoder = RobotEncoder().to(device)
	predictor = PredictorNetwork(d_action=ACTION_DIM, window=WINDOW).to(device)

	with zipfile.ZipFile(zip_path, "r") as zf, tempfile.TemporaryDirectory() as tmp:
		tmp = Path(tmp)
		zf.extract("encoder_weights.pth", tmp)
		zf.extract("predictor_weights.pth", tmp)
		encoder.load_state_dict(
			torch.load(tmp / "encoder_weights.pth", map_location=device, weights_only=True)
		)
		predictor.load_state_dict(
			torch.load(tmp / "predictor_weights.pth", map_location=device, weights_only=True)
		)

	encoder.eval()
	predictor.eval()
	return encoder, predictor


def _chw_from_h5(arr: np.ndarray) -> torch.Tensor:
	"""HDF5 images are HWC uint8 → float CHW in [0, 1]."""
	t = torch.from_numpy(arr).float() / 255.0
	if t.ndim == 3 and t.shape[-1] == 3:
		t = t.permute(2, 0, 1)
	return t


@torch.no_grad()
def load_goal_embedding(encoder, hdf5_path: Path, device: str, demo_name: str | None = None):
	"""Encode the final frame of a successful demo as the planning goal."""
	if not hdf5_path.exists():
		raise FileNotFoundError(
			f"Goal HDF5 not found: {hdf5_path}. Pass --goal-hdf5 to a MimicGen demo.hdf5."
		)

	with h5py.File(hdf5_path, "r") as f:
		if demo_name is None:
			if "mask" in f and "train" in f["mask"]:
				raw = f["mask/train"][0]
				demo_name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
			else:
				demo_name = sorted(f["data"].keys())[0]

		obs = f[f"data/{demo_name}/obs"]
		above = _chw_from_h5(obs["agentview_image"][-1])
		wrist = _chw_from_h5(obs["robot0_eye_in_hand_image"][-1])
		state = torch.from_numpy(
			np.concatenate(
				[
					obs["robot0_eef_pos"][-1],
					obs["robot0_eef_quat"][-1],
					obs["robot0_gripper_qpos"][-1],
				],
				axis=-1,
			)
		).float()

	above = above.unsqueeze(0).unsqueeze(0).to(device)
	wrist = wrist.unsqueeze(0).unsqueeze(0).to(device)
	state = state.unsqueeze(0).unsqueeze(0).to(device)
	zg = encoder(above, wrist, state).squeeze(0).squeeze(0)
	print(f"goal embedding from {hdf5_path.name}:{demo_name} (last frame)")
	return zg


def make_env(gui: bool, max_steps: int):
	controller = load_controller_config(default_controller="OSC_POSE")
	env = suite.make(
		"Stack",
		robots="Panda",
		controller_configs=controller,
		has_renderer=gui,
		has_offscreen_renderer=True,
		render_collision_mesh=False,
		render_visual_mesh=True,
		use_camera_obs=True,
		camera_names=[CAM_ABOVE, CAM_WRIST],
		camera_heights=IMG_SIZE,
		camera_widths=IMG_SIZE,
		control_freq=20,
		horizon=max_steps,
		ignore_done=True,
	)
	set_geom_groups(env, collision=False, visual=True)
	return env


def postprocess_action(action: np.ndarray) -> np.ndarray:
	action = np.asarray(action, dtype=np.float64).copy()
	action[6] = 1.0 if action[6] > 0 else -1.0
	return action


def run_episode(env, planner: Planner, encoder, zg, device, max_steps: int, gui: bool):
	obs = env.reset()
	set_geom_groups(env, collision=False, visual=True)
	if gui:
		env.render()

	# Warm-start context window with the first observation embedding.
	z0 = encode_obs(encoder, obs, device)
	zi = torch.stack([z0] * WINDOW, dim=0)  # (W, D)
	# Past actions a_{t-W+1} ... a_{t-1} (no a_t); zeros until we act.
	past_actions = deque(
		[torch.zeros(ACTION_DIM, dtype=z0.dtype, device=device) for _ in range(WINDOW - 1)],
		maxlen=WINDOW - 1,
	)

	for _ in range(max_steps):
		a_hist = torch.stack(list(past_actions), dim=0)  # (W-1, A)
		# CEM returns mean[0]: the single next action (action_dim,)
		action = postprocess_action(planner.CEM(zi, a_hist, zg))

		obs, _reward, done, _info = env.step(action)
		set_geom_groups(env, collision=False, visual=True)
		if gui:
			env.render()

		past_actions.append(
			torch.as_tensor(action, dtype=z0.dtype, device=device)
		)
		zt = encode_obs(encoder, obs, device)
		zi = torch.cat((zi[1:], zt.unsqueeze(0)), dim=0)

		if done or bool(env._check_success()):
			break

	return bool(env._check_success())


def evaluate(
	zip_path: Path,
	rollouts: int,
	gui: bool,
	max_steps: int,
	seed: int,
	device: str,
	goal_hdf5: Path,
	goal_demo: str | None,
):
	torch.manual_seed(seed)
	np.random.seed(seed)

	if not zip_path.exists():
		raise FileNotFoundError(f"Checkpoint zip not found: {zip_path}")

	encoder, predictor = load_models(zip_path, device)
	zg = load_goal_embedding(encoder, goal_hdf5, device, goal_demo)
	planner = Planner(device=device, enc=encoder, pred=predictor)

	env = make_env(gui=gui, max_steps=max_steps)
	successes = 0

	print(f"checkpoint: {zip_path}")
	print(
		f"device: {device} | rollouts: {rollouts} | max_steps: {max_steps} | "
		f"render: {'gui' if gui else 'headless'}"
	)

	for ep in range(rollouts):
		success = run_episode(env, planner, encoder, zg, device, max_steps, gui)
		successes += int(success)
		print(f"rollout {ep + 1}/{rollouts}: {'success' if success else 'fail'}")

	env.close()

	rate = successes / rollouts if rollouts else 0.0
	print(f"\nsuccess rate: {successes}/{rollouts} = {100 * rate:.1f}%")
	return rate


def main():
	parser = argparse.ArgumentParser(
		description="Evaluate primary world model + planner on Stack"
	)
	parser.add_argument(
		"zip_path",
		type=Path,
		help="Path to epoch_*_model.zip (encoder + predictor weights)",
	)
	parser.add_argument(
		"--rollouts",
		type=int,
		default=10,
		help="Number of evaluation rollouts",
	)
	parser.add_argument(
		"--max-steps",
		type=int,
		default=DEFAULT_MAX_STEPS,
		help=f"Env step cutoff per rollout (default {DEFAULT_MAX_STEPS})",
	)
	render = parser.add_mutually_exclusive_group()
	render.add_argument(
		"--gui",
		action="store_true",
		help="On-screen MuJoCo viewer",
	)
	render.add_argument(
		"--headless",
		action="store_true",
		help="No on-screen viewer (default)",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--goal-hdf5",
		type=Path,
		default=DEFAULT_HDF5,
		help="HDF5 used to build the goal embedding (last frame of a demo)",
	)
	parser.add_argument(
		"--goal-demo",
		type=str,
		default=None,
		help="Demo key inside the goal HDF5 (default: first train mask entry)",
	)
	args = parser.parse_args()

	gui = bool(args.gui)  # headless unless --gui
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if device == "cuda":
		print(f"using device: cuda ({torch.cuda.get_device_name(0)})")
	else:
		print("using device: cpu")

	evaluate(
		zip_path=args.zip_path,
		rollouts=args.rollouts,
		gui=gui,
		max_steps=args.max_steps,
		seed=args.seed,
		device=device,
		goal_hdf5=args.goal_hdf5,
		goal_demo=args.goal_demo,
	)


if __name__ == "__main__":
	main()
