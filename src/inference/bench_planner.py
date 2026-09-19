"""
Wall-clock timing for a single Planner.CEM call.

Usage:
  PYTHONPATH=src python src/inference/bench_planner.py models/primary_run1/models/epoch_9_model.zip
  PYTHONPATH=src python src/inference/bench_planner.py path/to/epoch_N_model.zip --repeats 5
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

_repo_rs = str((ROOT / "robosuite").resolve())
sys.path = [p for p in sys.path if str(Path(p).resolve()) != _repo_rs]
sys.path = [p for p in sys.path if p not in ("", str(ROOT), str(ROOT.resolve()))]

from architectures.predictor import PredictorNetwork
from architectures.robot_encoder import RobotEncoder
from inference.mpc_parallel import Planner
from primary_contract import ARCHITECTURE_ID, CHECKPOINT_FORMAT_VERSION

WINDOW = 3
FRAME_SKIP = 5
PRIMITIVE_ACTION_DIM = 7
ACTION_DIM = FRAME_SKIP * PRIMITIVE_ACTION_DIM
EMBED_DIM = 192


def load_models(zip_path: Path, device: str):
	encoder = RobotEncoder().to(device)
	predictor = PredictorNetwork(d_action=ACTION_DIM, window=WINDOW).to(device)
	stats = None

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
		if "state_stats.pth" not in zf.namelist():
			raise ValueError("Checkpoint has no state/action statistics")
		zf.extract("state_stats.pth", tmp)
		stats = torch.load(tmp / "state_stats.pth", map_location="cpu", weights_only=True)
		if stats.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
			raise ValueError(
				f"Benchmark requires checkpoint format {CHECKPOINT_FORMAT_VERSION}"
			)
		if stats.get("architecture_id") != ARCHITECTURE_ID:
			raise ValueError(
				f"Benchmark requires architecture {ARCHITECTURE_ID!r}"
			)

	encoder.eval()
	predictor.eval()
	return encoder, predictor, stats


def main():
	parser = argparse.ArgumentParser(description="Time a single CEM planning step")
	parser.add_argument("zip_path", type=Path, help="Path to epoch_*_model.zip")
	parser.add_argument(
		"--repeats",
		type=int,
		default=3,
		help="Timed CEM calls after one warmup (default 3)",
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help="cuda | cpu (default: cuda if available else cpu)",
	)
	args = parser.parse_args()

	if not args.zip_path.exists():
		raise FileNotFoundError(args.zip_path)

	device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	print(f"device: {device}")
	print(f"checkpoint: {args.zip_path}")

	t0 = time.perf_counter()
	encoder, predictor, stats = load_models(args.zip_path, device)
	print(f"load models: {time.perf_counter() - t0:.2f}s")

	primitive_low = ((stats["action_low"] - stats["action_mean"]) / stats["action_std"]).numpy()
	primitive_high = ((stats["action_high"] - stats["action_mean"]) / stats["action_std"]).numpy()
	planner = Planner(
		device=device,
		enc=encoder,
		pred=predictor,
		action_dim=ACTION_DIM,
		primitive_action_dim=PRIMITIVE_ACTION_DIM,
		action_low=np.tile(primitive_low, FRAME_SKIP),
		action_high=np.tile(primitive_high, FRAME_SKIP),
		raw_action_mean=stats["action_mean"].numpy(),
		raw_action_std=stats["action_std"].numpy(),
	)
	print(
		f"CEM defaults: H={planner.H} N={planner.N} K={planner.K} T={planner.T} "
		f"→ {planner.T * planner.H} batched predictor calls / plan "
		f"(batch size {planner.N})"
	)

	zi = torch.randn(1, EMBED_DIM, device=device)
	a = torch.empty(0, ACTION_DIM, device=device)
	zg = torch.randn(EMBED_DIM, device=device)
	schedules = planner.single_switch_gripper_schedules()

	# Warmup (JIT / allocator / first-call overhead)
	t0 = time.perf_counter()
	_ = planner.CEM(zi, a, zg, gripper_schedules=schedules)
	if device == "cuda":
		torch.cuda.synchronize()
	print(f"warmup CEM: {time.perf_counter() - t0:.2f}s")

	times = []
	for i in range(args.repeats):
		if device == "cuda":
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		action = planner.CEM(zi, a, zg, gripper_schedules=schedules)
		if device == "cuda":
			torch.cuda.synchronize()
		dt = time.perf_counter() - t0
		times.append(dt)
		print(f"CEM {i + 1}/{args.repeats}: {dt:.2f}s  action_shape={tuple(action.shape)}")

	mean_t = sum(times) / len(times)
	print(f"\nmean plan time: {mean_t:.2f}s")
	print(f"decoded execution shape: {planner.denormalize_action_blocks(action).shape}")


if __name__ == "__main__":
	main()
