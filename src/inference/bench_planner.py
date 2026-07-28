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
from inference.mpc import Planner

WINDOW = 3
ACTION_DIM = 7
EMBED_DIM = 192


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
	encoder, predictor = load_models(args.zip_path, device)
	print(f"load models: {time.perf_counter() - t0:.2f}s")

	planner = Planner(device=device, enc=encoder, pred=predictor)
	print(
		f"CEM defaults: H={planner.H} N={planner.N} K={planner.K} T={planner.T} "
		f"→ ~{planner.T * planner.N * planner.H} predictor forwards / plan"
	)

	zi = torch.randn(WINDOW, EMBED_DIM, device=device)
	a = torch.randn(WINDOW - 1, ACTION_DIM, device=device)
	zg = torch.randn(EMBED_DIM, device=device)

	# Warmup (JIT / allocator / first-call overhead)
	t0 = time.perf_counter()
	_ = planner.CEM(zi, a, zg)
	if device == "cuda":
		torch.cuda.synchronize()
	print(f"warmup CEM: {time.perf_counter() - t0:.2f}s")

	times = []
	for i in range(args.repeats):
		if device == "cuda":
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		action = planner.CEM(zi, a, zg)
		if device == "cuda":
			torch.cuda.synchronize()
		dt = time.perf_counter() - t0
		times.append(dt)
		print(f"CEM {i + 1}/{args.repeats}: {dt:.2f}s  action_shape={tuple(action.shape)}")

	mean_t = sum(times) / len(times)
	print(f"\nmean plan time: {mean_t:.2f}s")
	print(f"≈ {400 * mean_t / 60:.1f} min for a 400-step rollout (plan every step)")


if __name__ == "__main__":
	main()
