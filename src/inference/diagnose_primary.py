"""Check exact playback, multistep prediction, and action ranking for primary v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from inference.mpc_parallel import Planner
from inference.run_primary import (
	EMBEDDING_DIM,
	FRAME_SKIP,
	HORIZON,
	PREDICTOR_ACTION_DIM,
	PRIMITIVE_ACTION_DIM,
	ROOT,
	WINDOW,
	_text,
	encode_demo_frames,
	encode_observation,
	load_models,
	make_env,
	reset_to_eval_state,
)


DEFAULT_HDF5 = ROOT / "data/robot/mimicgen/evals.hdf5"


def select_demo(file, requested):
	if requested is not None:
		return requested
	if "mask/eval" in file:
		return _text(file["mask/eval"][0])
	return sorted(file["data"].keys())[0]


def entry_from_demo(name, demo):
	return {
		"name": name,
		"model_xml": _text(demo.attrs["model_file"]),
		"initial_state": np.asarray(demo["states"][0]).copy(),
	}


@torch.no_grad()
def playback_diagnostic(env, encoder, demo, name, device, stats):
	entry = entry_from_demo(name, demo)
	obs = reset_to_eval_state(env, entry)
	stored_start = encode_demo_frames(encoder, demo["obs"], [0], device, stats)[0]
	restored_start = encode_observation(encoder, obs, device, stats)
	pixel_mae = np.mean(np.abs(
		np.asarray(obs["agentview_image"], dtype=np.float32)
		- np.asarray(demo["obs/agentview_image"][0], dtype=np.float32)
	))
	state_errors = []
	for index, action in enumerate(demo["actions"][:]):
		env.step(action)
		actual = np.asarray(env.sim.get_state().flatten())
		expected = np.asarray(demo["states"][index + 1])
		state_errors.append(np.sqrt(np.mean((actual - expected) ** 2)))
	print("\nSimulator playback")
	print(f"  initial pixel MAE:     {pixel_mae:.4f} / 255")
	print(f"  initial embedding MSE: {(stored_start-restored_start).square().mean().item():.6f}")
	print(f"  final state RMSE:      {state_errors[-1]:.6e}")
	print(f"  task success:          {bool(env._check_success())}")


@torch.no_grad()
def model_diagnostic(encoder, predictor, planner, demo, device, stats, random_samples):
	raw_actions = np.asarray(demo["actions"][:], dtype=np.float32)
	num_transitions = len(raw_actions) // FRAME_SKIP
	frame_indices = [index * FRAME_SKIP for index in range(num_transitions + 1)]
	embeddings = encode_demo_frames(encoder, demo["obs"], frame_indices, device, stats)
	normalized = planner.normalize_primitive_actions(raw_actions[:num_transitions * FRAME_SKIP])
	blocks = torch.from_numpy(normalized.reshape(num_transitions, -1)).to(device)

	teacher_losses = []
	for start in range(max(0, num_transitions - WINDOW)):
		predicted = predictor(
			embeddings[start:start + WINDOW].unsqueeze(0),
			blocks[start:start + WINDOW].unsqueeze(0),
		)[0, -1]
		teacher_losses.append((predicted - embeddings[start + WINDOW]).square().mean())
	print("\nWorld model")
	if teacher_losses:
		print(f"  teacher-forced one-step MSE: {torch.stack(teacher_losses).mean().item():.6f}")

	for horizon in (1, 2, 3, 5):
		if horizon > num_transitions:
			continue
		costs = []
		for start in np.linspace(0, num_transitions - horizon, min(32, num_transitions - horizon + 1), dtype=int):
			context_start = max(0, start - (WINDOW - 1))
			zi = embeddings[context_start:start + 1]
			history = blocks[context_start:start]
			costs.append(planner.score_actions(
				zi, history, blocks[start:start + horizon], embeddings[start + horizon], apply_bounds=False
			)[0])
		print(f"  {horizon:>2}-step free-rollout MSE: {torch.stack(costs).mean().item():.6f}")

	rank_horizon = min(HORIZON, num_transitions)
	true_plan = blocks[:rank_horizon].cpu().numpy()
	random_plans = planner.sampler.normal(
		planner.action_mean, planner.action_std,
		size=(random_samples, rank_horizon, PREDICTOR_ACTION_DIM),
	).astype(np.float32)
	random_plans = planner._bound_actions(torch.from_numpy(random_plans).to(device)).cpu().numpy()
	plans = np.concatenate((true_plan[None], random_plans))
	costs = planner.score_actions(
		embeddings[:1], torch.empty(0, PREDICTOR_ACTION_DIM, device=device),
		plans, embeddings[rank_horizon], apply_bounds=False,
	).cpu().numpy()
	percentile = 100 * np.mean(costs[1:] < costs[0])
	print("\nLocal action ranking")
	print(f"  demonstrated cost: {costs[0]:.6f}")
	print(f"  random median:     {np.median(costs[1:]):.6f}")
	print(f"  true-plan percentile: {percentile:.1f}% (0% is best)")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("checkpoint", type=Path)
	parser.add_argument("--eval-hdf5", type=Path, default=DEFAULT_HDF5)
	parser.add_argument("--demo")
	parser.add_argument("--device")
	parser.add_argument("--random-samples", type=int, default=300)
	args = parser.parse_args()
	device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	encoder, predictor, stats = load_models(args.checkpoint, device)
	low = ((stats["action_low"] - stats["action_mean"]) / stats["action_std"]).numpy()
	high = ((stats["action_high"] - stats["action_mean"]) / stats["action_std"]).numpy()
	planner = Planner(
		device, encoder, predictor, H=HORIZON, N=args.random_samples,
		K=min(30, args.random_samples), T=10, action_dim=PREDICTOR_ACTION_DIM,
		primitive_action_dim=PRIMITIVE_ACTION_DIM,
		action_low=np.tile(low, FRAME_SKIP), action_high=np.tile(high, FRAME_SKIP),
		raw_action_mean=stats["action_mean"].numpy(),
		raw_action_std=stats["action_std"].numpy(), seed=42,
	)
	with h5py.File(args.eval_hdf5, "r") as file:
		name = select_demo(file, args.demo)
		demo = file[f"data/{name}"]
		env = make_env(int(demo["actions"].shape[0]) + 10)
		try:
			playback_diagnostic(env, encoder, demo, name, device, stats)
		finally:
			env.close()
		model_diagnostic(encoder, predictor, planner, demo, device, stats, args.random_samples)


if __name__ == "__main__":
	main()
