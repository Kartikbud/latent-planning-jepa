"""Combine standalone SpaceMouse eval episodes into one ``evals.hdf5``.

Input files are produced by ``collect_eval.py``. The output keeps every full
trajectory and adds an ``mask/eval`` list so inference can iterate through the
exact initial-state / grasp / goal triplets deterministically.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TASK_PATHS = {
	"stack": {
		"input": ROOT / "data" / "robot" / "eval_raw",
		"output": ROOT / "data" / "robot" / "mimicgen" / "evals.hdf5",
	},
	"lift": {
		"input": ROOT / "data" / "robot" / "eval_raw",
		"output": ROOT / "data" / "robot" / "mimicgen" / "evals.hdf5",
	},
}
REQUIRED_OBS = (
	"agentview_image",
	"robot0_eye_in_hand_image",
	"robot0_eef_pos",
	"robot0_eef_quat",
	"robot0_gripper_qpos",
)


def _raw_files(input_dir: Path) -> list[Path]:
	return sorted(input_dir.glob("eval_[0-9][0-9][0-9][0-9].hdf5"))


def validate_episode(path: Path, demo) -> int:
	missing = [key for key in ("states", "actions", "obs") if key not in demo]
	if missing:
		raise ValueError(f"{path}: missing {', '.join(missing)}")

	num_actions = int(demo["actions"].shape[0])
	num_frames = int(demo["states"].shape[0])
	if num_frames != num_actions + 1:
		raise ValueError(
			f"{path}: expected states={num_actions + 1}, got {num_frames}"
		)
	if int(demo.attrs.get("num_samples", -1)) != num_actions:
		raise ValueError(f"{path}: num_samples does not match actions")
	if "model_file" not in demo.attrs:
		raise ValueError(f"{path}: missing model_file needed for exact reset")

	initial_index = int(demo.attrs.get("initial_index", -1))
	grasp_index = int(demo.attrs.get("grasp_index", -1))
	goal_index = int(demo.attrs.get("goal_index", -1))
	if initial_index != 0:
		raise ValueError(f"{path}: initial_index must be 0")
	if not (0 < grasp_index < goal_index < num_frames):
		raise ValueError(
			f"{path}: invalid frame indices initial=0, grasp={grasp_index}, "
			f"goal={goal_index}, frames={num_frames}"
		)
	if not bool(demo.attrs.get("success", False)):
		raise ValueError(f"{path}: episode is not marked successful")

	obs = demo["obs"]
	for key in REQUIRED_OBS:
		if key not in obs:
			raise ValueError(f"{path}: missing obs/{key}")
		if obs[key].shape[0] != num_frames:
			raise ValueError(
				f"{path}: obs/{key} has {obs[key].shape[0]} frames, "
				f"expected {num_frames}"
			)
	return num_actions


def combine(input_dir: Path, output_path: Path, overwrite: bool = False) -> Path:
	files = _raw_files(input_dir)
	if not files:
		raise FileNotFoundError(f"No eval_####.hdf5 files found in {input_dir}")
	if output_path.exists() and not overwrite:
		raise FileExistsError(
			f"{output_path} already exists; pass --overwrite to replace it"
		)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	partial = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
	if partial.exists():
		raise FileExistsError(f"Temporary output already exists: {partial}")

	total_actions = 0
	demo_names = []
	first_data_attrs = None
	try:
		with h5py.File(partial, "w") as output:
			output_data = output.create_group("data")
			for index, path in enumerate(files):
				with h5py.File(path, "r") as source:
					if "data/demo_0" not in source:
						raise ValueError(f"{path}: expected data/demo_0")
					source_demo = source["data/demo_0"]
					total_actions += validate_episode(path, source_demo)
					demo_name = f"demo_{index}"
					source.copy(source_demo, output_data, name=demo_name)
					output_data[demo_name].attrs["source_file"] = path.name
					demo_names.append(demo_name)
					if first_data_attrs is None:
						first_data_attrs = dict(source["data"].attrs.items())
				print(f"[{index + 1:>3}/{len(files)}] {path.name} -> {demo_name}")

			if first_data_attrs is not None:
				for key, value in first_data_attrs.items():
					output_data.attrs[key] = value
			now = datetime.datetime.now()
			output_data.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
			output_data.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
			output_data.attrs["total"] = total_actions
			output_data.attrs["num_demos"] = len(demo_names)
			output_data.attrs["dataset_type"] = "exact_reset_eval"

			mask = output.create_group("mask")
			string_type = h5py.string_dtype(encoding="utf-8")
			mask.create_dataset(
				"eval",
				data=np.asarray(demo_names, dtype=object),
				dtype=string_type,
			)
			output.flush()

		os.replace(partial, output_path)
	except Exception:
		if partial.exists():
			partial.unlink()
		raise

	print(
		f"\nWrote {len(demo_names)} exact-reset evals ({total_actions} actions) "
		f"to {output_path}"
	)
	return output_path


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Combine collect_eval.py files into one evals.hdf5"
	)
	parser.add_argument(
		"--task",
		choices=TASK_PATHS,
		default="stack",
		help="Task whose default paths should be used (default: stack)",
	)
	parser.add_argument("--input-dir", type=Path, default=None)
	parser.add_argument("--output", type=Path, default=None)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Replace an existing output file",
	)
	args = parser.parse_args()
	paths = TASK_PATHS[args.task]
	input_dir = args.input_dir or paths["input"]
	output = args.output or paths["output"]
	combine(input_dir, output, overwrite=args.overwrite)


if __name__ == "__main__":
	main()
