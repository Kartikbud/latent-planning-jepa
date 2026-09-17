"""Interactively label one grasp subgoal frame for every exact-reset eval.

The HDF5 is opened read-only. Confirmed labels are saved immediately to a
small JSON sidecar, so labeling can be interrupted and resumed safely.

Controls (focus the OpenCV image window):
  A / left arrow    previous frame
  D / right arrow   next frame
  [ / ]             move backward / forward 5 frames
  Home / End         first / final frame
  Space              confirm this grasp frame and advance to the next demo
  S                  skip this demo for the current session
  Q / Escape         quit (already confirmed labels remain saved)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVALS = ROOT / "data" / "robot" / "mimicgen" / "evals.hdf5"
DEFAULT_LABELS = ROOT / "data" / "robot" / "mimicgen" / "subgoal_labels.json"
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
WINDOW_NAME = "Manual Lift subgoal labeling"


def _text(value) -> str:
	return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _demo_number(name: str) -> int:
	match = re.search(r"(\d+)$", name)
	return int(match.group(1)) if match else 10**12


def demo_names(file) -> list[str]:
	if "mask/eval" in file:
		return [_text(value) for value in file["mask/eval"][:]]
	return sorted(file["data"].keys(), key=lambda name: (_demo_number(name), name))


def load_labels(path: Path, source: Path) -> dict:
	if not path.exists():
		return {
			"format_version": 1,
			"source_hdf5": str(source.resolve()),
			"labels": {},
		}
	with path.open("r", encoding="utf-8") as stream:
		data = json.load(stream)
	if data.get("format_version") != 1 or not isinstance(data.get("labels"), dict):
		raise ValueError(f"Unsupported label file format: {path}")
	return data


def save_labels(path: Path, data: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	pending = path.with_suffix(path.suffix + ".tmp")
	with pending.open("w", encoding="utf-8") as stream:
		json.dump(data, stream, indent=2, sort_keys=True)
		stream.write("\n")
	os.replace(pending, path)


def _camera_panel(image, name: str, size: int):
	image = np.asarray(image)
	if image.ndim != 3 or image.shape[-1] != 3:
		raise ValueError(f"{name} must be an RGB image, got {image.shape}")
	if image.dtype != np.uint8:
		image = np.clip(image, 0, 255).astype(np.uint8)
	# HDF5 observations are RGB; OpenCV windows expect BGR.
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
	cv2.putText(
		image, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		(255, 255, 255), 2, cv2.LINE_AA,
	)
	return image


def render_label_view(demo, demo_name: str, demo_position: int, num_demos: int,
					  frame: int, display_size: int):
	obs = demo["obs"]
	panels = [
		_camera_panel(obs[key][frame], key.replace("_image", ""), display_size)
		for key in CAMERAS
	]
	canvas = np.concatenate(panels, axis=1)
	status_height = 112
	canvas = cv2.copyMakeBorder(
		canvas, 0, status_height, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
	)
	num_frames = int(obs[CAMERAS[0]].shape[0])
	auto_grasp = int(demo.attrs.get("grasp_index", -1))
	goal_frame = num_frames - 1
	gripper = np.asarray(obs["robot0_gripper_qpos"][frame]).round(4).tolist()
	lines = (
		f"{demo_name} ({demo_position + 1}/{num_demos})   frame {frame}/{goal_frame}",
		f"collector grasp_index={auto_grasp}   gripper_qpos={gripper}",
		"A/D: +/-1   [ ]: +/-5   Home/End   SPACE: confirm   S: skip   Q: quit",
	)
	for index, line in enumerate(lines):
		cv2.putText(
			canvas, line, (12, display_size + 31 + index * 31),
			cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA,
		)
	return canvas


def next_frame_for_key(key: int, frame: int, final_frame: int) -> int:
	character = key & 0xFF
	# Arrow / Home / End codes vary between OpenCV GUI backends, so letter
	# controls are always available as a portable fallback.
	left_keys = {ord("a"), ord("A"), 2424832, 65361}
	right_keys = {ord("d"), ord("D"), 2555904, 65363}
	home_keys = {ord("h"), ord("H"), 2359296, 65360}
	end_keys = {ord("e"), ord("E"), 2293760, 65367}
	if key in left_keys or character in left_keys:
		return max(0, frame - 1)
	if key in right_keys or character in right_keys:
		return min(final_frame, frame + 1)
	if character == ord("["):
		return max(0, frame - 5)
	if character == ord("]"):
		return min(final_frame, frame + 5)
	if key in home_keys or character in home_keys:
		return 0
	if key in end_keys or character in end_keys:
		return final_frame
	return frame


def label_subgoals(
	evals_path: Path,
	labels_path: Path,
	display_size: int = 420,
	start_demo: str | None = None,
	relabel: bool = False,
) -> dict:
	if not evals_path.exists():
		raise FileNotFoundError(evals_path)
	if display_size < 84:
		raise ValueError("display_size must be at least 84")

	labels = load_labels(labels_path, evals_path)
	with h5py.File(evals_path, "r") as file:
		names = demo_names(file)
		if not names:
			raise ValueError(f"No evaluation demos found in {evals_path}")
		total_demos = len(names)
		if start_demo is not None:
			if start_demo not in names:
				raise KeyError(f"{start_demo!r} is not present in {evals_path}")
			names = names[names.index(start_demo):]

		print(f"evals:  {evals_path}")
		print(f"labels: {labels_path}")
		print("Focus the image window, navigate to a stable pre-lift grasp, then press Space.")
		try:
			for demo_position, name in enumerate(names):
				if name in labels["labels"] and not relabel:
					print(f"[{demo_position + 1:>3}/{len(names)}] {name}: already labeled; skipping")
					continue

				demo = file[f"data/{name}"]
				obs = demo["obs"]
				missing = [key for key in CAMERAS if key not in obs]
				if missing:
					raise KeyError(f"{name} is missing observation cameras: {missing}")
				num_frames = int(obs[CAMERAS[0]].shape[0])
				if any(int(obs[key].shape[0]) != num_frames for key in CAMERAS):
					raise ValueError(f"{name} camera streams have different lengths")
				if num_frames < 3:
					raise ValueError(f"{name} has only {num_frames} frames")

				previous = labels["labels"].get(name, {})
				frame = int(previous.get("grasp_frame", demo.attrs.get("grasp_index", 1)))
				frame = int(np.clip(frame, 0, num_frames - 1))
				while True:
					view = render_label_view(
						demo, name, demo_position, len(names), frame, display_size
					)
					cv2.imshow(WINDOW_NAME, view)
					key = cv2.waitKeyEx(0)
					character = key & 0xFF
					if character == 32:  # Space
						if not 0 < frame < num_frames - 1:
							print(f"{name}: choose a grasp frame between 1 and {num_frames - 2}")
							continue
						labels["labels"][name] = {
							"grasp_frame": frame,
							"goal_frame": num_frames - 1,
							"num_frames": num_frames,
						}
						save_labels(labels_path, labels)
						print(
							f"[{demo_position + 1:>3}/{len(names)}] {name}: "
							f"grasp={frame}, goal={num_frames - 1} saved"
						)
						break
					if character in (ord("q"), ord("Q"), 27):
						print("Stopped. Confirmed labels are already saved.")
						return labels
					if character in (ord("s"), ord("S")):
						print(f"{name}: skipped for this session")
						break
					frame = next_frame_for_key(key, frame, num_frames - 1)
		finally:
			cv2.destroyAllWindows()

	completed = len(labels["labels"])
	print(f"Labeling complete: {completed}/{total_demos} demos saved to {labels_path}")
	return labels


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--evals", type=Path, default=DEFAULT_EVALS)
	parser.add_argument("--output", type=Path, default=DEFAULT_LABELS)
	parser.add_argument("--display-size", type=int, default=420)
	parser.add_argument(
		"--start-demo", type=str, default=None,
		help="Start at this demo name, for example demo_20",
	)
	parser.add_argument(
		"--relabel", action="store_true",
		help="Visit already-labeled demos instead of skipping them",
	)
	args = parser.parse_args()
	label_subgoals(
		args.evals,
		args.output,
		display_size=args.display_size,
		start_demo=args.start_demo,
		relabel=args.relabel,
	)


if __name__ == "__main__":
	main()
