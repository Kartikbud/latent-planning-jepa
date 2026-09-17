"""Repack a MimicGen HDF5 for fast random window reads.

MimicGen's default gzip chunking can span dozens of timesteps while splitting
each image spatially. Training reads short temporal windows, so that layout
decompresses much more data than is returned. This script preserves every
group, dataset, attribute, mask, and value while rewriting camera observations
as complete-image chunks spanning a small number of frames.

The original file is never modified. The default command writes
``demo_rechunked.hdf5`` next to ``demo.hdf5``::

    python src/datasets/robot/rechunking.py

An interrupted conversion leaves a ``.partial`` file, which the next run
removes after checking that it belongs to the requested output.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TASK_PATHS = {
	"stack": {
		"input": ROOT / "data" / "robot" / "mimicgen" / "demo.hdf5",
		"output": ROOT / "data" / "robot" / "mimicgen" / "demo_rechunked.hdf5",
	},
	"lift": {
		"input": ROOT / "data" / "robot" / "mimicgen" / "demo.hdf5",
		"output": ROOT / "data" / "robot" / "mimicgen" / "demo_rechunked.hdf5",
	},
}
DEFAULT_IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def _copy_attrs(source, destination) -> None:
	for key, value in source.attrs.items():
		destination.attrs[key] = value


def _copy_image_dataset(
	source: h5py.Dataset,
	destination_group: h5py.Group,
	name: str,
	chunk_frames: int,
	compression_level: int,
	block_frames: int,
) -> None:
	if source.ndim != 4:
		raise ValueError(f"Expected a THWC image dataset, got {source.name} {source.shape}")
	if source.shape[0] == 0:
		source.file.copy(source, destination_group, name=name)
		return

	chunks = (min(chunk_frames, source.shape[0]), *source.shape[1:])
	destination = destination_group.create_dataset(
		name,
		shape=source.shape,
		dtype=source.dtype,
		chunks=chunks,
		compression="gzip",
		compression_opts=compression_level,
	)
	_copy_attrs(source, destination)

	for start in range(0, source.shape[0], block_frames):
		end = min(start + block_frames, source.shape[0])
		destination[start:end] = source[start:end]


def _copy_episode(
	source_episode: h5py.Group,
	destination_data: h5py.Group,
	image_keys: set[str],
	chunk_frames: int,
	compression_level: int,
	block_frames: int,
) -> None:
	destination_episode = destination_data.create_group(source_episode.name.rsplit("/", 1)[-1])
	_copy_attrs(source_episode, destination_episode)

	for name, item in source_episode.items():
		if name != "obs":
			source_episode.file.copy(item, destination_episode, name=name)
			continue

		destination_obs = destination_episode.create_group("obs")
		_copy_attrs(item, destination_obs)
		for obs_name, observation in item.items():
			if obs_name in image_keys:
				_copy_image_dataset(
					observation,
					destination_obs,
					obs_name,
					chunk_frames,
					compression_level,
					block_frames,
				)
			else:
				source_episode.file.copy(observation, destination_obs, name=obs_name)


def _dataset_paths(group: h5py.Group) -> dict[str, tuple[tuple[int, ...], str]]:
	paths = {}

	def visit(name, item):
		if isinstance(item, h5py.Dataset):
			paths[name] = (item.shape, item.dtype.str)

	group.visititems(visit)
	return paths


def verify_rechunked(
	input_path: Path,
	output_path: Path,
	image_keys: set[str],
	chunk_frames: int,
) -> None:
	"""Verify structure and representative image values in every episode."""
	with h5py.File(input_path, "r") as source, h5py.File(output_path, "r") as destination:
		if set(source.keys()) != set(destination.keys()):
			raise ValueError("Top-level HDF5 groups differ after rechunking")
		if set(source["data"].keys()) != set(destination["data"].keys()):
			raise ValueError("Episode names differ after rechunking")

		for mask_name in source.get("mask", {}):
			if not np.array_equal(source[f"mask/{mask_name}"][:], destination[f"mask/{mask_name}"][:]):
				raise ValueError(f"mask/{mask_name} changed during rechunking")

		episode_names = list(source["data"].keys())
		for index, episode_name in enumerate(episode_names, start=1):
			source_episode = source[f"data/{episode_name}"]
			destination_episode = destination[f"data/{episode_name}"]
			if _dataset_paths(source_episode) != _dataset_paths(destination_episode):
				raise ValueError(f"Dataset structure differs for {episode_name}")

			for image_key in image_keys:
				source_image = source_episode[f"obs/{image_key}"]
				destination_image = destination_episode[f"obs/{image_key}"]
				expected_chunks = (
					min(chunk_frames, source_image.shape[0]),
					*source_image.shape[1:],
				)
				if destination_image.chunks != expected_chunks:
					raise ValueError(
						f"Unexpected chunks for {destination_image.name}: "
						f"{destination_image.chunks} != {expected_chunks}"
					)
				sample_indices = sorted({0, source_image.shape[0] // 2, source_image.shape[0] - 1})
				if not np.array_equal(source_image[sample_indices], destination_image[sample_indices]):
					raise ValueError(f"Image values changed in {destination_image.name}")

			if index % 1000 == 0 or index == len(episode_names):
				print(f"verified {index:,}/{len(episode_names):,} episodes", flush=True)


def rechunk_dataset(
	input_path: Path,
	output_path: Path,
	image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS,
	chunk_frames: int = 8,
	compression_level: int = 4,
	block_frames: int = 256,
	overwrite: bool = False,
	verify: bool = True,
) -> Path:
	input_path = input_path.expanduser().resolve()
	output_path = output_path.expanduser().resolve()
	partial_path = output_path.with_name(output_path.name + ".partial")

	if not input_path.is_file():
		raise FileNotFoundError(f"Input HDF5 not found: {input_path}")
	if input_path == output_path:
		raise ValueError("Input and output paths must be different")
	if chunk_frames < 1 or block_frames < 1:
		raise ValueError("chunk_frames and block_frames must be positive")
	if not 0 <= compression_level <= 9:
		raise ValueError("compression_level must be between 0 and 9")
	if output_path.exists() and not overwrite:
		raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	free_bytes = shutil.disk_usage(output_path.parent).free
	estimated_bytes = int(input_path.stat().st_size * 1.15)
	if free_bytes < estimated_bytes:
		raise OSError(
			f"Not enough free space: approximately {estimated_bytes / 1e9:.1f} GB "
			f"needed, {free_bytes / 1e9:.1f} GB available"
		)

	if partial_path.exists():
		partial_path.unlink()

	image_key_set = set(image_keys)
	started = time.perf_counter()
	try:
		with h5py.File(input_path, "r") as source, h5py.File(partial_path, "w") as destination:
			_copy_attrs(source, destination)
			for name, item in source.items():
				if name != "data":
					source.copy(item, destination, name=name)

			destination_data = destination.create_group("data")
			_copy_attrs(source["data"], destination_data)
			episode_names = list(source["data"].keys())
			total = len(episode_names)

			for index, episode_name in enumerate(episode_names, start=1):
				_copy_episode(
					source[f"data/{episode_name}"],
					destination_data,
					image_key_set,
					chunk_frames,
					compression_level,
					block_frames,
				)
				if index % 100 == 0 or index == total:
					elapsed = time.perf_counter() - started
					rate = index / elapsed
					eta_minutes = (total - index) / max(rate, 1e-9) / 60
					print(
						f"rechunked {index:,}/{total:,} episodes "
						f"({rate:.1f} episodes/s, ETA {eta_minutes:.1f} min)",
						flush=True,
					)
			destination.flush()

		if verify:
			print("conversion complete; verifying structure and sampled values...", flush=True)
			verify_rechunked(input_path, partial_path, image_key_set, chunk_frames)

		if output_path.exists():
			output_path.unlink()
		os.replace(partial_path, output_path)
	except BaseException:
		print(f"conversion failed; partial output remains at {partial_path}", flush=True)
		raise

	elapsed_minutes = (time.perf_counter() - started) / 60
	print(f"input:  {input_path} ({input_path.stat().st_size / 1e9:.2f} GB)")
	print(f"output: {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")
	print(f"finished in {elapsed_minutes:.1f} minutes")
	return output_path


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--task",
		choices=TASK_PATHS,
		default="stack",
		help="Task whose default input and output paths should be used (default: stack)",
	)
	parser.add_argument("--input", type=Path, default=None)
	parser.add_argument("--output", type=Path, default=None)
	parser.add_argument("--chunk-frames", type=int, default=8)
	parser.add_argument("--compression-level", type=int, default=4)
	parser.add_argument("--block-frames", type=int, default=256)
	parser.add_argument(
		"--image-key",
		action="append",
		dest="image_keys",
		help="Observation image key to rechunk; repeat for multiple keys",
	)
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--skip-verify", action="store_true")
	args = parser.parse_args()
	paths = TASK_PATHS[args.task]

	rechunk_dataset(
		input_path=args.input or paths["input"],
		output_path=args.output or paths["output"],
		image_keys=tuple(args.image_keys or DEFAULT_IMAGE_KEYS),
		chunk_frames=args.chunk_frames,
		compression_level=args.compression_level,
		block_frames=args.block_frames,
		overwrite=args.overwrite,
		verify=not args.skip_verify,
	)


if __name__ == "__main__":
	main()
