"""Save initial agent and wrist camera views from the Lift_D0 environment.

Outputs are written to ``src/datasets/robot/views/lift`` by default so the
existing Stack view images are left untouched.

Run from the repository root with the project environment::

    python src/datasets/robot/generate_lift_views.py
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_controller_config

# Registers the MimicGen-compatible Lift_D0 environment.
import mimicgen  # noqa: F401


CAMERAS = ("agentview", "robot0_eye_in_hand")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "views" / "lift"


def set_geom_groups(env, collision: bool = False, visual: bool = True) -> None:
	"""Select visual meshes in the offscreen renderer on MuJoCo 2.3.x."""
	context = env.sim._render_context_offscreen
	if context is None:
		return
	address = context.vopt.geomgroup.__array_interface__["data"][0]
	groups = (ctypes.c_uint8 * 6).from_address(address)
	for index in range(6):
		groups[index] = 0
	groups[0] = int(collision)
	groups[1] = int(visual)


def to_uint8(image: np.ndarray) -> np.ndarray:
	image = np.asarray(image)
	if image.dtype == np.uint8:
		return image
	if image.max(initial=0) <= 1.0:
		image = image * 255.0
	return np.clip(image, 0, 255).astype(np.uint8)


def generate_views(output_dir: Path, size: int, seed: int) -> list[Path]:
	if size < 1:
		raise ValueError(f"size must be positive, got {size}")

	output_dir.mkdir(parents=True, exist_ok=True)
	np.random.seed(seed)
	macros.IMAGE_CONVENTION = "opencv"
	controller = load_controller_config(default_controller="OSC_POSE")

	env = suite.make(
		"Lift_D0",
		robots="Panda",
		controller_configs=controller,
		has_renderer=False,
		has_offscreen_renderer=True,
		render_collision_mesh=False,
		render_visual_mesh=True,
		use_camera_obs=True,
		camera_names=list(CAMERAS),
		camera_heights=size,
		camera_widths=size,
		control_freq=20,
		horizon=10,
		ignore_done=True,
	)

	try:
		env.reset()
		set_geom_groups(env)
		env.sim.forward()
		# Render again after selecting visual geoms so the saved observation uses
		# the same visual-only camera setup as collection and inference.
		observation = env._get_observations(force_update=True)

		paths = []
		for camera in CAMERAS:
			key = f"{camera}_image"
			if key not in observation:
				raise KeyError(f"Environment observation is missing {key!r}")
			path = output_dir / f"{camera}.png"
			imageio.imwrite(path, to_uint8(observation[key]))
			paths.append(path)
			print(f"wrote {path} ({size}x{size})")
		return paths
	finally:
		env.close()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--size", type=int, default=256)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	generate_views(args.output_dir, args.size, args.seed)


if __name__ == "__main__":
	main()
