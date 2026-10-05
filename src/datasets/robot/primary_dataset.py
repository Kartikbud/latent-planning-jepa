import re
from bisect import bisect_right
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

"""
PyTorch dataset over MimicGen Stack demos (robomimic-style HDF5).

Each sample contains a strided observation sequence and one ordered action block for
each transition:

  observations: [o_t, o_{t+s}, ..., o_{t + window*s}]
  actions:      [[a_t, ..., a_{t+s-1}], ...]

Here ``s`` is ``frame_skip``. Every primitive command is retained in temporal
order; no averaging or accumulation is performed. Proprio
state is eef_pos (3) + eef_quat (4) + gripper_qpos (2) = 9D. Images are
returned as uint8 CHW tensors and ImageNet-normalized on the training device.

Train vs val is selected via the HDF5 mask group (`mask/train`, `mask/valid`).
Both splits use state statistics computed only from the training demonstrations.
"""


def _demo_number(name: str) -> int:
	return int(re.search(r"(\d+)", name).group(1))


def _decode_demo_name(raw) -> str:
	if isinstance(raw, bytes):
		return raw.decode("utf-8")
	return str(raw)


def build_robot_state(pos, quat, grip):
	"""Concatenate normalized robosuite proprioception tensors."""
	quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
	# Preserve robosuite's temporally continuous quaternion sign. Forcing a
	# fixed hemisphere creates artificial jumps when a trajectory crosses w=0.
	return torch.cat((pos, quat, grip), dim=-1)


def normalize_robot_state(state, state_mean, state_std):
	return (state - state_mean) / state_std


def normalize_robot_images(images):
	"""Apply the ImageNet normalization used by the released LeWM encoder."""
	if images.shape[-3] != 3:
		raise ValueError(f"expected RGB images, got shape {tuple(images.shape)}")
	shape = [1] * images.ndim
	shape[-3] = 3
	mean = images.new_tensor((0.485, 0.456, 0.406)).reshape(shape)
	std = images.new_tensor((0.229, 0.224, 0.225)).reshape(shape)
	return (images - mean) / std


def compute_train_state_stats(h5_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
	"""Compute 9-D proprioception statistics from training demos only."""
	count = 0
	total = torch.zeros(9, dtype=torch.float64)
	total_sq = torch.zeros(9, dtype=torch.float64)

	with h5py.File(h5_path, "r") as f:
		episodes = [_decode_demo_name(x) for x in f["mask/train"][:]]
		for ep_name in episodes:
			obs = f[f"data/{ep_name}/obs"]
			state = build_robot_state(
				torch.from_numpy(obs["robot0_eef_pos"][:]).to(torch.float64),
				torch.from_numpy(obs["robot0_eef_quat"][:]).to(torch.float64),
				torch.from_numpy(obs["robot0_gripper_qpos"][:]).to(torch.float64),
			)
			count += state.shape[0]
			total += state.sum(dim=0)
			total_sq += state.square().sum(dim=0)

	if count == 0:
		raise ValueError(f"No training observations found in {h5_path}")

	mean = total / count
	variance = (total_sq / count) - mean.square()
	std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
	return mean.float(), std.float()


def compute_train_action_stats(h5_path: Path):
	"""Compute primitive-action statistics and robust bounds from train demos."""
	action_arrays = []
	with h5py.File(h5_path, "r") as f:
		episodes = [_decode_demo_name(x) for x in f["mask/train"][:]]
		for ep_name in episodes:
			actions = np.asarray(f[f"data/{ep_name}/actions"][:], dtype=np.float32)
			if actions.ndim != 2:
				raise ValueError(f"{ep_name} actions must be rank 2, got {actions.shape}")
			action_arrays.append(actions)

	if not action_arrays:
		raise ValueError(f"No training actions found in {h5_path}")
	actions = np.concatenate(action_arrays, axis=0)
	mean = torch.from_numpy(actions.mean(axis=0, dtype=np.float64).astype(np.float32))
	std = torch.from_numpy(actions.std(axis=0, dtype=np.float64).astype(np.float32))
	std = std.clamp_min(1e-6)
	low = torch.from_numpy(np.quantile(actions, 0.005, axis=0).astype(np.float32))
	high = torch.from_numpy(np.quantile(actions, 0.995, axis=0).astype(np.float32))
	# Zero is a useful no-motion proposal and remains inside robosuite's bounds.
	low = torch.minimum(low, torch.zeros_like(low))
	high = torch.maximum(high, torch.zeros_like(high))
	degenerate = (high - low) < 1e-6
	low = torch.where(degenerate, mean - std, low)
	high = torch.where(degenerate, mean + std, high)
	return mean, std, low, high


class RobotPrimaryDataset(Dataset):
	def __init__(
		self,
		data_dir: Path,
		frame_skip: int,
		window: int,
		val: bool = False,
		state_mean: torch.Tensor | None = None,
		state_std: torch.Tensor | None = None,
		action_mean: torch.Tensor | None = None,
		action_std: torch.Tensor | None = None,
		h5_path: Path | None = None,
	):
		super().__init__()
		if frame_skip < 1:
			raise ValueError(f"frame_skip must be positive, got {frame_skip}")
		if window < 1:
			raise ValueError(f"window must be positive, got {window}")

		self.frame_skip = frame_skip
		self.window = window
		self.h5_path = Path(h5_path) if h5_path is not None else Path(data_dir) / "robot" / "mimicgen" / "demo.hdf5"
		if not self.h5_path.exists():
			raise FileNotFoundError(f"Dataset not found: {self.h5_path}")

		if state_mean is None or state_std is None:
			state_mean, state_std = compute_train_state_stats(self.h5_path)
		self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
		self.state_std = torch.as_tensor(state_std, dtype=torch.float32)
		if self.state_mean.shape != (9,) or self.state_std.shape != (9,):
			raise ValueError("state_mean and state_std must both have shape (9,)")
		if action_mean is None or action_std is None:
			action_mean, action_std, _, _ = compute_train_action_stats(self.h5_path)
		self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
		self.action_std = torch.as_tensor(action_std, dtype=torch.float32)
		if self.action_mean.ndim != 1 or self.action_std.shape != self.action_mean.shape:
			raise ValueError("action_mean and action_std must be matching vectors")

		mask_key = "valid" if val else "train"
		self.episodes = []
		self.cumulative_windows = []
		total_windows = 0
		with h5py.File(self.h5_path, "r") as f:
			episodes = [_decode_demo_name(x) for x in f["mask"][mask_key][:]]
			episodes = sorted(episodes, key=_demo_number)

			for ep_name in episodes:
				ep_len = int(
					f[f"data/{ep_name}"].attrs.get(
						"num_samples",
						f[f"data/{ep_name}/actions"].shape[0],
					)
				)
				num_windows = max(0, ep_len - window * frame_skip)
				if num_windows:
					total_windows += num_windows
					self.episodes.append(ep_name)
					self.cumulative_windows.append(total_windows)

		# Keep one cumulative count per episode instead of one Python tuple and
		# frame list per window. This stays small even for millions of windows.
		self.total_len = total_windows
		self._h5 = None

	def __len__(self):
		return self.total_len

	def _get_h5(self):
		# DataLoader workers each lazily open their own read-only handle.
		if self._h5 is None:
			self._h5 = h5py.File(self.h5_path, "r")
		return self._h5

	def _resolve_index(self, idx):
		if idx < 0:
			idx += self.total_len
		if idx < 0 or idx >= self.total_len:
			raise IndexError(idx)

		episode_idx = bisect_right(self.cumulative_windows, idx)
		episode_start = (
			0 if episode_idx == 0 else self.cumulative_windows[episode_idx - 1]
		)
		return self.episodes[episode_idx], idx - episode_start

	def __getstate__(self):
		state = self.__dict__.copy()
		state["_h5"] = None
		return state

	def close(self):
		if getattr(self, "_h5", None) is not None:
			self._h5.close()
			self._h5 = None

	def __del__(self):
		self.close()

	def _load_images(self, obs, frame_slice, cam_key):
		# HDF5 stepped hyperslabs are several times slower than reading the
		# already-touched temporal chunks contiguously and subsampling in memory.
		imgs = torch.from_numpy(
			obs[cam_key][frame_slice.start:frame_slice.stop][::frame_slice.step]
		)
		return imgs.permute(0, 3, 1, 2)

	def _load_state(self, obs, frame_slice):
		contiguous_slice = slice(frame_slice.start, frame_slice.stop)
		step = frame_slice.step
		state = build_robot_state(
			torch.from_numpy(obs["robot0_eef_pos"][contiguous_slice][::step]).float(),
			torch.from_numpy(obs["robot0_eef_quat"][contiguous_slice][::step]).float(),
			torch.from_numpy(obs["robot0_gripper_qpos"][contiguous_slice][::step]).float(),
		)
		return normalize_robot_state(state, self.state_mean, self.state_std)

	def __getitem__(self, idx):
		ep_name, start = self._resolve_index(idx)
		f = self._get_h5()
		episode = f["data"][ep_name]
		obs = episode["obs"]
		frame_slice = slice(
			start,
			start + self.window * self.frame_skip + 1,
			self.frame_skip,
		)

		above = self._load_images(obs, frame_slice, "agentview_image")
		wrist = self._load_images(obs, frame_slice, "robot0_eye_in_hand_image")
		state = self._load_state(obs, frame_slice)

		# Each transition keeps all frame_skip primitive OSC commands in order.
		primitive_actions = torch.from_numpy(
			episode["actions"][
				start:start + self.window * self.frame_skip
			]
		).float().reshape(self.window, self.frame_skip, -1)
		if primitive_actions.shape[-1] != self.action_mean.numel():
			raise ValueError(
				f"dataset actions have size {primitive_actions.shape[-1]}, "
				f"statistics have size {self.action_mean.numel()}"
			)
		actions = (primitive_actions - self.action_mean) / self.action_std
		return above, wrist, state, actions
