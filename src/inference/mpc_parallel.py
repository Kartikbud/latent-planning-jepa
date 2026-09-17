"""Batched cross-entropy-method planning in standardized action space."""

import numpy as np
import torch
from torch import nn


class Planner:
	def __init__(
		self,
		device,
		enc: nn.Module,
		pred: nn.Module,
		H: int = 5,
		N: int = 300,
		K: int = 30,
		T: int = 10,
		action_dim: int = 35,
		primitive_action_dim: int = 7,
		action_mean=None,
		action_std=None,
		action_low=None,
		action_high=None,
		raw_action_mean=None,
		raw_action_std=None,
		min_std=0.05,
		momentum: float = 0.1,
		seed: int | None = None,
	):
		self.device = torch.device(device)
		self.enc = enc.to(self.device)
		self.pred = pred.to(self.device)
		self.H, self.N, self.K, self.T = H, N, K, T
		self.action_dim = action_dim
		self.primitive_action_dim = primitive_action_dim
		if min(H, N, K, T, action_dim, primitive_action_dim) < 1:
			raise ValueError("planner dimensions and CEM settings must be positive")
		if K > N:
			raise ValueError(f"K ({K}) cannot exceed N ({N})")
		if action_dim % primitive_action_dim:
			raise ValueError("action_dim must contain complete primitive-action blocks")
		self.block_size = action_dim // primitive_action_dim
		self.gripper_indices = np.arange(
			primitive_action_dim - 1, action_dim, primitive_action_dim
		)

		# CEM works in the exact standardized coordinates consumed by the model.
		self.action_mean = self._vector(action_mean, 0.0, action_dim, "action_mean")
		self.action_std = self._vector(action_std, 1.0, action_dim, "action_std")
		self.min_std = self._vector(min_std, 0.05, action_dim, "min_std")
		self.action_low = self._optional_vector(action_low, action_dim, "action_low")
		self.action_high = self._optional_vector(action_high, action_dim, "action_high")
		if np.any(self.action_std <= 0) or np.any(self.min_std <= 0):
			raise ValueError("action_std and min_std must be positive")
		if (self.action_low is None) != (self.action_high is None):
			raise ValueError("action_low and action_high must be provided together")
		if self.action_low is not None:
			if np.any(self.action_low >= self.action_high):
				raise ValueError("Every action_low must be below action_high")
			self.action_mean = np.clip(self.action_mean, self.action_low, self.action_high)

		self.raw_action_mean = self._vector(
			raw_action_mean, 0.0, primitive_action_dim, "raw_action_mean"
		)
		self.raw_action_std = self._vector(
			raw_action_std, 1.0, primitive_action_dim, "raw_action_std"
		)
		if np.any(self.raw_action_std <= 0):
			raise ValueError("raw_action_std must be positive")
		if not 0.0 <= momentum < 1.0:
			raise ValueError(f"momentum must be in [0, 1), got {momentum}")
		self.momentum = float(momentum)
		self.sampler = np.random.default_rng(seed)

	@staticmethod
	def _vector(value, default, size, name):
		array = np.asarray(default if value is None else value, dtype=np.float32)
		if array.ndim == 0:
			array = np.full(size, float(array), dtype=np.float32)
		if array.shape != (size,) or not np.isfinite(array).all():
			raise ValueError(f"{name} must be finite and have shape ({size},)")
		return array.copy()

	@classmethod
	def _optional_vector(cls, value, size, name):
		return None if value is None else cls._vector(value, 0.0, size, name)

	def normalize_primitive_actions(self, actions):
		actions = np.asarray(actions, dtype=np.float32)
		if actions.shape[-1] != self.primitive_action_dim:
			raise ValueError("primitive action dimension does not match planner")
		return (actions - self.raw_action_mean) / self.raw_action_std

	def denormalize_action_blocks(self, blocks):
		blocks = np.asarray(blocks, dtype=np.float32)
		if blocks.shape[-1] != self.action_dim:
			raise ValueError("flattened action block dimension does not match planner")
		primitive = blocks.reshape(*blocks.shape[:-1], self.block_size, self.primitive_action_dim)
		return primitive * self.raw_action_std + self.raw_action_mean

	def single_switch_gripper_schedules(self):
		"""Return coherent gripper schedules with at most one block-boundary switch."""
		total_steps = self.H * self.block_size
		schedules = []
		for start in (-1.0, 1.0):
			schedules.append(np.full(total_steps, start, dtype=np.float32))
			for switch in range(self.block_size, total_steps, self.block_size):
				schedule = np.full(total_steps, start, dtype=np.float32)
				schedule[switch:] = -start
				schedules.append(schedule)
		raw = np.stack(schedules).reshape(-1, self.H, self.block_size)
		return (raw - self.raw_action_mean[-1]) / self.raw_action_std[-1]

	def shifted_plan(self, plan):
		plan = np.asarray(plan, dtype=np.float32)
		if plan.shape != (self.H, self.action_dim):
			raise ValueError(f"plan must have shape {(self.H, self.action_dim)}")
		return np.concatenate((plan[1:], self.action_mean[None]), axis=0)

	def _bound_actions(self, actions):
		if self.action_low is None:
			return actions
		low = torch.as_tensor(self.action_low, dtype=actions.dtype, device=actions.device)
		high = torch.as_tensor(self.action_high, dtype=actions.dtype, device=actions.device)
		return torch.maximum(torch.minimum(actions, high), low)

	def _expand_gripper_schedules(self, schedules, device):
		schedules = torch.as_tensor(schedules, dtype=torch.float32, device=device)
		if schedules.ndim == 2 and schedules.shape == (self.H, self.block_size):
			schedules = schedules.unsqueeze(0)
		if schedules.ndim != 3 or schedules.shape[1:] != (self.H, self.block_size):
			raise ValueError(
				"gripper_schedules must have shape "
				f"(S, {self.H}, {self.block_size}), got {tuple(schedules.shape)}"
			)
		return schedules

	def _prepare_inputs(self, zi, a, zg):
		zi = torch.as_tensor(zi, dtype=torch.float32, device=self.device)
		a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
		zg = torch.as_tensor(zg, dtype=torch.float32, device=self.device)
		if zi.ndim != 2:
			raise ValueError(f"zi must have shape (context, D), got {tuple(zi.shape)}")
		if a.ndim != 2 or a.shape != (zi.shape[0] - 1, self.action_dim):
			raise ValueError(
				f"a must have shape ({zi.shape[0] - 1}, {self.action_dim}), got {tuple(a.shape)}"
			)
		if zg.ndim == 1:
			zg = zg.unsqueeze(0)
		if zg.ndim != 2 or zg.shape[-1] != zi.shape[-1]:
			raise ValueError("zg must contain one or more embeddings matching zi")
		return zi, a, zg

	@torch.no_grad()
	def score_actions(self, zi, a, actions, zg, apply_bounds=True):
		"""Return terminal latent MSE for each standardized action sequence."""
		zi, a, zg = self._prepare_inputs(zi, a, zg)
		actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
		if actions.ndim == 2:
			actions = actions.unsqueeze(0)
		if actions.ndim != 3 or actions.shape[2] != self.action_dim:
			raise ValueError("actions must have shape (H, A) or (N, H, A)")
		if apply_bounds:
			actions = self._bound_actions(actions)

		num_sequences = actions.shape[0]
		window = zi.unsqueeze(0).expand(num_sequences, -1, -1)
		action_history = a.unsqueeze(0).expand(num_sequences, -1, -1)
		max_context = int(getattr(self.pred, "window", zi.shape[0]))
		for step in range(actions.shape[1]):
			model_actions = torch.cat((action_history, actions[:, step:step + 1]), dim=1)
			prediction = self.pred(window, model_actions)[:, -1]
			window = torch.cat((window, prediction.unsqueeze(1)), dim=1)
			action_history = model_actions
			if window.shape[1] > max_context:
				window = window[:, -max_context:]
				action_history = action_history[:, -(max_context - 1):]

		goal_costs = (window[:, -1, None] - zg[None]).square().mean(dim=-1)
		return goal_costs.min(dim=1).values

	@torch.no_grad()
	def CEM(self, zi, a, zg, initial_mean=None, gripper_schedules=None):
		"""Optimize and return a complete standardized ``H x action_dim`` plan."""
		zi, a, zg = self._prepare_inputs(zi, a, zg)
		self.pred.eval()
		self.enc.eval()
		base_mean = np.broadcast_to(self.action_mean, (self.H, self.action_dim)).copy()
		if initial_mean is not None:
			base_mean = np.asarray(initial_mean, dtype=np.float32).copy()
			if base_mean.shape != (self.H, self.action_dim):
				raise ValueError(f"initial_mean must have shape {(self.H, self.action_dim)}")
		if self.action_low is not None:
			base_mean = np.clip(base_mean, self.action_low, self.action_high)
		base_std = np.broadcast_to(self.action_std, base_mean.shape).copy()

		schedules = (
			None if gripper_schedules is None
			else self._expand_gripper_schedules(gripper_schedules, self.device)
		)
		mean = base_mean
		std = base_std
		for _ in range(self.T):
			samples = self.sampler.normal(
				mean, std, size=(self.N, self.H, self.action_dim),
			)
			actions = torch.as_tensor(samples, dtype=torch.float32, device=self.device)
			actions = self._bound_actions(actions)
			if schedules is not None:
				# Give all N candidates coherent schedules while retaining the
				# original global N/K CEM update for the continuous motion values.
				schedule_indices = torch.as_tensor(
					self.sampler.permutation(self.N) % schedules.shape[0],
					device=self.device,
				)
				actions[..., self.gripper_indices] = schedules[schedule_indices]
			costs = self.score_actions(zi, a, actions, zg, apply_bounds=False)
			elite_indices = torch.topk(costs, k=self.K, largest=False).indices
			elites = actions[elite_indices]
			elite_mean = elites.mean(dim=0).cpu().numpy()
			elite_std = elites.std(dim=0, unbiased=False).cpu().numpy()
			mean = self.momentum * mean + (1.0 - self.momentum) * elite_mean
			std = np.maximum(
				self.momentum * std + (1.0 - self.momentum) * elite_std,
				self.min_std,
			)

		candidate_means = torch.as_tensor(mean, dtype=torch.float32, device=self.device).unsqueeze(0)
		candidate_means = self._bound_actions(candidate_means)
		if schedules is not None:
			candidate_means = candidate_means.expand(schedules.shape[0], -1, -1).clone()
			candidate_means[..., self.gripper_indices] = schedules
		costs = self.score_actions(zi, a, candidate_means, zg, apply_bounds=False)
		best = candidate_means[costs.argmin()]
		return best.cpu().numpy().astype(np.float32, copy=False)
