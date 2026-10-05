"""Fixed-block actor-residual CEM, with terminal-only latent goal cost."""
import numpy as np
import torch

from src.inference.mpc_parallel import Planner


class ActorPlanner(Planner):
    def __init__(self, *args, actor, mode='arcem', temperature=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ('arcem', 'direct', 'cem'):
            raise ValueError('mode must be arcem, direct, or cem')
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be nonnegative')
        if self.action_low is None:
            raise ValueError('Actor planning requires action bounds')
        self.actor, self.mode, self.temperature = actor, mode, temperature

    def executable(self, actions):
        """Match raw [-1,1] clipping and binary gripper used by env execution."""
        mean = torch.as_tensor(self.raw_action_mean, device=actions.device, dtype=actions.dtype)
        std = torch.as_tensor(self.raw_action_std, device=actions.device, dtype=actions.dtype)
        raw = (actions * std + mean).clamp(-1, 1)
        raw = torch.cat((raw[..., :-1], torch.where(raw[..., -1:] >= 0, 1., -1.)), -1)
        return (raw - mean) / std

    def _bound_actions(self, actions):
        bounded = super()._bound_actions(actions)
        return self.executable(bounded.reshape(*bounded.shape[:-1], self.block_size,
                                               self.primitive_action_dim)).reshape_as(bounded)

    @torch.no_grad()
    def _roll_candidates(self, zi, history, goal, residuals):
        count = residuals.shape[0]
        states = zi[None].expand(count, -1, -1)
        actions = history[None].expand(count, -1, -1)
        goal = goal.reshape(1, -1).expand(count, -1)
        blocks = []
        for step in range(self.H):
            if actions.shape[1]:
                previous = actions[:, -1].reshape(count, self.block_size, self.primitive_action_dim)
            else:
                zero = (-torch.as_tensor(self.raw_action_mean, device=zi.device) /
                        torch.as_tensor(self.raw_action_std, device=zi.device))
                previous = zero.expand(count, self.block_size, -1)
            prefix = zi.new_empty(count, 0, self.primitive_action_dim)
            for j in range(self.block_size):
                means, _ = self.actor(states[:, -1], goal, previous, prefix)
                proposal = means[:, -1] + self.temperature * residuals[:, step, j]
                low = torch.as_tensor(self.action_low, device=zi.device).reshape(self.block_size, -1)[j]
                high = torch.as_tensor(self.action_high, device=zi.device).reshape(self.block_size, -1)[j]
                proposal = self.executable(proposal.clamp(low, high))
                prefix = torch.cat((prefix, proposal[:, None]), dim=1)
            block = prefix.flatten(1)
            blocks.append(block)
            actions = torch.cat((actions, block[:, None]), dim=1)
            predicted = self.pred(states, actions)[:, -1]
            states = torch.cat((states, predicted[:, None]), dim=1)[:, -self.pred.window:]
            actions = actions[:, -(self.pred.window - 1):]
        cost = (states[:, -1] - goal).square().mean(-1)
        return torch.stack(blocks, dim=1), cost

    @torch.no_grad()
    def CEM(self, zi, a, zg, initial_mean=None, gripper_schedules=None):
        if initial_mean is not None or gripper_schedules is not None:
            raise ValueError('Actor planner does not use warm starts or hand-coded gripper schedules')
        self.actor.eval()
        if self.mode == 'cem':
            return super().CEM(zi, a, zg)
        zi, a, goals = self._prepare_inputs(zi, a, zg)
        if goals.shape[0] != 1:
            raise ValueError('Actor planner expects exactly one current subgoal')
        self.pred.eval()
        self.enc.eval()
        shape = (self.H, self.block_size, self.primitive_action_dim)
        zeros = torch.zeros(1, *shape, device=zi.device)
        direct, cost = self._roll_candidates(zi, a, goals[0], zeros)
        if self.mode == 'direct':
            return direct[0].cpu().numpy()
        best_plan, best_cost = direct[0], cost[0]
        mean, std = np.zeros(shape, np.float32), np.ones(shape, np.float32)
        for _ in range(self.T):
            residuals = self.sampler.normal(size=(self.N, *shape)).astype(np.float32) * std + mean
            residuals[0] = 0  # Exact actor-only candidate in every generation.
            if self.N > 1:
                residuals[1] = mean
            residuals = torch.as_tensor(residuals, device=zi.device)
            plans, costs = self._roll_candidates(zi, a, goals[0], residuals)
            if not torch.isfinite(costs).all():
                raise FloatingPointError('Nonfinite actor-planner cost')
            winner = costs.argmin()
            if costs[winner] < best_cost:
                best_plan, best_cost = plans[winner].clone(), costs[winner].clone()
            elites = residuals[torch.topk(costs, self.K, largest=False).indices]
            mean = elites.mean(0).cpu().numpy()
            std = np.maximum(elites.std(0, unbiased=False).cpu().numpy(),
                             self.min_std.reshape(self.block_size, self.primitive_action_dim))
        return best_plan.cpu().numpy()
