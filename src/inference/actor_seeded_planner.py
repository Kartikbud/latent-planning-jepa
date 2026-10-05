"""Actor initializes one complete plan; ordinary CEM then searches action space."""
import numpy as np
import torch

from src.inference.mpc_parallel import Planner


class ActorSeededPlanner(Planner):
    def __init__(self, *args, actor, mode='actor_seeded', **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ('actor_seeded', 'direct', 'cem'):
            raise ValueError('mode must be actor_seeded, direct, or cem')
        if self.action_low is None:
            raise ValueError('Actor planning requires action bounds')
        self.actor, self.mode = actor, mode

    def executable(self, actions):
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
    def actor_plan(self, zi, history, goal):
        states, actions = zi[None], history[None]
        low = torch.as_tensor(self.action_low, device=zi.device).reshape(self.block_size, -1)
        high = torch.as_tensor(self.action_high, device=zi.device).reshape(self.block_size, -1)
        blocks = []
        for _ in range(self.H):
            if actions.shape[1]:
                previous = actions[:, -1].reshape(1, self.block_size, self.primitive_action_dim)
            else:
                zero = (-torch.as_tensor(self.raw_action_mean, device=zi.device) /
                        torch.as_tensor(self.raw_action_std, device=zi.device))
                previous = zero.expand(1, self.block_size, -1)
            prefix = zi.new_empty(1, 0, self.primitive_action_dim)
            for j in range(self.block_size):
                means, _ = self.actor(states[:, -1], goal.reshape(1, -1), previous, prefix)
                proposal = self.executable(means[:, -1].clamp(low[j], high[j]))
                prefix = torch.cat((prefix, proposal[:, None]), dim=1)
            block = prefix.flatten(1)
            blocks.append(block[0])
            actions = torch.cat((actions, block[:, None]), dim=1)
            predicted = self.pred(states, actions)[:, -1]
            states = torch.cat((states, predicted[:, None]), dim=1)[:, -self.pred.window:]
            actions = actions[:, -(self.pred.window - 1):]
        return torch.stack(blocks).cpu().numpy()

    @torch.no_grad()
    def CEM(self, zi, a, zg, initial_mean=None, gripper_schedules=None):
        if initial_mean is not None or gripper_schedules is not None:
            raise ValueError('This planner uses actor initialization, not warm starts or gripper schedules')
        self.actor.eval()
        self.pred.eval()
        self.enc.eval()
        zi, a, goals = self._prepare_inputs(zi, a, zg)
        if goals.shape[0] != 1:
            raise ValueError('Expected exactly one active subgoal')
        if self.mode == 'cem':
            return super().CEM(zi, a, goals)
        initial_plan = self.actor_plan(zi, a, goals[0])
        if not np.isfinite(initial_plan).all():
            raise FloatingPointError('Nonfinite actor initialization')
        if self.mode == 'direct':
            return initial_plan
        # No actor calls below: Gaussian means/stds are fitted to elite ACTIONS.
        return super().CEM(zi, a, goals, initial_mean=initial_plan)
