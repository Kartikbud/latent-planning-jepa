import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from src.architectures.actor import GoalActor
from src.inference.actor_seeded_planner import ActorSeededPlanner


class Predictor(nn.Module):
    window = 3
    def forward(self, states, actions):
        return states + actions[..., :12] * .1


class ActorSeededTests(unittest.TestCase):
    def test_actor_called_only_for_initialization(self):
        torch.set_num_threads(2)
        actor = GoalActor(d_model=12, nlayers=1, ff_dim=24, dropout=0)
        calls = []
        hook = actor.register_forward_hook(lambda *args: calls.append(1))
        planner = ActorSeededPlanner('cpu', nn.Identity(), Predictor(), actor=actor,
            H=2, N=8, K=3, T=3, action_dim=35, primitive_action_dim=7,
            action_low=np.full(35,-1.), action_high=np.full(35,1.),
            noise_correlation=0., fresh_fraction=0., seed=42)
        for context in (1, 3):
            zi, history, goal = torch.randn(context,12), torch.zeros(context-1,35), torch.randn(12)
            calls.clear()
            planner.mode = 'direct'
            initial = planner.CEM(zi, history, goal)
            self.assertEqual(len(calls), 10)
            calls.clear()
            planner.mode = 'actor_seeded'
            result = planner.CEM(zi, history, goal)
            self.assertEqual(len(calls), 10)  # Not N*T*H*5 calls as in residual CEM.
            self.assertEqual(result.shape, (2,35))
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(np.isin(result.reshape(2,5,7)[..., -1], [-1,1]).all())
            costs = planner.score_actions(zi, history, np.stack([initial,result]), goal)
            self.assertLessEqual(costs[1].item(), costs[0].item()+1e-6)
        hook.remove()

    def test_standalone_notebook(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root/'notebooks/actor/run_primary_v1.ipynb').read_text())
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':
                code = '\n'.join(x for x in ''.join(cell['source']).splitlines() if not x.startswith(('%','!')))
                compile(code, str(i), 'exec')
        planner = (root/'src/inference/actor_seeded_planner.py').read_text().replace(
            'from src.inference.mpc_parallel import Planner\n','')
        self.assertIn(planner, ''.join(nb['cells'][5]['source']))
        self.assertNotIn('RESIDUAL_TEMPERATURE', str(nb))
        self.assertNotIn('gripper_schedules=schedules', ''.join(nb['cells'][11]['source']))
        original = json.loads((root/'notebooks/actor/run_primary.ipynb').read_text())
        for i in (1,2,3,4,6,7):
            self.assertEqual(nb['cells'][i]['source'], original['cells'][i]['source'])


if __name__ == '__main__':
    unittest.main()
