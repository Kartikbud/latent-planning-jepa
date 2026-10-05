"""CPU tests of standalone actor notebooks, causal targets and residual planning."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import torch

from src.architectures.actor import GoalActor, actor_objective
from src.datasets.robot.processing_actor import sample_indices
from src.inference.actor_planner import ActorPlanner
from src.datasets.robot.actor_dataset import RobotActorDataset

ROOT = Path(__file__).resolve().parents[1]


def cells(name):
    return json.loads((ROOT / f'notebooks/actor/{name}.ipynb').read_text())['cells']


def code(cell):
    return ''.join(cell['source'])


def training_namespace():
    content = cells('train_primary')
    ns = {}
    exec(code(content[4]).split('\n', 1)[1], ns)
    for i in range(6, 14):
        exec(code(content[i]), ns)
    return ns


def make_data(root):
    path = root / 'robot/actor/demo_actor.hdf5'
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(42)
    with h5py.File(path, 'w') as f:
        f.attrs['actor_dataset_version'] = 1
        f.attrs['frame_skip'] = 5
        for i in range(2):
            g = f.create_group(f'data/demo_{i}')
            g.create_dataset('actions', data=rng.normal(0, .1, (19, 7)).astype('float32'))
            for key in ('agentview_image', 'robot0_eye_in_hand_image'):
                # Frame-number images make exact tuple alignment testable.
                g.create_dataset(f'obs/{key}', data=np.broadcast_to(
                    np.arange(19, dtype='uint8')[:, None, None, None], (19, 84, 84, 3)))
            for key, width in [('robot0_eef_pos', 3), ('robot0_eef_quat', 4), ('robot0_gripper_qpos', 2)]:
                g.create_dataset(f'obs/{key}', data=rng.normal(size=(19, width)).astype('float32'))
            for key, value in sample_indices(19, 12).items():
                g.create_dataset(f'actor_samples/{key}', data=value)
        f.create_dataset('mask/train', data=np.array([b'demo_0']))
        f.create_dataset('mask/valid', data=np.array([b'demo_1']))
    return path


class ActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_notebook_syntax_and_standalone_components(self):
        for name in ('train_primary', 'run_primary'):
            for i, cell in enumerate(cells(name)):
                if cell['cell_type'] == 'code':
                    s = '\n'.join(line for line in code(cell).splitlines() if not line.startswith(('%', '!')))
                    compile(s, f'{name}:{i}', 'exec')
                    self.assertEqual(cell['outputs'], [])
            self.assertIn('primary_actor', str(cells(name)))
            self.assertNotIn('primary_straight', str(cells(name)))
        self.assertNotIn('temporal_straightening_loss', str(cells('train_primary')))
        self.assertIn((ROOT / 'src/architectures/actor.py').read_text(), code(cells('train_primary')[12]))
        self.assertIn((ROOT / 'src/architectures/actor.py').read_text(), code(cells('run_primary')[4]))
        self.assertIn((ROOT / 'src/inference/actor_planner.py').read_text().replace(
            'from src.inference.mpc_parallel import Planner\n', ''), code(cells('run_primary')[5]))

    def test_spawned_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); make_data(root)
            ds = RobotActorDataset(root, 5, 3)
            loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=1,
                                                 multiprocessing_context='spawn', timeout=30)
            batches = list(loader)
            self.assertEqual(len(batches), 2)
            self.assertEqual(batches[0][0].shape, (2, 5, 3, 84, 84))
            ds.close()

    def test_fixed_subgoals_and_success_stop(self):
        class Env:
            steps = 0
            def step(self, action):
                self.steps += 1
                return self.steps, 0, False, {}
            def _check_success(self):
                return self.steps >= 17
        class Writer:
            def append_data(self, frame):
                pass
        class PlannerStub:
            goals = []
            def CEM(self, zi, history, goal):
                self.goals.append(goal.clone())
                assert history.shape[0] == zi.shape[0] - 1
                return np.zeros((2, 35), dtype=np.float32)
            def denormalize_action_blocks(self, plan):
                return plan.reshape(2, 5, 7)
            def normalize_primitive_actions(self, actions):
                return actions
        ns = dict(torch=torch, np=np, deque=deque, FRAME_SKIP=5, WINDOW=3, PREDICTOR_ACTION_DIM=35,
                  reset_to_eval_state=lambda env, entry: 0, render_frame=lambda env: None,
                  encode_observation=lambda *args: torch.zeros(12), set_geom_groups=lambda env: None,
                  postprocess_action=lambda a: a)
        tree = ast.parse(code(cells('run_primary')[6]))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_rollout')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'rollout', 'exec'), ns)
        entry = dict(name='demo_0', start=torch.zeros(12), stages=[
            dict(name='grasp', goal=torch.ones(12), frame=12),
            dict(name='lift', goal=torch.ones(12)*2, frame=18)])
        settings = {s: dict(budget=b, threshold=0., use_threshold=False)
                    for s,b in [('grasp',15),('lift',10)]}
        planner = PlannerStub()
        with contextlib.redirect_stdout(io.StringIO()):
            success, steps = ns['run_rollout'](Env(), planner, None, entry, 'cpu', {}, Writer(), settings)
        self.assertTrue(success)
        self.assertEqual(steps, 17)
        self.assertEqual([g[0].item() for g in planner.goals], [1,1,1,2])

    def test_actor_causality_mask_and_gradients(self):
        actor = GoalActor(d_model=12, nlayers=1, ff_dim=24, dropout=0)
        z = torch.randn(2, 12, requires_grad=True)
        goal = torch.randn(2, 12, requires_grad=True)
        prev = torch.randn(2, 5, 7)
        targets = torch.randn(2, 5, 7)
        mask = torch.tensor([[1, 1, 0, 0, 0]]).expand(2, -1)
        loss, _ = actor_objective(actor, z, goal, prev, targets, mask)
        altered = targets.clone(); altered[:, 2:] += 99
        other, _ = actor_objective(actor, z, goal, prev, altered, mask)
        torch.testing.assert_close(loss, other)
        loss.backward()
        self.assertGreater(z.grad.abs().sum(), 0)
        self.assertIsNone(goal.grad)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))
        sf, _ = actor_objective(actor, z, goal, prev, targets, mask, 1.0)
        self.assertTrue(torch.isfinite(sf))
        actor.eval()
        means, _ = actor(z, goal, prev, targets[:, :-1])
        changed, _ = actor(z, goal + 2, prev, targets[:, :-1])
        self.assertFalse(torch.allclose(means, changed))

    def test_train_resume_and_eval_loader(self):
        ns = training_namespace()
        wb = SimpleNamespace(id='test-id', project='primary_actor', entity='test', name='test',
                             log=Mock(), finish=Mock(), summary={})
        ns['start_actor_wandb'] = Mock(return_value=wb)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); make_data(root)
            ds = ns['RobotActorDataset'](root, 5, 3)
            above, wrist, state, actions, mask = ds[0]
            np.testing.assert_array_equal(above[:, 0, 0, 0], [0, 5, 10, 15, 12])
            np.testing.assert_array_equal(mask, [1, 1, 0, 0, 0])
            self.assertEqual(actions.shape, (3, 5, 7))
            ds.close()
            kw = dict(device='cpu', data_dir=root, log_dir=root/'logs', save_dir=root/'models',
                      frame_skip=5, lr=5e-5, seed=42, lambd=.1, epochs=1, batch_size=2,
                      num_workers=0, persistent_workers=False, pin_memory=False, prefetch_factor=2,
                      in_order=True, loader_timeout=120, loader_start_method='spawn', compile_model=False,
                      window_size=3, embedding_dim=12, action_dim=7, pred_nheads=2, dropout=0.1,
                      pred_nlayers=1, pred_dim_head=6, pred_mlp_dim=24, action_smoothing_dim=64,
                      projector_hidden_dim=24, patch_size=14, enc_nheads=3, enc_nlayers=1,
                      enc_npatches=36, knots=3, num_proj=4, optimizer_betas=(.9,.999),
                      optimizer_weight_decay=.001, gradient_clip_norm=1., bn_calibration_batches=1,
                      amp_dtype='off', actor_weight=.1, student_probability=.5)
            with contextlib.redirect_stdout(io.StringIO()):
                ns['train'](**kw)
                result = ns['train'](**kw, resume=True)
            self.assertEqual([call.args[0]['epoch'] for call in wb.log.call_args_list], [0, 1])
            self.assertEqual(wb.finish.call_count, 2)
            self.assertEqual(ns['start_actor_wandb'].call_args.kwargs['saved_run']['id'], 'test-id')
            for split in ('train', 'val'):
                self.assertEqual(len(result[f'{split}_actor_nll']), 2)
                expected = (result[f'{split}_prediction_loss'][-1] + .1*result[f'{split}_sigreg_loss'][-1]
                            + .1*result[f'{split}_actor_nll'][-1])
                self.assertAlmostEqual(result[f'{split}_loss'][-1], expected, places=5)
            checkpoint = root/'models/epoch_1_model.zip'
            with zipfile.ZipFile(checkpoint) as zf:
                actor_weights = torch.load(io.BytesIO(zf.read('actor_weights.pth')), weights_only=True)
                state = torch.load(io.BytesIO(zf.read('training_state.pth')), weights_only=False)
                self.assertEqual(state['epoch'], 1)
                self.assertEqual(state['checkpoint_format_version'], 5)
                self.assertEqual(state['wandb_run']['id'], 'test-id')
            ev = dict(ns, FRAME_SKIP=5, WINDOW=3, PREDICTOR_ACTION_DIM=35, PRIMITIVE_ACTION_DIM=7,
                      ACTION_SMOOTHING_DIM=64)
            ev['RobotEncoder'] = lambda: ns['RobotEncoder'](d_model=12, nheads=3, nlayers=1, projector_hidden_dim=24)
            ev['PredictorNetwork'] = lambda **k: ns['PredictorNetwork'](d_model=12, nheads=2, nlayers=1,
                d_action=35, window=3, dim_head=6, mlp_dim=24, projector_hidden_dim=24, action_smoothing_dim=64)
            for node in ast.parse(code(cells('run_primary')[6])).body:
                if isinstance(node, ast.FunctionDef) and node.name in ('_load_torch', 'load_models'):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), 'eval_loader', 'exec'), ev)
            enc, pred, actor, stats = ev['load_models'](checkpoint, 'cpu')
            for key, value in actor_weights.items():
                torch.testing.assert_close(value, actor.state_dict()[key])
            # Real trained models exercise the complete planner in all three modes.
            for mode in ('direct', 'arcem', 'cem'):
                planner = ActorPlanner('cpu', enc, pred, actor=actor, mode=mode,
                    H=2, N=4, K=2, T=2, action_dim=35, primitive_action_dim=7,
                    action_low=np.full(35, -1.), action_high=np.full(35, 1.))
                plan = planner.CEM(torch.randn(1,12), torch.empty(0,35), torch.randn(12))
                self.assertEqual(plan.shape, (2,35))
                self.assertTrue(np.isfinite(plan).all())
                self.assertTrue(np.isin(plan.reshape(2,5,7)[..., -1], [-1,1]).all())
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    ns['train'](**dict(kw, actor_weight=.2), resume=True)


if __name__ == '__main__':
    unittest.main()
