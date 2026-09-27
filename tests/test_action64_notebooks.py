"""CPU checks for action-width migration, fine-tuning/resume, and new evaluator.

Run: python tests/test_action64_notebooks.py -v
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    return json.loads((ROOT / 'notebooks' / name).read_text())


def training_namespace():
    nb = notebook('finetune.ipynb')
    ns = {}
    exec(''.join(nb['cells'][4]['source']).split('\n', 1)[1], ns)
    for i in range(6, 14):
        exec(''.join(nb['cells'][i]['source']), ns)
    return ns


def source_zip(path, encoder, predictor, stats):
    with zipfile.ZipFile(path, 'w') as zf:
        for name, data in [('encoder_weights.pth', encoder.state_dict()),
                           ('predictor_weights.pth', predictor.state_dict()),
                           ('state_stats.pth', dict(stats, checkpoint_format_version=3,
                            architecture_id='lewm_robot_dual_encoder_v3',
                            action_encoding_id='standardized_ordered_primitive_block_v2',
                            image_normalization_id='imagenet', frame_skip=5,
                            primitive_action_dim=7, predictor_action_dim=35))]:
            stream = io.BytesIO()
            torch.save(data, stream)
            zf.writestr(name, stream.getvalue())


class Action64Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ns = training_namespace()

    def test_notebook_syntax_and_unchanged_data_and_loss(self):
        for name in ('finetune.ipynb', 'run_primary_new.ipynb'):
            for i, cell in enumerate(notebook(name)['cells']):
                if cell['cell_type'] == 'code':
                    source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                                       if not line.startswith(('%', '!')))
                    compile(source, f'{name}:{i}', 'exec')
        old, new = notebook('train_primary.ipynb'), notebook('finetune.ipynb')
        for i in range(3, 13):
            self.assertEqual(old['cells'][i]['source'], new['cells'][i]['source'])
        for name in ('_prepare_batch', '_recalibrate_batch_norm', '_make_optimizer'):
            nodes = []
            for nb in (old, new):
                tree = ast.parse(''.join(nb['cells'][13]['source']))
                nodes.append(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]))
        old_source, new_source = (''.join(nb['cells'][13]['source']) for nb in (old, new))
        self.assertEqual(old_source[old_source.index('    for epoch in tqdm'):],
                         new_source[new_source.index('    for epoch in tqdm'):])

    def test_real_predictor_equivalence_and_new_feature_gradients(self):
        ns = self.ns
        torch.manual_seed(7)
        old = ns['PredictorNetwork'](d_action=35, action_smoothing_dim=10).eval()
        # Newly initialized AdaLN gates are zero. Use nonzero gates to test action dependence.
        with torch.no_grad():
            for block in old.transformer_layers:
                torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=.02)
        new = ns['PredictorNetwork'](d_action=35, action_smoothing_dim=64).eval()
        new.load_state_dict(ns['expand_action_state'](old.state_dict(), new), strict=True)
        z, a = torch.randn(2, 3, 192), torch.randn(2, 3, 35)
        with torch.no_grad():
            torch.testing.assert_close(old(z, a), new(z, a), atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(old.action_encoder(a), new.action_encoder(a), atol=1e-6, rtol=1e-5)
        self.assertGreater(new.action_encoder.smoother.weight[10:].abs().sum().item(), 0)
        optimizer = torch.optim.SGD(new.action_encoder.parameters(), lr=.01)
        for step in range(2):
            optimizer.zero_grad()
            new.action_encoder(a).square().mean().backward()
            self.assertGreater(new.action_encoder.mlp[0].weight.grad[:, 10:].abs().sum().item(), 0)
            if step:
                self.assertGreater(new.action_encoder.smoother.weight.grad[10:].abs().sum().item(), 0)
            optimizer.step()

    def test_rollout_horizon_control_and_planning_shapes(self):
        nb = notebook('run_primary_new.ipynb')
        controls = ''.join(nb['cells'][9]['source'])
        tree = ast.parse(controls)
        assignments = {node.targets[0].id: node for node in tree.body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        validation = next(node for node in tree.body if isinstance(node, ast.If))
        ns = dict(torch=torch, np=np, nn=torch.nn, FRAME_SKIP=5)
        exec(''.join(nb['cells'][5]['source']), ns)
        self.assertNotIn('HORIZON', ''.join(nb['cells'][3]['source']))
        self.assertEqual(ast.literal_eval(assignments['HORIZON'].value), 5)
        planner_args = {kw.arg: kw.value for kw in assignments['planner'].value.keywords}
        self.assertEqual(ast.dump(planner_args['H']), ast.dump(ast.Name(id='HORIZON', ctx=ast.Load())))
        validate = compile(ast.Module(body=[validation, assignments['PLAN_STEPS']], type_ignores=[]),
                           'horizon_controls', 'exec')
        predictor = self.ns['PredictorNetwork'](
            d_model=12, nheads=2, nlayers=1, d_action=35, window=3,
            dim_head=6, mlp_dim=24, projector_hidden_dim=24, action_smoothing_dim=64,
        ).eval()
        for horizon in (1, 2, 3, 5):
            with self.subTest(horizon=horizon):
                ns['HORIZON'] = horizon
                exec(validate, ns)
                self.assertEqual(ns['PLAN_STEPS'], horizon * 5)
                planner = ns['Planner']('cpu', torch.nn.Identity(), predictor,
                                        H=horizon, N=20, K=4, T=2, seed=42,
                                        noise_correlation=0., fresh_fraction=0.,
                                        separate_gripper_searches=False)
                zi, history, goal = torch.randn(3, 12), torch.randn(2, 35), torch.randn(12)
                grasp = planner.single_switch_gripper_schedules()
                self.assertEqual(grasp.shape, (2*horizon, horizon, 5))
                for schedules in (grasp, np.ones((1, horizon, 5), dtype=np.float32)):
                    plan = planner.CEM(zi, history, goal, gripper_schedules=schedules)
                    self.assertEqual(plan.shape, (horizon, 35))
                    self.assertTrue(np.isfinite(plan).all())
                    decoded = planner.denormalize_action_blocks(plan)
                    self.assertEqual(decoded[:1].shape, (1, 5, 7))
        for invalid in (0, -1, 2.5, True, '3'):
            ns['HORIZON'] = invalid
            with self.assertRaisesRegex(ValueError, 'HORIZON must be'):
                exec(validate, ns)
        self.assertIn('for raw_block in raw_blocks[:1]:', ''.join(nb['cells'][6]['source']))
        self.assertIn('planner.H * planner.block_size', ''.join(nb['cells'][11]['source']))

    def test_actual_training_save_resume_on_tiny_dataset(self):
        ns = self.ns
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'robot/mimicgen'
            folder.mkdir(parents=True)
            h5 = folder / 'demo.hdf5'
            rng = np.random.default_rng(4)
            with h5py.File(h5, 'w') as f:
                for i in range(2):
                    demo = f.create_group(f'data/demo_{i}')
                    demo.attrs['num_samples'] = 19
                    demo.create_dataset('actions', data=rng.normal(0, .1, (19, 7)).astype('float32'))
                    obs = demo.create_group('obs')
                    for camera in ('agentview_image', 'robot0_eye_in_hand_image'):
                        obs.create_dataset(camera, data=rng.integers(0, 256, (19, 84, 84, 3), dtype='uint8'))
                    for key, width in [('robot0_eef_pos', 3), ('robot0_eef_quat', 4), ('robot0_gripper_qpos', 2)]:
                        obs.create_dataset(key, data=rng.normal(size=(19, width)).astype('float32'))
                f.create_dataset('mask/train', data=np.array([b'demo_0']))
                f.create_dataset('mask/valid', data=np.array([b'demo_1']))
            state_mean, state_std = ns['compute_train_state_stats'](h5)
            action_mean, action_std, action_low, action_high = ns['compute_train_action_stats'](h5)
            stats = dict(state_mean=state_mean, state_std=state_std, action_mean=action_mean,
                         action_std=action_std, action_low=action_low, action_high=action_high)
            # Small networks exercise the full unchanged optimizer/loss/checkpoint loop quickly.
            enc = ns['RobotEncoder'](d_model=12, nheads=3, nlayers=1, projector_hidden_dim=24)
            pred = ns['PredictorNetwork'](d_model=12, nheads=2, nlayers=1, d_action=35,
                                         dim_head=6, mlp_dim=24, projector_hidden_dim=24)
            source = root / 'epoch_52_model.zip'
            source_zip(source, enc, pred, stats)
            kwargs = dict(device='cpu', data_dir=root, log_dir=root / 'logs', save_dir=root / 'models',
                          frame_skip=5, lr=5e-5, seed=42, lambd=.1, epochs=1, batch_size=2,
                          num_workers=0, persistent_workers=False, pin_memory=False, prefetch_factor=2,
                          in_order=True, loader_timeout=120, loader_start_method='spawn', compile_model=False,
                          window_size=3, embedding_dim=12, action_dim=7, pred_nheads=2, dropout=.1,
                          pred_nlayers=1, pred_dim_head=6, pred_mlp_dim=24, action_smoothing_dim=64,
                          projector_hidden_dim=24, patch_size=14, enc_nheads=3, enc_nlayers=1,
                          enc_npatches=36, knots=3, num_proj=4, optimizer_betas=(.9, .999),
                          optimizer_weight_decay=.001, gradient_clip_norm=1., bn_calibration_batches=1,
                          amp_dtype='off', config={'source_epoch': 52}, source_checkpoint=source)
            with contextlib.redirect_stdout(io.StringIO()):
                first = ns['train'](**kwargs)
                source.unlink()  # Resume must use the widened checkpoint, not remigrate.
                resumed = ns['train'](**kwargs, resume=True)
            self.assertEqual(len(first['val_loss']), 1)
            self.assertEqual(len(resumed['val_loss']), 2)
            checkpoint = root / 'models/epoch_1_model.zip'
            with zipfile.ZipFile(checkpoint) as zf:
                saved = torch.load(io.BytesIO(zf.read('training_state.pth')), weights_only=False)
                weights = torch.load(io.BytesIO(zf.read('predictor_weights.pth')), weights_only=True)
            self.assertEqual(saved['epoch'], 1)
            self.assertEqual(saved['source_epoch'], 52)
            self.assertEqual(saved['action_smoothing_dim'], 64)
            self.assertEqual(saved['checkpoint_format_version'], 4)
            self.assertEqual(weights['action_encoder.smoother.weight'].shape, (64, 35, 1))
            self.assertTrue(saved['optimizer']['state'])
            self.assertTrue((root / 'models/best_model.zip').is_file())

    def test_new_inference_loads_widened_checkpoint(self):
        ns = self.ns
        eval_ns = dict(torch=torch, zipfile=zipfile, tempfile=tempfile, Path=Path,
                       RobotEncoder=ns['RobotEncoder'], PredictorNetwork=ns['PredictorNetwork'],
                       CHECKPOINT_FORMAT_VERSION=4, ARCHITECTURE_ID='lewm_robot_dual_encoder_action64_v4',
                       ACTION_ENCODING_ID='standardized_ordered_primitive_block_v2', IMAGE_NORMALIZATION_ID='imagenet',
                       FRAME_SKIP=5, PRIMITIVE_ACTION_DIM=7, PREDICTOR_ACTION_DIM=35, WINDOW=3,
                       ACTION_SMOOTHING_DIM=64)
        tree = ast.parse(''.join(notebook('run_primary_new.ipynb')['cells'][6]['source']))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ('_load_torch', 'load_models'):
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'loader', 'exec'), eval_ns)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'new.zip'
            enc = ns['RobotEncoder']()
            pred = ns['PredictorNetwork'](d_action=35, action_smoothing_dim=64)
            stats = {key: torch.ones(size) for key, size in [('state_mean', 9), ('state_std', 9),
                     ('action_mean', 7), ('action_std', 7), ('action_low', 7), ('action_high', 7)]}
            stats.update(checkpoint_format_version=4, architecture_id=eval_ns['ARCHITECTURE_ID'],
                         action_encoding_id=eval_ns['ACTION_ENCODING_ID'], image_normalization_id='imagenet',
                         frame_skip=5, primitive_action_dim=7, predictor_action_dim=35, action_smoothing_dim=64)
            with zipfile.ZipFile(path, 'w') as zf:
                for member, data in [('encoder_weights.pth', enc.state_dict()),
                                     ('predictor_weights.pth', pred.state_dict()), ('state_stats.pth', stats)]:
                    stream = io.BytesIO(); torch.save(data, stream); zf.writestr(member, stream.getvalue())
            loaded_enc, loaded_pred, loaded_stats = eval_ns['load_models'](path, 'cpu')
            self.assertFalse(loaded_enc.training)
            self.assertEqual(loaded_pred.action_encoder.smoother.out_channels, 64)
            for key, value in pred.state_dict().items():
                torch.testing.assert_close(value, loaded_pred.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
