"""CPU smoke/regression tests; no Colab, Drive, GPU, or simulator required.

Run: python tests/test_straightening_notebooks.py -v
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


def source(nb, index):
    return ''.join(nb['cells'][index]['source'])


def training_namespace(name='straightening/finetune.ipynb'):
    nb = notebook(name)
    ns = {}
    exec(source(nb, 4).split('\n', 1)[1], ns)
    for index in range(6, 14):
        exec(source(nb, index), ns)
    return ns


def save_source(path, encoder, predictor, stats, **metadata):
    stats = dict(stats, checkpoint_format_version=4,
                 architecture_id='lewm_robot_dual_encoder_action64_v4',
                 action_encoding_id='standardized_ordered_primitive_block_v2',
                 image_normalization_id='imagenet', frame_skip=5,
                 primitive_action_dim=7, predictor_action_dim=35, action_smoothing_dim=64)
    stats.update(metadata)
    with zipfile.ZipFile(path, 'w') as zf:
        for member, value in [('state_stats.pth', stats),
                              ('encoder_weights.pth', encoder.state_dict()),
                              ('predictor_weights.pth', predictor.state_dict())]:
            stream = io.BytesIO()
            torch.save(value, stream)
            zf.writestr(member, stream.getvalue())


class StraighteningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ns = training_namespace()

    def small_models(self):
        encoder = self.ns['RobotEncoder'](d_model=12, nheads=3, nlayers=1,
                                          projector_hidden_dim=24)
        predictor = self.ns['PredictorNetwork'](
            d_model=12, nheads=2, nlayers=1, d_action=35, window=3,
            dim_head=6, mlp_dim=24, projector_hidden_dim=24, action_smoothing_dim=64,
        )
        return encoder, predictor

    def test_syntax_and_preserved_pipeline_architecture_and_planner(self):
        train = notebook('straightening/finetune.ipynb')
        run = notebook('straightening/run_primary.ipynb')
        for nb in (train, run):
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    code = '\n'.join(line for line in source(nb, i).splitlines()
                                     if not line.startswith(('%', '!')))
                    compile(code, f'cell_{i}', 'exec')
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
        original = notebook('finetune.ipynb')
        for i in range(3, 13):
            self.assertEqual(source(train, i).replace('primary_straight', 'primary_split'),
                             source(original, i))
        for nb in (train, run):
            self.assertNotIn('primary_split', json.dumps(nb))
            self.assertIn('/content/drive/MyDrive/primary_straight', json.dumps(nb))
        original_run = notebook('run_primary_new.ipynb')
        for i in (1, 4, 5, 6, 7, 9, 11):
            self.assertEqual(source(run, i), source(original_run, i))
        for name in ('_prepare_batch', '_recalibrate_batch_norm', '_make_optimizer'):
            functions = []
            for nb in (train, original):
                functions.append(next(node for node in ast.parse(source(nb, 13)).body
                                      if isinstance(node, ast.FunctionDef) and node.name == name))
            self.assertEqual(ast.dump(functions[0]), ast.dump(functions[1]))
        code = source(train, 13)
        self.assertNotIn('expand_action_state', code)
        self.assertIn('target_embeddings = all_embeddings[:, 1:]', code)
        self.assertNotIn('target_embeddings.detach', code)
        self.assertIn("RUN_DIR = DRIVE_DIR / 'straightening' / RUN_NAME", source(train, 14))
        self.assertIn("RUN_DIR = PRIMARY_DIR", source(run, 3))

    def test_curvature_geometry_stationary_and_batch_boundaries(self):
        fn = self.ns['temporal_straightening_loss']
        # Straight motion, right-angle turn, reversal, and entirely stationary.
        trajectories = torch.tensor([
            [[0., 0.], [1., 0.], [3., 0.]],
            [[0., 0.], [1., 0.], [1., 1.]],
            [[0., 0.], [1., 0.], [0., 0.]],
            [[5., 5.], [5., 5.], [5., 5.]],
        ], requires_grad=True)
        for i, expected in enumerate((0., 1., 2., 0.)):
            loss, valid = fn(trajectories[i:i+1])
            self.assertAlmostEqual(loss.item(), expected, places=6)
            self.assertEqual(valid.item(), float(i != 3))
        loss, valid = fn(trajectories)
        self.assertAlmostEqual(loss.item(), 1., places=6)
        self.assertEqual(valid.item(), .75)
        loss.backward()
        self.assertTrue(torch.isfinite(trajectories.grad).all())
        self.assertGreater(trajectories.grad[1].abs().sum().item(), 0)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            stationary = torch.ones(2, 4, 12, dtype=dtype, requires_grad=True)
            value, valid = fn(stationary)
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.item(), 0.)
            self.assertEqual(valid.item(), 0.)
            value.backward()
            self.assertTrue(torch.isfinite(stationary.grad).all())
        with self.assertRaises(ValueError):
            fn(torch.zeros(2, 2, 12))
        for eps in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                fn(trajectories, eps=eps)

    def test_straightening_gradients_reach_both_encoders(self):
        encoder, _ = self.small_models()
        encoder.eval()
        images = torch.randn(2, 4, 3, 84, 84)
        state = torch.randn(2, 4, 9)
        z = encoder(images, images.flip(1), state)
        loss, valid = self.ns['temporal_straightening_loss'](z)
        self.assertEqual(valid.item(), 1.)
        loss.backward()
        for name in ('agent_encoder', 'wrist_encoder', 'fusion_mlp', 'projector'):
            gradients = [p.grad for p in getattr(encoder, name).parameters() if p.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)

    def test_zero_weight_preserves_loss_and_gradients(self):
        z = torch.randn(2, 4, 12, requires_grad=True)
        base = (z[:, 1:] - z[:, :-1]).square().mean() + .1 * z.square().mean()
        regularizer, _ = self.ns['temporal_straightening_loss'](z)
        expected, = torch.autograd.grad(base, z, retain_graph=True)
        actual, = torch.autograd.grad(base + 0. * regularizer, z)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_source_loading_preserves_width64_and_rejects_bad_contract(self):
        encoder, predictor = self.small_models()
        stats = {key: torch.ones(size) for key, size in [('state_mean', 9), ('state_std', 9),
                 ('action_mean', 7), ('action_std', 7), ('action_low', 7), ('action_high', 7)]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.zip'
            save_source(path, encoder, predictor, stats)
            enc, pred = self.small_models()
            with contextlib.redirect_stdout(io.StringIO()):
                self.ns['load_source_checkpoint'](path, enc, pred, stats)
            for original, loaded in ((encoder, enc), (predictor, pred)):
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
            wrong = dict(stats, action_mean=torch.zeros(7))
            with self.assertRaisesRegex(ValueError, 'Dataset action_mean'):
                self.ns['load_source_checkpoint'](path, enc, pred, wrong)
            save_source(path, encoder, predictor, stats, action_smoothing_dim=10)
            with self.assertRaisesRegex(ValueError, 'action_smoothing_dim'):
                self.ns['load_source_checkpoint'](path, enc, pred, stats)

    def test_scratch_training_resume_and_inference_load(self):
        original_ns = self.ns
        self.ns = training_namespace('straightening/train_primary.ipynb')
        try:
            nb = notebook('straightening/train_primary.ipynb')
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    code = '\n'.join(line for line in source(nb, i).splitlines()
                                     if not line.startswith(('%', '!')))
                    compile(code, f'scratch_cell_{i}', 'exec')
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
            for i in range(3, 13):
                self.assertEqual(source(nb, i), source(notebook('straightening/finetune.ipynb'), i))
            self.assertNotIn('source_checkpoint', source(nb, 13))
            self.assertIn("RESUME = False", source(nb, 14))
            self.test_training_resume_logging_and_inference_load(scratch=True)
        finally:
            self.ns = original_ns

    def test_training_resume_logging_and_inference_load(self, scratch=False):
        ns = self.ns
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            h5 = root / 'robot/mimicgen/demo.hdf5'
            h5.parent.mkdir(parents=True)
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
            mean, std = ns['compute_train_state_stats'](h5)
            am, astdev, low, high = ns['compute_train_action_stats'](h5)
            stats = dict(state_mean=mean, state_std=std, action_mean=am,
                         action_std=astdev, action_low=low, action_high=high)
            encoder, predictor = self.small_models()
            source_path = root / 'width64.zip'
            if not scratch:
                save_source(source_path, encoder, predictor, stats)
            kwargs = dict(device='cpu', data_dir=root, log_dir=root / 'logs', save_dir=root / 'models',
                          frame_skip=5, lr=5e-5, seed=42, lambd=.1, epochs=1, batch_size=2,
                          num_workers=0, persistent_workers=False, pin_memory=False, prefetch_factor=2,
                          in_order=True, loader_timeout=120, loader_start_method='spawn', compile_model=False,
                          window_size=3, embedding_dim=12, action_dim=7, pred_nheads=2, dropout=.1,
                          pred_nlayers=1, pred_dim_head=6, pred_mlp_dim=24, action_smoothing_dim=64,
                          projector_hidden_dim=24, patch_size=14, enc_nheads=3, enc_nlayers=1,
                          enc_npatches=36, knots=3, num_proj=4, optimizer_betas=(.9, .999),
                          optimizer_weight_decay=.001, gradient_clip_norm=1., bn_calibration_batches=1,
                          amp_dtype='off', config={'source_epoch': None}, source_checkpoint=source_path,
                          straightening_weight=.01, straightening_eps=1e-6)
            if scratch:
                kwargs.pop('source_checkpoint')
                kwargs['config'] = {'initialization': 'scratch'}
            with contextlib.redirect_stdout(io.StringIO()):
                first = ns['train'](**kwargs)
                if not scratch:
                    source_path.unlink()  # Resume must not depend on the original weights file.
                result = ns['train'](**kwargs, resume=True)
            self.assertEqual(len(first['val_loss']), 1)
            for split in ('train', 'val'):
                self.assertEqual(len(result[f'{split}_straightening_loss']), 2)
                for i in range(2):
                    expected = (result[f'{split}_prediction_loss'][i]
                                + .1 * result[f'{split}_sigreg_loss'][i]
                                + .01 * result[f'{split}_straightening_loss'][i])
                    self.assertAlmostEqual(result[f'{split}_loss'][i], expected, places=5)
                    self.assertTrue(np.isfinite(expected))
            checkpoint = root / 'models/epoch_1_model.zip'
            with zipfile.ZipFile(checkpoint) as zf:
                state = torch.load(io.BytesIO(zf.read('training_state.pth')), weights_only=False)
                weights = torch.load(io.BytesIO(zf.read('predictor_weights.pth')), weights_only=True)
            self.assertEqual(state['epoch'], 1)
            self.assertEqual(state['straightening_weight'], .01)
            self.assertEqual(state['straightening_space'], 'fused_projected_embedding')
            self.assertEqual(state['action_smoothing_dim'], 64)
            if scratch:
                self.assertEqual(state['initialization'], 'scratch')
            self.assertTrue(state['optimizer']['state'])
            self.assertTrue((root / 'models/best_model.zip').is_file())
            self.assertTrue(list((root / 'logs/Loss_Straightening_train').glob('events*')))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'straightening_weight'):
                    ns['train'](**dict(kwargs, straightening_weight=.1), resume=True)
            # Run the unchanged evaluator loader on an actually trained checkpoint.
            eval_ns = dict(ns, WINDOW=3, FRAME_SKIP=5, PREDICTOR_ACTION_DIM=35,
                           PRIMITIVE_ACTION_DIM=7, ACTION_SMOOTHING_DIM=64)
            eval_ns['RobotEncoder'] = lambda: self.small_models()[0]
            eval_ns['PredictorNetwork'] = lambda **kw: self.small_models()[1]
            tree = ast.parse(source(notebook('straightening/run_primary.ipynb'), 6))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in ('_load_torch', 'load_models'):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), 'loader', 'exec'), eval_ns)
            enc, pred, loaded_stats = eval_ns['load_models'](checkpoint, 'cpu')
            self.assertFalse(enc.training)
            self.assertFalse(pred.training)
            for key, value in weights.items():
                torch.testing.assert_close(pred.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(loaded_stats['action_mean'], am)


if __name__ == '__main__':
    unittest.main()
