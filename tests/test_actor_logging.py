"""W&B behavior tests with a fake SDK: no login or network writes."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.train.actor_logging import start_actor_wandb, log_actor_epoch


class ActorLoggingTests(unittest.TestCase):
    def setUp(self):
        self.run = SimpleNamespace(id='run123', project='primary_actor', entity='tester',
                                   name='primary_actor', url='https://example.invalid/run123',
                                   define_metric=Mock(), log=Mock(), finish=Mock(), summary={})
        self.sdk = SimpleNamespace(run=None, login=Mock(), init=Mock(return_value=self.run))
        self.modules = {'wandb': self.sdk, 'google.colab': SimpleNamespace(
            userdata=SimpleNamespace(get=Mock(return_value='test-secret')))}

    def test_new_then_auto_resume_identity(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('sys.modules', self.modules):
            options = dict(enabled=True, project='primary_actor', mode='auto')
            start_actor_wandb(options, training_resume=False, saved_run=None, output_dir=folder, config={})
            self.assertEqual(self.sdk.init.call_args.kwargs['resume'], 'never')
            saved = json.loads((Path(folder)/'wandb_run.json').read_text())
            self.assertEqual(saved['id'], 'run123')
            self.assertNotIn('test-secret', json.dumps(saved))
            start_actor_wandb(options, training_resume=True, saved_run=saved, output_dir=folder, config={})
            self.assertEqual(self.sdk.init.call_args.kwargs['resume'], 'must')
            self.assertEqual(self.sdk.init.call_args.kwargs['id'], 'run123')
            start_actor_wandb(dict(options, mode='new'), training_resume=True,
                              saved_run=saved, output_dir=folder, config={})
            self.assertEqual(self.sdk.init.call_args.kwargs['resume'], 'never')
            self.assertIsNone(self.sdk.init.call_args.kwargs['id'])

    def test_explicit_resume_and_scratch_protection(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('sys.modules', self.modules):
            options = dict(mode='resume', run_id='chosen')
            start_actor_wandb(options, training_resume=True, saved_run=None, output_dir=folder, config={})
            self.assertEqual(self.sdk.init.call_args.kwargs['id'], 'chosen')
            with self.assertRaises(ValueError):
                start_actor_wandb(options, training_resume=False, saved_run=None, output_dir=folder, config={})
            self.assertIsNone(start_actor_wandb({'enabled':False}, training_resume=False,
                                              saved_run=None, output_dir=folder, config={}))

    def test_all_epoch_metrics(self):
        results = {f'{split}_{suffix}':[1.] for split in ('train','val')
                   for suffix in ('loss','prediction_loss','sigreg_loss','actor_nll','actor_mse')}
        results.update({key:[2.] for key in ('train_grad_norm','encoder_bn_gamma_rms',
                       'encoder_bn_running_var_min','encoder_bn_running_var_max')})
        log_actor_epoch(self.run, 7, results, 4, .2, 5e-5)
        metrics = self.run.log.call_args.args[0]
        self.assertEqual(metrics['epoch'], 7)
        self.assertEqual(len(metrics), 16)
        self.assertEqual(self.run.summary['best_epoch'], 4)


if __name__ == '__main__':
    unittest.main()
