import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'src/datasets/robot/processing_actor.py'
SPEC = importlib.util.spec_from_file_location('processing_actor', PATH)
actor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(actor)


class ActorProcessingTests(unittest.TestCase):
    def test_indices_and_boundary_mask(self):
        result = actor.sample_indices(30, 12)
        np.testing.assert_array_equal(result['observation_indices'][0], [0, 5, 10, 15, 12])
        np.testing.assert_array_equal(result['actor_action_mask'][0], [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(result['observation_indices'][2], [2, 7, 12, 17, 29])
        self.assertEqual(result['goal_phase'][2], 1)
        self.assertTrue(result['actor_action_mask'][-1].all())
        self.assertEqual(result['observation_indices'][-1, 3], 29)
        np.testing.assert_array_equal(result['action_start_indices'][0], [0, 5, 10])

    def test_filter_copy_and_split_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            src, out = Path(folder) / 'source.hdf5', Path(folder) / 'actor.hdf5'
            with h5py.File(src, 'w') as f:
                for i in range(4):
                    demo = f.create_group(f'data/demo_{i}')
                    demo.create_dataset('actions', data=np.zeros((30, 7)))
                    demo.create_dataset('states', data=np.arange(30)[:, None])
                    for key in actor.OBS_KEYS:
                        demo.create_dataset(f'obs/{key}', data=np.arange(30), chunks=(1,), compression='gzip')
                    flags = np.zeros(30, dtype=np.uint8)
                    if i != 2:
                        flags[12:] = 1
                    if i == 3:
                        flags[16:18] = 0
                    demo.create_dataset(actor.SIGNAL, data=flags)
                f.create_dataset('mask/train', data=np.array([b'demo_0', b'demo_2', b'demo_3']))
                f.create_dataset('mask/valid', data=np.array([b'demo_1']))
            counts = actor.process(src, out)
            self.assertEqual(counts['kept'], 2)
            self.assertEqual(counts['no_grasp'], 1)
            self.assertEqual(counts['multiple_grasps'], 1)
            with h5py.File(out) as f, h5py.File(src) as original:
                self.assertEqual(list(f['data']), ['demo_0', 'demo_1'])
                self.assertEqual(f['mask/train'].asstr()[:].tolist(), ['demo_0'])
                self.assertEqual(f['mask/valid'].asstr()[:].tolist(), ['demo_1'])
                self.assertEqual(f['data'].attrs['total'], 60)
                image = f['data/demo_0/obs/agentview_image']
                self.assertEqual(image.chunks, (1,))
                self.assertEqual(image.compression, 'gzip')
                np.testing.assert_array_equal(image[:], original[image.name][:])
                self.assertNotIn('actor_samples', original['data/demo_0'])
                self.assertEqual(f['data/demo_0/actor_samples/observation_indices'].shape, (15, 5))
            with self.assertRaises(FileExistsError):
                actor.process(src, out)
            with self.assertRaises(ValueError):
                actor.process(src, src)


if __name__ == '__main__':
    unittest.main()
