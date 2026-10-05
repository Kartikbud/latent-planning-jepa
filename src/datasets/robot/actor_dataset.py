"""Spawn-safe loader for processing_actor.py's indexed HDF5 schema."""
import h5py
import numpy as np
import torch
from pathlib import Path

from src.datasets.robot.primary_dataset import RobotPrimaryDataset


class RobotActorDataset(RobotPrimaryDataset):
    def __init__(self, data_dir, *args, **kwargs):
        kwargs.setdefault('h5_path', Path(data_dir) / 'robot/actor/demo_actor.hdf5')
        super().__init__(data_dir, *args, **kwargs)
        if self.window != 3:
            raise ValueError('Actor dataset has three transitions per sample')
        # Compact tables are loaded once, not repeatedly decompressed per sample.
        observations, actions, masks = [], [], []
        with h5py.File(self.h5_path, 'r') as f:
            if f.attrs.get('actor_dataset_version') != 1 or f.attrs.get('frame_skip') != self.frame_skip:
                raise ValueError('Incompatible actor dataset version or frame skip')
            for name in self.episodes:
                group = f[f'data/{name}/actor_samples']
                observations.append(group['observation_indices'][:])
                actions.append(group['action_start_indices'][:])
                masks.append(group['actor_action_mask'][:])
        if not observations:
            raise ValueError('Empty actor dataset split')
        self.observation_indices = np.concatenate(observations)
        self.action_indices = np.concatenate(actions)
        self.actor_masks = np.concatenate(masks)
        if self.observation_indices.shape != (len(self), 5):
            raise ValueError('Actor sample counts differ from chronological windows')
        if self.action_indices.shape != (len(self), 3) or self.actor_masks.shape != (len(self), self.frame_skip):
            raise ValueError('Malformed actor action indices/masks')
        if not np.all(np.diff(self.observation_indices[:, :4], axis=1) == self.frame_skip):
            raise ValueError('Nonconsecutive world-model indices')
        if not np.array_equal(self.action_indices, self.observation_indices[:, :3]):
            raise ValueError('Misaligned primitive action blocks')
        expected = (self.observation_indices[:, 2, None] + np.arange(self.frame_skip)
                    < self.observation_indices[:, 4, None])
        if not np.array_equal(expected, self.actor_masks) or not self.actor_masks.any(axis=1).all():
            raise ValueError('Invalid actor boundary masks')

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        name, offset = self._resolve_index(idx)
        indices = self.observation_indices[idx]
        start, goal = int(indices[0]), int(indices[-1])
        if start != offset:
            raise ValueError('Unexpected sample ordering')
        episode = self._get_h5()[f'data/{name}']
        obs = episode['obs']
        end = int(indices[3]) + 1
        def frames(key):
            # Retain contiguous-read optimization; goal can precede last frame
            # or duplicate a history frame, so do not use HDF5 fancy indexing.
            history = obs[key][start:end][::self.frame_skip]
            if goal in indices[:4]:
                target = history[np.flatnonzero(indices[:4] == goal)[0]]
            else:
                target = obs[key][goal]
            return torch.from_numpy(np.concatenate((history, target[None]), axis=0))
        above = frames('agentview_image').permute(0, 3, 1, 2)
        wrist = frames('robot0_eye_in_hand_image').permute(0, 3, 1, 2)
        # Reuse the base loader's quaternion handling and normalization.
        from_state = self._load_state(obs, slice(start, end, self.frame_skip))
        goal_state = self._load_state(obs, slice(goal, goal + 1, 1))
        state = torch.cat((from_state, goal_state), dim=0)
        actions = torch.from_numpy(episode['actions'][start:start + 3*self.frame_skip]).float()
        actions = ((actions - self.action_mean) / self.action_std).reshape(3, self.frame_skip, -1)
        return above, wrist, state, actions, torch.from_numpy(self.actor_masks[idx].copy())
