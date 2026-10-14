"""Prepare goal-conditioned actor samples without duplicating camera images.

    python src/datasets/robot/processing_actor.py --scan-only
    python src/datasets/robot/processing_actor.py
    hf auth login
    python src/datasets/robot/processing_actor.py --upload-only --repo-id USERNAME/DATASET

Output: data/robot/actor/demo_actor.hdf5. Source files are never modified.
The output is a directly downloadable HDF5, not a Hugging Face Arrow dataset.
Use hf_hub_download(repo_id='USERNAME/DATASET',
    repo_type='dataset', filename='demo_actor.hdf5', local_dir='/content/data/actor').
No archive extraction or second full-size copy is needed.

Schema: data/demo_X retains the original episode. Its actor_samples group has
observation_indices [N,5] (four chronological frames, then the goal),
action_start_indices [N,3] (each starts a frame_skip-long primitive block),
goal_phase [N] (0 grasp, 1 final), and actor_action_mask [N,frame_skip].
The actor target is the LAST action block. Masked commands reach beyond the
grasp goal; they remain valid world-model targets. Never feed the future fourth
observation into the actor or treat the goal as a consecutive world-model frame.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / 'data/robot/mimicgen/demo_rechunked.hdf5'
DEFAULT_OUTPUT = ROOT / 'data/robot/actor/demo_actor.hdf5'
SIGNAL = 'datagen_info/subtask_term_signals/grasp'
OBS_KEYS = ('agentview_image', 'robot0_eye_in_hand_image',
            'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')


def episode_info(demo, frame_skip):
    """Return (exclusion reason, grasp index, later loss flag). Fail on corruption."""
    n = len(demo['actions'])
    if demo['actions'].shape != (n, 7):
        raise ValueError(f'{demo.name}: expected primitive actions [T,7]')
    for key in OBS_KEYS:
        if len(demo[f'obs/{key}']) != n:
            raise ValueError(f'{demo.name}: observation/action length mismatch')
    if SIGNAL not in demo:
        return 'missing_signal', None, False
    signal = np.asarray(demo[SIGNAL]).reshape(-1)
    if len(signal) != n or not np.isin(signal, [0, 1]).all():
        raise ValueError(f'{demo.name}: invalid grasp signal')
    flags = signal.astype(bool)
    onsets = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    if len(onsets) != 1:
        return ('no_grasp' if len(onsets) == 0 else 'multiple_grasps'), None, False
    grasp = int(onsets[0])
    if not 0 < grasp < n - 1:
        return 'invalid_goal_order', None, False
    if n <= 3 * frame_skip:
        return 'too_short', None, False
    return None, grasp, bool((~flags[grasp:]).any())


def sample_indices(n, grasp, frame_skip=5):
    starts = np.arange(n - 3 * frame_skip, dtype=np.int64)
    chronological = starts[:, None] + np.arange(4) * frame_skip
    current = chronological[:, 2]
    phase = (current >= grasp).astype(np.uint8)
    goal = np.where(phase == 0, grasp, n - 1)
    return {
        'observation_indices': np.column_stack((chronological, goal)),
        'action_start_indices': chronological[:, :3],
        'goal_phase': phase,
        'actor_action_mask': current[:, None] + np.arange(frame_skip) < goal[:, None],
    }


def scan(source, frame_skip=5):
    kept, excluded, counts = {}, {}, Counter()
    for name in tqdm(sorted(source['data']), desc='Checking episodes'):
        reason, grasp, lost = episode_info(source['data'][name], frame_skip)
        if reason:
            excluded[name] = reason
            counts[reason] += 1
        else:
            kept[name] = (grasp, lost)
            counts['kept'] += 1
            counts['kept_with_later_grasp_loss'] += int(lost)
    if not kept:
        raise ValueError('No valid episodes remain')
    for split in ('train', 'valid'):
        if f'mask/{split}' not in source:
            raise ValueError(f'Missing original {split} split; refusing to invent a split')
    train = set(source['mask/train'].asstr()[:])
    valid = set(source['mask/valid'].asstr()[:])
    if train & valid or not (train | valid) <= set(source['data']):
        raise ValueError('Invalid or overlapping train/validation episode masks')
    if not set(kept) <= train | valid:
        raise ValueError('Some retained episodes have no train/validation assignment')
    for split, names in [('train', train), ('valid', valid)]:
        counts[f'{split}_episodes'] = len(names & set(kept))
        if not counts[f'{split}_episodes']:
            raise ValueError(f'Filtering leaves an empty {split} split')
    return kept, excluded, dict(counts)


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def process(input_path, output_path, frame_skip=5, scan_only=False):
    if frame_skip < 1:
        raise ValueError('frame_skip must be positive')
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError('Output must differ from input')
    partial = output_path.with_suffix('.hdf5.partial')
    if not scan_only and (output_path.exists() or partial.exists()):
        raise FileExistsError(f'Output or partial already exists: {output_path}; use another output path')
    with h5py.File(input_path, 'r') as src:
        kept, excluded, counts = scan(src, frame_skip)
        print(json.dumps(counts, indent=2))
        if scan_only:
            return counts
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output_path.parent).free < input_path.stat().st_size * 1.1:
            raise OSError('Need free space of at least 1.1 times the source size')
        with h5py.File(partial, 'x') as dst:
            copy_attrs(src, dst)
            dst.attrs['actor_dataset_version'] = 1
            dst.attrs['frame_skip'] = frame_skip
            dst.attrs['history_transitions'] = 3
            dst.attrs['processing_summary'] = json.dumps(counts)
            dst.attrs['excluded_episodes'] = json.dumps(excluded)
            data = dst.create_group('data')
            copy_attrs(src['data'], data)
            total = 0
            for name, (grasp, lost) in tqdm(kept.items(), desc='Copying retained episodes'):
                # HDF5 copies compressed chunks directly, retaining image layout.
                src.copy(src['data'][name], data, name=name)
                demo = data[name]
                n = len(demo['actions'])
                total += n
                demo.attrs['actor_grasp_index'] = grasp
                demo.attrs['actor_goal_index'] = n - 1
                demo.attrs['actor_grasp_lost_later'] = lost
                samples = demo.create_group('actor_samples')
                for key, values in sample_indices(n, grasp, frame_skip).items():
                    samples.create_dataset(key, data=values, compression='gzip', compression_opts=1)
            data.attrs['total'] = total
            masks = dst.create_group('mask')
            copy_attrs(src['mask'], masks)
            for name, values in src['mask'].items():
                selected = [x for x in values.asstr()[:] if x in kept]
                target = masks.create_dataset(name, data=np.asarray(selected, dtype='S'))
                copy_attrs(values, target)
        # Publish only a closed, completed local file. Interrupted copies remain partial.
        partial.rename(output_path)
    print(f'Saved {output_path} ({output_path.stat().st_size / 2**30:.2f} GiB)')
    return counts


def upload(output_path, repo_id):
    from huggingface_hub import HfApi
    with h5py.File(output_path, 'r') as f:
        if f.attrs.get('actor_dataset_version') != 1:
            raise ValueError('Not a processed actor dataset')
    api = HfApi()
    api.whoami()  # Uses the machine's saved Hugging Face login (or HF_TOKEN).
    api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)
    if not api.repo_info(repo_id=repo_id, repo_type='dataset').private:
        raise ValueError('Repository is public; refusing to publish data without explicit approval')
    if 'demo_actor.hdf5' in api.list_repo_files(repo_id=repo_id, repo_type='dataset'):
        raise FileExistsError('Remote demo_actor.hdf5 already exists; refusing to overwrite it')
    api.upload_file(path_or_fileobj=str(output_path), path_in_repo='demo_actor.hdf5',
                    repo_id=repo_id, repo_type='dataset',
                    commit_message='Add filtered Lift actor dataset with precomputed goal indices')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--frame-skip', type=int, default=5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--scan-only', action='store_true')
    mode.add_argument('--upload', action='store_true', help='Process, then upload privately')
    mode.add_argument('--upload-only', action='store_true', help='Upload an already processed output')
    parser.add_argument('--repo-id', help='Upload destination: USERNAME/DATASET or ORGANIZATION/DATASET')
    args = parser.parse_args()
    if args.upload or args.upload_only:
        if not args.repo_id or len(args.repo_id.split('/')) != 2 or not all(args.repo_id.split('/')):
            parser.error('Uploads require --repo-id USERNAME/DATASET (or ORGANIZATION/DATASET)')
    if not args.upload_only:
        process(args.input, args.output, args.frame_skip, args.scan_only)
    if args.upload or args.upload_only:
        upload(args.output, args.repo_id)


if __name__ == '__main__':
    main()
