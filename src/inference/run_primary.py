"""Evaluate Lift with goal-conditioned actor-residual CEM and save rollout videos."""
import argparse
import os
import sys
import ctypes
import json
import re
import tempfile
import time
import zipfile
from collections import deque
from pathlib import Path
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.architectures.actor import GoalActor
from src.architectures.robot_encoder import RobotEncoder
from src.architectures.predictor import PredictorNetwork
from src.datasets.robot.primary_dataset import (build_robot_state,
    normalize_robot_state, normalize_robot_images)
from src.inference.actor_planner import ActorPlanner
from src.primary_contract import (CHECKPOINT_FORMAT_VERSION, ARCHITECTURE_ID,
    ACTION_ENCODING_ID, IMAGE_NORMALIZATION_ID)
WINDOW, EMBEDDING_DIM, FRAME_SKIP = 3, 192, 5
PRIMITIVE_ACTION_DIM, PREDICTOR_ACTION_DIM = 7, 35
CONTROL_FREQ, IMAGE_SIZE, ACTION_SMOOTHING_DIM = 20, 84, 64
ENV_NAME, CAM_ABOVE, CAM_WRIST = 'Lift_D0', 'agentview', 'robot0_eye_in_hand'

def setup_simulator():
    # Delay simulator imports so --help and checkpoint tests work without MuJoCo.
    global mujoco, suite
    os.environ.setdefault('MUJOCO_GL', 'egl' if sys.platform.startswith('linux') else 'glfw')
    import mujoco
    import robosuite as suite
    import robosuite.macros as macros
    from robosuite.environments.manipulation.lift import Lift
    from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG
    if 'Lift_D0' not in suite.ALL_ENVIRONMENTS:
        class Lift_D0(Lift, SingleArmEnv_MG):
            def edit_model_xml(self, xml_str):
                return SingleArmEnv_MG.edit_model_xml(self, xml_str)
            def reward(self, action=None):
                return Lift.reward(self, action=action)
    macros.IMAGE_CONVENTION = 'opencv'

def load_controller_config(*args, **kwargs):
    from robosuite.controllers import load_controller_config as load
    return load(*args, **kwargs)

def select_checkpoint(model_dir: Path) -> Path:
	best = model_dir / "best_model.zip"
	if best.exists():
		return best
	checkpoints = []
	for path in model_dir.glob("epoch_*_model.zip"):
		match = re.fullmatch(r"epoch_(\d+)_model\.zip", path.name)
		if match:
			checkpoints.append((int(match.group(1)), path))
	if not checkpoints:
		raise FileNotFoundError(f"No primary checkpoint found in {model_dir}")
	return max(checkpoints)[1]


def _load_torch(zf, member, tmp, weights_only):
	zf.extract(member, tmp)
	try:
		return torch.load(tmp / member, map_location="cpu", weights_only=weights_only)
	except TypeError:
		return torch.load(tmp / member, map_location="cpu")


def load_models(checkpoint: Path, device):
	with zipfile.ZipFile(checkpoint, "r") as zf, tempfile.TemporaryDirectory() as tmp_name:
		tmp = Path(tmp_name)
		if "state_stats.pth" not in zf.namelist():
			raise ValueError("Checkpoint has no state/action statistics; retrain it")
		stats = _load_torch(zf, "state_stats.pth", tmp, True)
		expected = {
			"checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
			"architecture_id": ARCHITECTURE_ID,
			"action_encoding_id": ACTION_ENCODING_ID,
			"image_normalization_id": IMAGE_NORMALIZATION_ID,
			"frame_skip": FRAME_SKIP,
			"primitive_action_dim": PRIMITIVE_ACTION_DIM,
			"predictor_action_dim": PREDICTOR_ACTION_DIM,
			"action_smoothing_dim": ACTION_SMOOTHING_DIM,
		}
		for key, value in expected.items():
			if stats.get(key) != value:
				raise ValueError(
					f"Incompatible checkpoint: {key}={stats.get(key)!r}, expected {value!r}. "
					"This evaluator requires a newly trained corrected checkpoint."
				)

		actor = GoalActor(**stats["actor_config"]).to(device)
		actor.load_state_dict(_load_torch(zf, "actor_weights.pth", tmp, True))
		actor.eval()
		encoder = RobotEncoder().to(device)
		predictor = PredictorNetwork(d_action=PREDICTOR_ACTION_DIM, window=WINDOW, action_smoothing_dim=ACTION_SMOOTHING_DIM).to(device)
		encoder.load_state_dict(_load_torch(zf, "encoder_weights.pth", tmp, True))
		predictor.load_state_dict(_load_torch(zf, "predictor_weights.pth", tmp, True))

	encoder.eval()
	predictor.eval()
	return encoder, predictor, actor, {key: torch.as_tensor(stats[key]).float() for key in (
		"state_mean", "state_std", "action_mean", "action_std", "action_low", "action_high"
	)}


def set_geom_groups(env, collision=False, visual=True):
	context = env.sim._render_context_offscreen
	if context is None:
		return
	address = context.vopt.geomgroup.__array_interface__["data"][0]
	groups = (ctypes.c_uint8 * 6).from_address(address)
	for index in range(6):
		groups[index] = 0
	groups[0] = int(collision)
	groups[1] = int(visual)


def observation_tensors(obs, device, stats):
	above = torch.from_numpy(obs[f"{CAM_ABOVE}_image"]).permute(2, 0, 1).float() / 255.0
	wrist = torch.from_numpy(obs[f"{CAM_WRIST}_image"]).permute(2, 0, 1).float() / 255.0
	above = normalize_robot_images(above)
	wrist = normalize_robot_images(wrist)
	state = build_robot_state(
		torch.from_numpy(obs["robot0_eef_pos"]).float(),
		torch.from_numpy(obs["robot0_eef_quat"]).float(),
		torch.from_numpy(obs["robot0_gripper_qpos"]).float(),
	)
	state = normalize_robot_state(state, stats["state_mean"], stats["state_std"])
	return above[None, None].to(device), wrist[None, None].to(device), state[None, None].to(device)


@torch.no_grad()
def encode_observation(encoder, obs, device, stats):
	return encoder(*observation_tensors(obs, device, stats))[0, 0]


@torch.no_grad()
def encode_demo_frames(encoder, obs, indices, device, stats, batch_size=128):
	outputs = []
	for offset in range(0, len(indices), batch_size):
		batch_indices = indices[offset:offset + batch_size]
		above = torch.from_numpy(obs["agentview_image"][batch_indices]).permute(0, 3, 1, 2).float() / 255.0
		wrist = torch.from_numpy(obs["robot0_eye_in_hand_image"][batch_indices]).permute(0, 3, 1, 2).float() / 255.0
		above = normalize_robot_images(above).unsqueeze(1).to(device)
		wrist = normalize_robot_images(wrist).unsqueeze(1).to(device)
		state = build_robot_state(
			torch.from_numpy(obs["robot0_eef_pos"][batch_indices]).float(),
			torch.from_numpy(obs["robot0_eef_quat"][batch_indices]).float(),
			torch.from_numpy(obs["robot0_gripper_qpos"][batch_indices]).float(),
		)
		state = normalize_robot_state(state, stats["state_mean"], stats["state_std"])
		outputs.append(encoder(above, wrist, state[:, None].to(device))[:, 0].cpu())
	return torch.cat(outputs).to(device)


def _text(value):
	return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_subgoal_labels(path):
	"""Read recorded grasp labels, not runtime simulator state or a manual JSON."""
	labels = {}
	with h5py.File(path, "r") as file:
		for name, demo in file["data"].items():
			count = len(demo["states"])
			grasp = int(demo.attrs.get("grasp_index", -1))
			if not 0 < grasp < count - 1:
				raise ValueError(f"{name}: missing or invalid recorded grasp_index: {grasp}")
			if "grasp_flags" in demo:
				flags = np.asarray(demo["grasp_flags"], dtype=bool)
				indices = np.flatnonzero(flags)
				if flags.shape != (count,) or not len(indices) or int(indices[0]) != grasp:
					raise ValueError(f"{name}: grasp_index disagrees with recorded grasp_flags")
			labels[name] = {"grasp_frame": grasp, "goal_frame": count - 1, "num_frames": count}
	return labels


@torch.no_grad()
def load_eval_bank(encoder, path, labels_path, device, stats, demo_name=None):
	labels = load_subgoal_labels(path)
	entries = []
	with h5py.File(path, "r") as file:
		if _text(file["data"].attrs.get("env", "")) != ENV_NAME:
			raise ValueError(f"{path} does not contain {ENV_NAME} evals")
		version = file["data"].attrs.get("mujoco_version")
		if version is not None and _text(version) != mujoco.__version__:
			print(f"WARNING: evals use MuJoCo {_text(version)}; inference uses {mujoco.__version__}")
		if demo_name is not None:
			if demo_name not in labels:
				raise KeyError(f"{demo_name!r} has no recorded grasp label in {labels_path}")
			names = [demo_name]
		elif "mask/eval" in file:
			all_names = [_text(value) for value in file["mask/eval"][:]]
			names = [name for name in all_names if name in labels]
		else:
			all_names = sorted(file["data"].keys())
			names = [name for name in all_names if name in labels]
		if not names:
			raise ValueError(f"No labeled eval demos from {path} were found in {labels_path}")
		if demo_name is None and len(names) != len(all_names):
			print(f"using {len(names)}/{len(all_names)} eval demos with recorded grasp labels")
		for name in names:
			demo = file[f"data/{name}"]
			num_frames = int(demo["states"].shape[0])
			label = labels[name]
			grasp_index = int(label["grasp_frame"])
			goal_index = int(label.get("goal_frame", num_frames - 1))
			if "num_frames" in label and int(label["num_frames"]) != num_frames:
				raise ValueError(
					f"{name}: labels describe {label['num_frames']} frames, eval has {num_frames}"
				)
			if not 0 < grasp_index < goal_index < num_frames:
				raise ValueError(
					f"{name}: expected 0 < grasp_frame < goal_frame < {num_frames}, "
					f"got {grasp_index} and {goal_index}"
				)
			if goal_index != num_frames - 1:
				raise ValueError(
					f"{name}: goal_frame must be the final frame ({num_frames - 1}), got {goal_index}"
				)
			embeddings = encode_demo_frames(
				encoder, demo["obs"], [0, grasp_index, goal_index], device, stats
			)
			model_xml = _text(demo.attrs["model_file"])
			entries.append({
				"name": name,
				"start": embeddings[0],
				"stages": (
					{"name": "grasp", "goal": embeddings[1], "frame": grasp_index},
					{"name": "lift", "goal": embeddings[2], "frame": goal_index},
				),
				"initial_state": np.asarray(demo["states"][0]).copy(),
				"model_xml": model_xml,
			})
	return entries


def make_env(horizon):
	controller = load_controller_config(default_controller="OSC_POSE")
	env = suite.make(
		ENV_NAME, robots="Panda", controller_configs=controller,
		has_renderer=False, has_offscreen_renderer=True,
		render_collision_mesh=False, render_visual_mesh=True,
		use_camera_obs=True, camera_names=[CAM_ABOVE, CAM_WRIST],
		camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
		control_freq=CONTROL_FREQ, horizon=horizon, ignore_done=True,
	)
	set_geom_groups(env)
	return env


def reset_to_eval_state(env, entry):
	env.reset()
	env.reset_from_xml_string(env.edit_model_xml(entry["model_xml"]))
	env.sim.reset()
	env.sim.set_state_from_flattened(entry["initial_state"])
	env.sim.forward()
	set_geom_groups(env)
	return env._get_observations(force_update=True)


def render_frame(env, size=256):
	views = [
		env.sim.render(height=size, width=size, camera_name=camera)[::-1]
		for camera in (CAM_ABOVE, CAM_WRIST)
	]
	return np.concatenate(views, axis=1).astype(np.uint8)


def postprocess_action(action):
	action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0).copy()
	action[-1] = 1.0 if action[-1] >= 0 else -1.0
	return action


def held_gripper_schedule(planner, command):
	"""Keep a specified gripper command fixed throughout a CEM plan."""
	raw = np.full((1, planner.H, planner.block_size), command, dtype=np.float32)
	return (raw - planner.raw_action_mean[-1]) / planner.raw_action_std[-1]


def run_rollout(
	env,
	planner,
	encoder,
	entry,
	device,
	stats,
	writer,
	stage_settings,
	warm_start=True,
):
	for stage in entry["stages"]:
		if stage["name"] not in stage_settings:
			raise KeyError(f"Missing settings for {stage['name']!r} stage")
		settings = stage_settings[stage["name"]]
		threshold = float(settings["threshold"])
		if int(settings["budget"]) < 1 or not np.isfinite(threshold) or threshold < 0:
			raise ValueError(f"Invalid settings for {stage['name']!r} stage")
		if float(settings["budget"]) % FRAME_SKIP:
			raise ValueError(f"Stage budgets must be multiples of {FRAME_SKIP} primitive steps")
	obs = reset_to_eval_state(env, entry)
	writer.append_data(render_frame(env))
	z = encode_observation(encoder, obs, device, stats)
	print(f"exact eval {entry['name']} | start embedding MSE: {(z-entry['start']).square().mean().item():.6f}")
	context = deque([z], maxlen=WINDOW)
	action_history = deque(maxlen=WINDOW - 1)
	total_steps = 0

	for number, stage in enumerate(entry["stages"], 1):
		# Warm starts belong to this rollout and goal; never carry grasp plans into lift.
		previous_plan = None
		goal = stage["goal"]
		budget = int(stage_settings[stage["name"]]["budget"])
		threshold = float(stage_settings[stage["name"]]["threshold"])
		use_threshold = bool(stage_settings[stage["name"]].get("use_threshold", True))
		steps = 0
		current_mse = (context[-1] - goal).square().mean().item()
		reached = use_threshold and current_mse <= threshold
		threshold_text = f"{threshold:.4f}" if use_threshold else "disabled"
		print(
			f"{stage['name']} stage {number}/{len(entry['stages'])}: "
			f"demo frame {stage['frame']}, threshold={threshold_text}"
		)
		if reached:
			print(f"  already within threshold (latent MSE {current_mse:.4f})")
		while steps < budget and not reached:
			zi = torch.stack(tuple(context))
			a = (
				torch.stack(tuple(action_history))
				if action_history else torch.empty(0, PREDICTOR_ACTION_DIM, device=device)
			)
			plan = planner.CEM(zi, a, goal)
			previous_plan = plan
			raw_blocks = planner.denormalize_action_blocks(plan)
			# Execute one ordered 5-action block, then replan from observation.
			for raw_block in raw_blocks[:1]:
				executed = []
				for raw_action in raw_block:
					action = postprocess_action(raw_action)
					obs, _, done, _ = env.step(action)
					set_geom_groups(env)
					writer.append_data(render_frame(env))
					executed.append(action)
					steps += 1
					total_steps += 1
					if env._check_success():
						return True, total_steps
					if done:
						return False, total_steps
					if done or steps >= budget:
						break

				normalized_block = planner.normalize_primitive_actions(np.stack(executed))
				if len(executed) == FRAME_SKIP:
					action_history.append(torch.as_tensor(
						normalized_block.reshape(-1), dtype=z.dtype, device=device
					))
				z = encode_observation(encoder, obs, device, stats)
				context.append(z)
				goal_mse = (z - goal).square().mean().item()
				print(f"  steps {steps:>3}/{budget} | latent MSE {goal_mse:.4f}")
				if use_threshold and goal_mse <= threshold:
					reached = True
					break
				if done or steps >= budget:
					break
		if not use_threshold:
			print(f"  fixed {stage['name']} budget complete ({steps}/{budget} steps)")
			continue
		if not reached:
			print("subgoal budget exhausted without reaching the latent goal")
			return False, total_steps
	return bool(env._check_success()), total_steps


def next_rollout_number(rollout_dir: Path) -> int:
	"""Continue after the largest completed rollout_N_pass/fail video."""
	latest = 0
	# Search anywhere in the name so videos from the earlier
	# <checkpoint>_<demo>_rollout_N_<result> convention are counted too.
	pattern = re.compile(r"(?:^|_)rollout_(\d+)_(?:pass|fail)\.mp4$")
	for path in rollout_dir.glob("*rollout_*_*.mp4"):
		match = pattern.search(path.name)
		if match:
			latest = max(latest, int(match.group(1)))
	return latest + 1

def run_evaluation(run_dir, *, checkpoint_epoch=None, num_rollouts=50,
                   eval_demo=None, seed=42, use_latent_thresholds=False,
                   grasp_budget=150, lift_budget=100, grasp_threshold=.05,
                   lift_threshold=.05, horizon=5, planner_mode='arcem',
                   temperature=.2, samples=128, elites=16, iterations=3,
                   device='cuda', evals=None):
    PRIMARY_DIR = Path(run_dir)
    MODEL_DIR = PRIMARY_DIR / 'models'
    ROLLOUT_DIR = PRIMARY_DIR / 'rollouts'
    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_HDF5 = Path(evals) if evals else PRIMARY_DIR / 'evals.hdf5'
    SUBGOAL_LABELS = EVAL_HDF5  # Labels are embedded in the evaluation episodes.
    for path in (EVAL_HDF5, SUBGOAL_LABELS):
        if not path.is_file():
            raise FileNotFoundError(path)
    CHECKPOINT_EPOCH, NUM_ROLLOUTS, EVAL_DEMO = checkpoint_epoch, num_rollouts, eval_demo
    EVAL_SEED, USE_LATENT_THRESHOLDS = seed, use_latent_thresholds
    GRASP_BUDGET, LIFT_BUDGET = grasp_budget, lift_budget
    GRASP_THRESHOLD, LIFT_THRESHOLD = grasp_threshold, lift_threshold
    HORIZON, PLANNER_MODE, RESIDUAL_TEMPERATURE = horizon, planner_mode, temperature
    NUM_SAMPLES, NUM_ELITES, CEM_ITERS, DEVICE = samples, elites, iterations, device
    setup_simulator()
    if DEVICE.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; select --device cpu for diagnostics')

    if not isinstance(HORIZON, int) or isinstance(HORIZON, bool) or HORIZON < 1:
        raise ValueError('HORIZON must be a positive integer number of 5-action blocks')
    PLAN_STEPS = HORIZON * FRAME_SKIP
    print(f'lookahead: {PLAN_STEPS} primitive steps; execute {FRAME_SKIP} before replanning')

    if CHECKPOINT_EPOCH is None:
        CHECKPOINT = select_checkpoint(MODEL_DIR)
    else:
        if not isinstance(CHECKPOINT_EPOCH, int) or isinstance(CHECKPOINT_EPOCH, bool) or CHECKPOINT_EPOCH < 0:
            raise ValueError('CHECKPOINT_EPOCH must be a nonnegative integer or None')
        CHECKPOINT = MODEL_DIR / f'epoch_{CHECKPOINT_EPOCH}_model.zip'
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f'Requested epoch {CHECKPOINT_EPOCH}, but checkpoint does not exist: {CHECKPOINT}'
            )

    # Reload every model-dependent object in this cell. This is important because
    # goal embeddings from one encoder must never be reused with another epoch.
    encoder, predictor, actor, STATS = load_models(CHECKPOINT, DEVICE)
    print(f'checkpoint: {CHECKPOINT}')

    for stage_name, budget, threshold in (
        ('grasp', GRASP_BUDGET, GRASP_THRESHOLD),
        ('lift', LIFT_BUDGET, LIFT_THRESHOLD),
    ):
        if budget < 1:
            raise ValueError(f'{stage_name.upper()}_BUDGET must be positive')
        if threshold < 0:
            raise ValueError(f'{stage_name.upper()}_THRESHOLD must be nonnegative')
        if budget % FRAME_SKIP:
            raise ValueError(f'{stage_name.upper()}_BUDGET must be a multiple of {FRAME_SKIP}')

    primitive_mean = STATS['action_mean'].numpy()
    primitive_std = STATS['action_std'].numpy()
    normalized_low = ((STATS['action_low'] - STATS['action_mean']) / STATS['action_std']).numpy()
    normalized_high = ((STATS['action_high'] - STATS['action_mean']) / STATS['action_std']).numpy()
    planner = ActorPlanner(
        DEVICE, encoder, predictor, actor=actor, mode=PLANNER_MODE, temperature=RESIDUAL_TEMPERATURE,
        H=HORIZON, N=NUM_SAMPLES, K=NUM_ELITES, T=CEM_ITERS,
        action_dim=PREDICTOR_ACTION_DIM,
        primitive_action_dim=PRIMITIVE_ACTION_DIM,
        action_low=np.tile(normalized_low, FRAME_SKIP),
        action_high=np.tile(normalized_high, FRAME_SKIP),
        raw_action_mean=primitive_mean,
        raw_action_std=primitive_std,
        min_std=0.05,
        momentum=0.1,
        seed=EVAL_SEED,
        noise_correlation=0.0, fresh_fraction=0.0, separate_gripper_searches=False,
    )
    entries = load_eval_bank(
        encoder, EVAL_HDF5, SUBGOAL_LABELS, DEVICE, STATS, EVAL_DEMO
    )
    stage_settings = {
        'grasp': {
            'budget': GRASP_BUDGET,
            'threshold': GRASP_THRESHOLD,
            'use_threshold': USE_LATENT_THRESHOLDS,
        },
        'lift': {
            'budget': LIFT_BUDGET,
            'threshold': LIFT_THRESHOLD,
            'use_threshold': USE_LATENT_THRESHOLDS,
        },
    }
    entries.sort(key=lambda entry: int(re.fullmatch(r'demo_(\d+)', entry['name']).group(1)))
    if not isinstance(NUM_ROLLOUTS, int) or isinstance(NUM_ROLLOUTS, bool) or NUM_ROLLOUTS < 1:
        raise ValueError('NUM_ROLLOUTS must be a positive integer')
    if EVAL_DEMO is None:
        if NUM_ROLLOUTS > len(entries):
            raise ValueError(f'Requested {NUM_ROLLOUTS} unique evals, but only {len(entries)} labeled demos are available')
        indices = range(NUM_ROLLOUTS)
    else:
        indices = [0] * NUM_ROLLOUTS
    print(f'Evaluating {NUM_ROLLOUTS} rollouts in demo order: {entries[indices[0]]["name"]} through {entries[indices[-1]]["name"]}')
    env = make_env(max(200, GRASP_BUDGET + LIFT_BUDGET))

    first_rollout_number = next_rollout_number(ROLLOUT_DIR)
    print(
        f'completed rollout videos found through #{first_rollout_number - 1}; '
        f'starting at rollout_{first_rollout_number}'
    )
    successes = 0
    started = time.perf_counter()
    try:
        for batch_index, index in enumerate(indices):
            rollout_number = first_rollout_number + batch_index
            entry = entries[int(index)]
            pending = ROLLOUT_DIR / f'rollout_{rollout_number}_pending.mp4'
            pending.unlink(missing_ok=True)
            writer = imageio.get_writer(str(pending), fps=CONTROL_FREQ, codec='libx264')
            try:
                success, steps = run_rollout(
                    env, planner, encoder, entry, DEVICE, STATS, writer, stage_settings,
                    warm_start=False,
                )
            finally:
                writer.close()
            result = ROLLOUT_DIR / (
                f"rollout_{rollout_number}_{'pass' if success else 'fail'}.mp4"
            )
            result.unlink(missing_ok=True)
            pending.replace(result)
            successes += int(success)
            print(
                f"batch {batch_index + 1}/{NUM_ROLLOUTS} → rollout_{rollout_number}: "
                f"{'pass' if success else 'fail'} ({steps} steps, eval {entry['name']})"
            )
            print(f'video: {result}')
    finally:
        env.close()
    print(f'success rate: {successes}/{NUM_ROLLOUTS} = {100*successes/NUM_ROLLOUTS:.1f}%')
    print(f'total inference time: {time.perf_counter()-started:.2f}s')

    return successes, NUM_ROLLOUTS

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/primary_actor')
    parser.add_argument('--checkpoint-epoch', type=int)
    parser.add_argument('--num-rollouts', type=int, default=50)
    parser.add_argument('--eval-demo')
    parser.add_argument('--evals', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--planner-mode', choices=('arcem','direct','cem'), default='arcem')
    parser.add_argument('--use-latent-thresholds', action='store_true')
    for name, default in [('grasp-budget',150), ('lift-budget',100), ('horizon',5),
                          ('samples',128), ('elites',16), ('iterations',3)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in [('grasp-threshold',.05), ('lift-threshold',.05), ('temperature',.2)]:
        parser.add_argument('--' + name, type=float, default=default)
    run_evaluation(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()
