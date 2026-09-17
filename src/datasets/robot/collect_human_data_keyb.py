"""
Teleop data collection for robosuite Stack or Lift (Panda) — MimicGen-compatible.

Success = task success held briefly AND episode length < 600.
Candidates are shown as a review video; approve with y / reject with n.
Approved demos are saved under data/robot/raw_demo/.
"""

import argparse
import ctypes
import datetime
import json
import os
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import h5py
import imageio
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action

macros.IMAGE_CONVENTION = "opencv"

MAX_STEPS = 600
ROOT = Path(__file__).resolve().parents[3]
CAMERA = "agentview"
SUCCESS_HOLD = 10  # consecutive success steps before ending episode
TASKS = {
    "stack": {
        "env_name": "Stack",
        "object_attr": "cubeA",
        "save_dir": ROOT / "data" / "robot" / "raw_demo",
        "instruction": "pick cubeA and place it on cubeB",
    },
    "lift": {
        "env_name": "Lift",
        "object_attr": "cube",
        "save_dir": ROOT / "data" / "robot" / "raw_demo",
        "instruction": "grasp and lift the cube",
    },
}


def set_geom_groups(env, collision=False, visual=True):
    """Hide collision geoms / show visual meshes.

    MuJoCo 2.3.2 + NumPy 2.x exposes vopt.geomgroup with stride 0, so
    ``geomgroup[i] = ...`` writes every group at once and robosuite's
    ``render_collision_mesh=False`` is a no-op. Write the underlying bytes
    via ctypes instead.
    """
    ctx = env.sim._render_context_offscreen
    if ctx is None:
        return
    addr = ctx.vopt.geomgroup.__array_interface__["data"][0]
    groups = (ctypes.c_uint8 * 6).from_address(addr)
    for i in range(6):
        groups[i] = 0
    groups[0] = int(collision)
    groups[1] = int(visual)


def hide_overlays(env):
    """Hide teleop sites so they never appear in review frames / saved XML."""
    env.sim.model.site_rgba[:, 3] = 0.0


def show_ee_guide(env, object_attr):
    """Show EE grip site + green pointing cylinder for live teleop only.

    grip_site turns red→green as it nears the task object; grip_cylinder shows
    the EE z-axis.
    """
    robot = env.robots[0]
    env.sim.model.site_rgba[robot.eef_cylinder_id] = np.array([0.0, 1.0, 0.0, 0.3])
    env.sim.model.site_rgba[robot.eef_site_id] = np.array([1.0, 0.0, 0.0, 0.5])
    env._visualize_gripper_to_target(
        gripper=robot.gripper,
        target=getattr(env, object_attr),
    )


def make_env(env_name):
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,  # needed for geomgroup / visual meshes
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_camera=CAMERA,
        ignore_done=True,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=MAX_STEPS,
    )
    set_geom_groups(env, collision=False, visual=True)
    return env, {
        "env_name": env_name,
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }


def collect_episode(env, device, object_attr):
    """Run one teleop episode. Returns (states, actions, frames, success)."""
    env.reset()
    set_geom_groups(env, collision=False, visual=True)
    hide_overlays(env)
    show_ee_guide(env, object_attr)
    env.render()
    device.start_control()

    states = [np.array(env.sim.get_state().flatten())]
    actions = []
    frames = []
    hold = -1

    while len(actions) < MAX_STEPS:
        action, _ = input2action(device=device, robot=env.robots[0], active_arm="right")
        if action is None:  # q / reset
            break

        env.step(action)

        # live window: show EE guide
        show_ee_guide(env, object_attr)
        env.render()

        # review / data path: hide overlays first so they are not recorded
        hide_overlays(env)
        actions.append(np.array(action))
        states.append(np.array(env.sim.get_state().flatten()))
        frames.append(np.array(env.sim.render(height=256, width=256, camera_name=CAMERA)))

        if env._check_success():
            if hold > 0:
                hold -= 1
            else:
                hold = SUCCESS_HOLD
            if hold == 0:
                break
        else:
            hold = -1

    # drop trailing state (robosuite convention: one state after last action)
    if len(states) == len(actions) + 1:
        states = states[:-1]

    # ensure saved model_file has sites hidden
    hide_overlays(env)

    success = env._check_success() and len(actions) < MAX_STEPS and len(actions) > 0
    return states, actions, frames, success


def review_video(frames, task):
    if not frames:
        return False
    path = os.path.join(tempfile.gettempdir(), f"{task}_demo_review.mp4")
    writer = imageio.get_writer(path, fps=20)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"\nReview video: {path}")
    try:
        subprocess.run(["open", path], check=False)
    except Exception:
        pass
    while True:
        ans = input("Approve this demo? [y/n]: ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"


def next_demo_index(save_dir):
    existing = sorted(glob(str(save_dir / "demo_*.hdf5")))
    if not existing:
        return 0
    last = Path(existing[-1]).stem  # demo_012
    return int(last.split("_")[1]) + 1


def save_demo(save_dir, idx, states, actions, model_xml, env_info, env_name):
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"demo_{idx:03d}.hdf5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        demo = grp.create_group("demo_0")
        demo.attrs["model_file"] = model_xml
        demo.attrs["num_samples"] = len(actions)
        demo.create_dataset("states", data=np.array(states))
        demo.create_dataset("actions", data=np.array(actions))
        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = json.dumps(env_info)
        grp.attrs["total"] = len(actions)
    print(f"Saved {path} ({len(actions)} steps)")
    return path


def count_existing_demos(save_dir):
    return len(glob(str(save_dir / "demo_*.hdf5")))


def main():
    parser = argparse.ArgumentParser(description="Collect Stack or Lift teleop demos")
    parser.add_argument(
        "num_episodes",
        nargs="?",
        type=int,
        default=10,
        help="Target total approved demos in the save folder (default: 10)",
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="stack",
        help="Robosuite task to collect (default: stack)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the task-specific output directory",
    )
    args = parser.parse_args()
    task = TASKS[args.task]
    save_dir = args.output_dir or task["save_dir"]

    existing = count_existing_demos(save_dir)
    if existing >= args.num_episodes:
        print(f"Already have {existing} demos in {save_dir} (target {args.num_episodes}). Nothing to do.")
        return

    env, env_info = make_env(task["env_name"])
    device = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)

    print(f"Resuming {args.task.title()} collection → {save_dir}")
    print(f"Already have {existing}/{args.num_episodes}; need {args.num_episodes - existing} more.")
    print(f"Keyboard teleop. Press q to abort an episode. Success must be < {MAX_STEPS} steps.")
    print(f"Task: {task['instruction']}.")

    saved = existing
    try:
        while saved < args.num_episodes:
            print(f"\n=== Trial for demo {saved + 1}/{args.num_episodes} ===")
            states, actions, frames, success = collect_episode(
                env, device, task["object_attr"]
            )
            model_xml = env.sim.model.get_xml()

            if not success:
                reason = f"too long (>= {MAX_STEPS})" if len(actions) >= MAX_STEPS else "task not succeeded / aborted"
                print(f"Rejected automatically ({reason}, steps={len(actions)}). Resetting.")
                continue

            print(f"Candidate OK ({len(actions)} steps). Opening review video...")
            if not review_video(frames, args.task):
                print("Rejected by review. Resetting with new randomization.")
                continue

            idx = next_demo_index(save_dir)
            save_demo(
                save_dir, idx, states, actions, model_xml, env_info, task["env_name"]
            )
            saved += 1
            print(f"Progress: {saved}/{args.num_episodes} saved. Next trial randomized on reset.")
    finally:
        env.close()

    print(f"\nDone. {saved} demos in {save_dir}")


if __name__ == "__main__":
    main()
