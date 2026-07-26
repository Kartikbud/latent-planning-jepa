"""
Teleop data collection for robosuite Stack (Panda) — MimicGen-compatible.

SpaceMouse version of collect_human_data_keyb.py.
Success = task success held briefly AND episode length < 600.
Candidates are shown as a review video; approve with y / reject with n.
Approved demos are saved under data/robot/raw_demo/.

Controls:
  - Move / twist SpaceMouse for XYZ + yaw (roll/pitch disabled)
  - Left button (click) = toggle gripper open/close
  - Right button = abort / reset episode
  - Review: y/n keys, or left=approve / right=reject
"""

import argparse
import ctypes
import datetime
import json
import os
import select
import subprocess
import sys
import tempfile
import time
from glob import glob
from pathlib import Path

import h5py
import imageio
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.devices import SpaceMouse
from robosuite.utils.input_utils import input2action

macros.IMAGE_CONVENTION = "opencv"

MAX_STEPS = 600
SAVE_DIR = Path(__file__).resolve().parents[3] / "data" / "robot" / "raw_demo"
CAMERA = "agentview"
SUCCESS_HOLD = 10  # consecutive success steps before ending episode
ENV_NAME = "Stack"


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


def show_ee_guide(env):
    """Show EE grip site + green pointing cylinder for live teleop only.

    grip_site turns red→green as it nears cubeA; grip_cylinder shows EE z-axis.
    """
    robot = env.robots[0]
    env.sim.model.site_rgba[robot.eef_cylinder_id] = np.array([0.0, 1.0, 0.0, 0.3])
    env.sim.model.site_rgba[robot.eef_site_id] = np.array([1.0, 0.0, 0.0, 0.5])
    env._visualize_gripper_to_target(gripper=robot.gripper, target=env.cubeA)


def make_env():
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=ENV_NAME,
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
        "env_name": ENV_NAME,
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }


def collect_episode(env, device, record=True):
    """Run one teleop episode. Returns (states, actions, frames, success)."""
    env.reset()
    set_geom_groups(env, collision=False, visual=True)
    hide_overlays(env)
    show_ee_guide(env)
    env.render()
    device.start_control()

    states = [np.array(env.sim.get_state().flatten())] if record else []
    actions = []
    frames = []
    hold = -1
    gripper_closed = False
    prev_held = False
    steps = 0

    while steps < MAX_STEPS:
        action, _ = input2action(device=device, robot=env.robots[0], active_arm="right")
        if action is None:  # right button reset
            break

        # Keep only yaw (z) rotation; zero roll/pitch for easier tabletop teleop.
        # OSC_POSE action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = np.array(action, dtype=np.float64, copy=True)
        action[3] = 0.0
        action[4] = 0.0

        # Left-button rising edge toggles gripper (still binary ±1 in the action).
        held = bool(device.single_click_and_hold)
        if held and not prev_held:
            gripper_closed = not gripper_closed
        prev_held = held
        action[-1] = 1.0 if gripper_closed else -1.0

        env.step(action)
        steps += 1

        # live window: show EE guide
        show_ee_guide(env)
        env.render()

        if record:
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
    if record and len(states) == len(actions) + 1:
        states = states[:-1]

    # ensure saved model_file has sites hidden
    hide_overlays(env)

    success = env._check_success() and steps < MAX_STEPS and steps > 0
    return states, actions, frames, success


def review_video(frames, device=None):
    if not frames:
        return False
    path = os.path.join(tempfile.gettempdir(), "stack_demo_review.mp4")
    writer = imageio.get_writer(path, fps=20)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"\nReview video: {path}")
    try:
        subprocess.run(["open", path], check=False)
    except Exception:
        pass

    print("Approve this demo? [y/n]  |  SpaceMouse: left=yes, right=no")

    # Re-enable device (right-button abort disables it) and wait for a clean release.
    if device is not None:
        device.start_control()
        while device.single_click_and_hold or device._reset_state:
            if device._reset_state:
                device.start_control()
            time.sleep(0.05)
        prev_held = False

    while True:
        if select.select([sys.stdin], [], [], 0.05)[0]:
            ans = sys.stdin.readline().strip().lower()
            if ans in ("y", "n"):
                return ans == "y"

        if device is None:
            continue

        held = bool(device.single_click_and_hold)
        if held and not prev_held:
            print("Approved (left button).")
            return True
        prev_held = held

        if device._reset_state:
            print("Rejected (right button).")
            device.start_control()  # re-enable for the next episode
            return False


def next_demo_index(save_dir):
    existing = sorted(glob(str(save_dir / "demo_*.hdf5")))
    if not existing:
        return 0
    last = Path(existing[-1]).stem  # demo_012
    return int(last.split("_")[1]) + 1


def save_demo(save_dir, idx, states, actions, model_xml, env_info):
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
        grp.attrs["env"] = ENV_NAME
        grp.attrs["env_info"] = json.dumps(env_info)
        grp.attrs["total"] = len(actions)
    print(f"Saved {path} ({len(actions)} steps)")
    return path


def count_existing_demos(save_dir):
    return len(glob(str(save_dir / "demo_*.hdf5")))


def main():
    parser = argparse.ArgumentParser(description="Collect Stack teleop demos with SpaceMouse")
    parser.add_argument(
        "num_episodes",
        nargs="?",
        type=int,
        default=10,
        help="Target total approved demos in the save folder (default: 10)",
    )
    parser.add_argument("--pos_sensitivity", type=float, default=1.0)
    parser.add_argument("--rot_sensitivity", type=float, default=1.0)
    parser.add_argument(
        "--fun",
        action="store_true",
        help="Practice teleop only — no recording, review, or saving.",
    )
    args = parser.parse_args()

    env, env_info = make_env()
    device = SpaceMouse(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )

    print(f"SpaceMouse teleop. Right button aborts / resets. Success held for {SUCCESS_HOLD} steps ends episode.")
    print("Left button click toggles gripper. Roll/pitch disabled — yaw + XYZ only.")
    print("Task: pick cubeA and place it on cubeB.")

    if args.fun:
        print("\nFUN MODE — nothing will be saved. Ctrl+C to quit.")
        try:
            while True:
                print("\n=== Practice trial (right button to reset) ===")
                _, _, _, success = collect_episode(env, device, record=False)
                print("Success!" if success else "Reset / incomplete. New randomization.")
        except KeyboardInterrupt:
            print("\nDone (fun mode).")
        finally:
            env.close()
        return

    existing = count_existing_demos(SAVE_DIR)
    if existing >= args.num_episodes:
        print(f"Already have {existing} demos in {SAVE_DIR} (target {args.num_episodes}). Nothing to do.")
        env.close()
        return

    print(f"Resuming Stack collection → {SAVE_DIR}")
    print(f"Already have {existing}/{args.num_episodes}; need {args.num_episodes - existing} more.")

    saved = existing
    try:
        while saved < args.num_episodes:
            print(f"\n=== Trial for demo {saved + 1}/{args.num_episodes} ===")
            states, actions, frames, success = collect_episode(env, device, record=True)
            model_xml = env.sim.model.get_xml()

            if not success:
                reason = f"too long (>= {MAX_STEPS})" if len(actions) >= MAX_STEPS else "task not succeeded / aborted"
                print(f"Rejected automatically ({reason}, steps={len(actions)}). Resetting.")
                continue

            print(f"Candidate OK ({len(actions)} steps). Opening review video...")
            if not review_video(frames, device):
                print("Rejected by review. Resetting with new randomization.")
                continue

            idx = next_demo_index(SAVE_DIR)
            save_demo(SAVE_DIR, idx, states, actions, model_xml, env_info)
            saved += 1
            print(f"Progress: {saved}/{args.num_episodes} saved. Next trial randomized on reset.")
    finally:
        env.close()

    print(f"\nDone. {saved} demos in {SAVE_DIR}")


if __name__ == "__main__":
    main()
