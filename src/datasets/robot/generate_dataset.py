"""
Prepare processed Stack or Lift demos for MimicGen and run data generation.

Stack defaults (unchanged):
  source:  data/robot/processed/demo.hdf5
  config:  data/robot/stack_mg_config.json
  output:  data/robot/mimicgen/

Camera visuals: MimicGen renders through robomimic's EnvRobosuite wrapper,
which applies the collision-geom ctypes fix so generated images use normal
visual meshes (not the green/blue collision overlay).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MIMICGEN_ROOT = ROOT / "mimicgen"
TASKS = {
    "stack": {
        "source": ROOT / "data" / "robot" / "processed" / "demo.hdf5",
        "config": ROOT / "data" / "robot" / "stack_mg_config.json",
        "out_dir": ROOT / "data" / "robot" / "mimicgen",
        "experiment": "stack_from_human",
        "template": MIMICGEN_ROOT / "mimicgen" / "exps" / "templates" / "robosuite" / "stack.json",
        "env_name": "Stack_D0",
        "interface": "MG_Stack",
    },
    "lift": {
        "source": ROOT / "data" / "robot" / "processed" / "demo.hdf5",
        "config": ROOT / "data" / "robot" / "lift_mg_config.json",
        "out_dir": ROOT / "data" / "robot" / "mimicgen",
        "experiment": "lift_from_human",
        "template": MIMICGEN_ROOT / "mimicgen" / "exps" / "templates" / "robosuite" / "lift.json",
        "env_name": "Lift_D0",
        "interface": "MG_Lift",
    },
}


def run(cmd):
    print("Running:", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=str(MIMICGEN_ROOT))


def write_train_val_mask(demo_hdf5: Path, train_ratio: float = 0.9, seed: int = 0):
    """Write robomimic-style mask/train + mask/valid (90/10 by default)."""
    with h5py.File(demo_hdf5, "a") as f:
        demos = sorted(
            f["data"].keys(),
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),
        )
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(demos))
        n_train = int(len(demos) * train_ratio)
        train = [demos[i] for i in order[:n_train]]
        valid = [demos[i] for i in order[n_train:]]
        if "mask" in f:
            del f["mask"]
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.array(train, dtype="S"))
        mask.create_dataset("valid", data=np.array(valid, dtype="S"))
    print(f"Wrote mask → train={len(train)} valid={len(valid)} in {demo_hdf5}")


def cleanup_mimicgen_output(out_dir: Path, exp_name: str):
    """Drop failed demos/playback and flatten MimicGen's nested exp folder into out_dir."""
    exp_dir = out_dir / exp_name
    if not exp_dir.is_dir():
        print(f"No nested experiment folder to clean: {exp_dir}")
        return

    for name in ("demo_failed.hdf5", f"playback_{exp_name}_failed.mp4"):
        path = exp_dir / name
        if path.exists():
            path.unlink()
            print(f"Deleted {path}")

    for item in exp_dir.iterdir():
        dest = out_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))

    exp_dir.rmdir()
    print(f"Flattened {exp_dir} → {out_dir}")

    demo_hdf5 = out_dir / "demo.hdf5"
    if demo_hdf5.exists():
        write_train_val_mask(demo_hdf5)


def main():
    parser = argparse.ArgumentParser(description="Run MimicGen generation from processed Stack or Lift demos")
    parser.add_argument(
        "num_trials",
        nargs="?",
        type=int,
        default=1000,
        help="Number of successful demos to generate (default: 1000)",
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="stack",
        help="Task to generate (default: stack)",
    )
    args = parser.parse_args()
    task = TASKS[args.task]
    source = task["source"]
    config = task["config"]
    out_dir = task["out_dir"]

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if not task["template"].exists():
        raise FileNotFoundError(f"MimicGen template not found: {task['template']}")

    # 1) annotate source with datagen_info
    run([
        sys.executable,
        str(MIMICGEN_ROOT / "mimicgen" / "scripts" / "prepare_src_dataset.py"),
        "--dataset", str(source),
        "--env_interface", task["interface"],
        "--env_interface_type", "robosuite",
    ])

    # 2) write config from the selected task template
    cfg = json.loads(task["template"].read_text())
    cfg["experiment"]["name"] = task["experiment"]
    cfg["experiment"]["source"]["dataset_path"] = str(source)
    cfg["experiment"]["generation"]["path"] = str(out_dir)
    cfg["experiment"]["generation"]["guarantee"] = True
    cfg["experiment"]["generation"]["num_trials"] = args.num_trials
    cfg["experiment"]["task"]["name"] = task["env_name"]
    cfg["experiment"]["task"]["interface"] = task["interface"]
    cfg["experiment"]["task"]["interface_type"] = "robosuite"
    cfg["obs"]["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    cfg["obs"]["camera_height"] = 84
    cfg["obs"]["camera_width"] = 84
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(cfg, indent=2))
    print(f"Wrote config → {config}")

    # 3) generate
    run([
        sys.executable,
        str(MIMICGEN_ROOT / "mimicgen" / "scripts" / "generate_dataset.py"),
        "--config", str(config),
        "--auto-remove-exp",
    ])

    # 4) drop failures, flatten into data/robot/mimicgen/, write train/valid mask
    cleanup_mimicgen_output(out_dir, task["experiment"])
    print(f"Done. Generated {args.task} data under {out_dir}")


if __name__ == "__main__":
    main()
