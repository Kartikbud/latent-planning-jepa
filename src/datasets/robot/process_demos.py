"""
Merge per-demo raw HDF5s and convert with robomimic.

Keeps demo_XXX.hdf5 in the raw folder untouched; writes processed demo.hdf5
to the selected task's processed directory (or --out_dir).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[3]
TASK_PATHS = {
    "stack": {
        "raw": ROOT / "data" / "robot" / "raw_demo",
        "processed": ROOT / "data" / "robot" / "processed",
    },
    "lift": {
        "raw": ROOT / "data" / "robot" / "raw_demo",
        "processed": ROOT / "data" / "robot" / "processed",
    },
}


def merge_demos(dataset_dir: Path, out_dir: Path, out_name: str = "demo.hdf5") -> Path:
    files = sorted(dataset_dir.glob("demo_*.hdf5"))
    files = [p for p in files if p.name != out_name]
    if not files:
        raise FileNotFoundError(f"No demo_*.hdf5 files found in {dataset_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name
    with h5py.File(out, "w") as dst:
        dgrp = dst.create_group("data")
        for i, path in enumerate(files):
            with h5py.File(path, "r") as src:
                if i == 0:
                    for k, v in src["data"].attrs.items():
                        dgrp.attrs[k] = v
                src.copy("data/demo_0", dgrp, name=f"demo_{i}")

    print(f"Merged {len(files)} demos → {out}")
    return out


def convert_robosuite(dataset_path: Path) -> None:
    script = (
        ROOT / "robomimic" / "robomimic" / "scripts" / "conversion" / "convert_robosuite.py"
    )
    if not script.exists():
        raise FileNotFoundError(f"convert_robosuite.py not found at {script}")

    cmd = [sys.executable, str(script), "--dataset", str(dataset_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Stack or Lift demos and convert them with robomimic"
    )
    parser.add_argument(
        "--task",
        choices=TASK_PATHS,
        default="stack",
        help="Task whose default input and output paths should be used (default: stack)",
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Directory with demo_*.hdf5 (default: selected task's raw directory)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Where to write merged demo.hdf5 (default: selected task's processed directory)",
    )
    args = parser.parse_args()
    paths = TASK_PATHS[args.task]
    dataset_dir = (args.dataset_dir or paths["raw"]).expanduser().resolve()
    out_dir = (args.out_dir or paths["processed"]).expanduser().resolve()

    merged = merge_demos(dataset_dir, out_dir)
    convert_robosuite(merged)
    print(f"Done. Raw demos preserved in {dataset_dir}; processed file: {merged}")


if __name__ == "__main__":
    main()
