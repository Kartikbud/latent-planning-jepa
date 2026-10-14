# Simulation dependency patches

The submodules remain pinned upstream repositories with their original licenses.
Run `python scripts/apply_sim_patches.py` after initializing them. It checks the
upstream commits, accepts patches already applied, and refuses conflicting edits.

- `mimicgen.patch`: registers `Lift_D0`, adds `MG_Lift` (cube pose and grasp signal),
  and adds the object-relative Lift generation configuration/template.
- `robomimic.patch`: preserves the local camera visual/collision geometry fix.
- Robosuite needs no source patch. Its local recorded demonstrations are data,
  not required source code, and are intentionally not included.

The exact upstream commit IDs are recorded in `scripts/apply_sim_patches.py`
and the repository's submodule entries. Patches derived from dependency code
remain subject to the respective upstream licenses, including MimicGen's NVIDIA
Source Code License. No original license notices are removed.
