"""
Add delta_eef_action key to a robomimic HDF5 dataset.

Mixed label approach:
  delta_eef_action[t, :3]  = (next_eef_pos - eef_pos) / output_max_pos   -- real EEF position delta
  delta_eef_action[t, 3:7] = actions[t, 3:7]                               -- OSC rotation + gripper (unchanged)

Position uses actual EEF displacement (eliminates action→trajectory mapping error).
Rotation and gripper use original OSC signals (preserves open-loop replay fidelity).
"""

import argparse
import json
import os
import shutil
import h5py
import numpy as np
from tqdm import tqdm


def add_delta_eef_label(
    dataset_path: str,
    output_path: str | None = None,
    n_demos: int | None = None,
) -> None:
    if output_path is None:
        output_path = dataset_path
        mode = "r+"
    else:
        if os.path.abspath(output_path) != os.path.abspath(dataset_path):
            print(f"Copying {dataset_path} -> {output_path} ...")
            shutil.copy2(dataset_path, output_path)
        mode = "r+"

    f = h5py.File(output_path, mode="r+")

    # Read controller config to get output_max scalars
    env_args = json.loads(f["data"].attrs["env_args"])
    cc = env_args["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    output_max = np.asarray(cc["output_max"], dtype=np.float64)
    output_min = np.asarray(cc["output_min"], dtype=np.float64)
    # scale = (output_max - output_min) / 2 = output_max (symmetric about 0)
    action_scale = (output_max - output_min) / 2.0
    print(f"output_max  = {output_max}")
    print(f"action_scale = {action_scale}")

    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    if n_demos is not None:
        demos = demos[:n_demos]
    print(f"Processing {len(demos)} demos...")

    total_steps = 0
    pos_abs_max = 0.0
    pos_violations = 0

    for demo_name in tqdm(demos, desc="Adding delta_eef_action"):
        grp = f[f"data/{demo_name}"]

        # Read required data
        eef_pos = grp["obs/robot0_eef_pos"][:]          # [T, 3]
        next_eef_pos = grp["next_obs/robot0_eef_pos"][:] # [T, 3]
        actions = grp["actions"][:]                       # [T, 7]

        T = eef_pos.shape[0]
        assert next_eef_pos.shape[0] == T
        assert actions.shape[0] == T

        delta_eef = np.zeros((T, 7), dtype=np.float64)

        # Position delta: metres -> OSC units
        dpos = next_eef_pos - eef_pos
        delta_eef[:, :3] = dpos / action_scale[0]  # /0.05

        # Rotation + gripper: copy from original OSC actions
        delta_eef[:, 3:7] = actions[:, 3:7]

        # Track statistics
        pos_abs_max = max(pos_abs_max, np.max(np.abs(delta_eef[:, :3])))
        pos_violations += int(np.sum(np.abs(delta_eef[:, :3]) > 1.0))
        total_steps += T

        # Write to HDF5 (replace if exists, create otherwise)
        if "delta_eef_action" in grp:
            del grp["delta_eef_action"]
        grp.create_dataset("delta_eef_action", data=delta_eef)

    f.close()

    print(f"\nDone.  Total steps: {total_steps}")
    print(f"  Position norm max:  {pos_abs_max:.4f}  "
          f"(violations: {pos_violations}/{total_steps})")
    print(f"  (Rotation + gripper copied from original actions)")


def main():
    parser = argparse.ArgumentParser(
        description="Add delta_eef_action (real EEF delta in OSC units) to robomimic HDF5 dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the source HDF5 dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output HDF5. If not provided, modifies source in-place.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) only process first n demos",
    )
    args = parser.parse_args()
    add_delta_eef_label(args.dataset, output_path=args.output, n_demos=args.n)


if __name__ == "__main__":
    main()
