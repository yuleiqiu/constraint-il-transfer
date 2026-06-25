"""
Validate delta_eef_action labels in an HDF5 dataset.

Checks per step:
  1. shape/dtype consistency
  2. position reconstruction: delta_eef[:,:3] * action_scale == next_pos - obs_pos
  3. rotation+gripper match original actions exactly: delta_eef[:,3:7] == actions[:,3:7]

Usage:
    uv run python scripts/validate_delta_eef_dataset.py \
        --dataset third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5
"""

import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm


def validate_delta_eef_dataset(dataset_path: str) -> dict:
    f = h5py.File(dataset_path, "r")

    env_args = json.loads(f["data"].attrs["env_args"])
    cc = env_args["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    output_max = np.asarray(cc["output_max"], dtype=np.float64)
    output_min = np.asarray(cc["output_min"], dtype=np.float64)
    action_scale_val = (output_max - output_min) / 2.0

    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    n_demos = len(demos)

    results = dict(
        n_demos=n_demos,
        total_steps=0,
        action_scale=list(action_scale_val),
        shape_ok=True,
        rot_grip_ok=True,
        pos_max_error=0.0,
        pos_error_per_demo=[],
        violations_pos=0,
        demo_errors=[],
    )

    for demo_name in tqdm(demos, desc="Validating"):
        grp = f[f"data/{demo_name}"]
        delta = grp["delta_eef_action"][:]
        actions = grp["actions"][:]
        obs_pos = grp["obs/robot0_eef_pos"][:]
        next_pos = grp["next_obs/robot0_eef_pos"][:]
        obs_quat = grp["obs/robot0_eef_quat"][:]
        next_quat = grp["next_obs/robot0_eef_quat"][:]

        T = delta.shape[0]

        # shape
        if delta.shape != actions.shape:
            results["shape_ok"] = False
            results["demo_errors"].append(f"{demo_name}: shape mismatch delta={delta.shape} actions={actions.shape}")
            continue

        results["total_steps"] += T

        # rotation + gripper match
        if not np.allclose(delta[:, 3:7], actions[:, 3:7]):
            results["rot_grip_ok"] = False
            results["demo_errors"].append(f"{demo_name}: rot/grip mismatch")
            continue

        # position reconstruction
        dpos_actual = next_pos - obs_pos
        dpos_from_label = delta[:, :3] * action_scale_val[0]
        pos_err = np.max(np.abs(dpos_from_label - dpos_actual))
        results["pos_max_error"] = max(results["pos_max_error"], pos_err)
        results["pos_error_per_demo"].append(float(pos_err))

        # violations
        results["violations_pos"] += int(np.sum(np.abs(delta[:, :3]) > 1.0))

        if pos_err > 1e-10:
            results["demo_errors"].append(
                f"{demo_name}: pos_err={pos_err:.2e}, rot_err={rot_err:.2e}"
            )

    f.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    results = validate_delta_eef_dataset(args.dataset)

    print(f"\n{'='*60}")
    print(f"Validation Report")
    print(f"{'='*60}")
    print(f"Demos:           {results['n_demos']}")
    print(f"Total steps:     {results['total_steps']}")
    print(f"Action scale:    {results['action_scale']}")
    print(f"Shape OK:        {results['shape_ok']}")
    print(f"Rot+Grip OK:     {results['rot_grip_ok']}")
    print(f"Position max err:{results['pos_max_error']:.2e} m")
    print(f"Pos violations:  {results['violations_pos']}/{results['total_steps']}")

    pos_errs = results["pos_error_per_demo"]
    print(f"\nPer-demo position error: mean={np.mean(pos_errs):.2e}  "
          f"max={np.max(pos_errs):.2e}  demos with err>1e-10: {sum(1 for e in pos_errs if e > 1e-10)}/{len(pos_errs)}")

    if results["demo_errors"]:
        print(f"\nDemos with issues ({len(results['demo_errors'])}):")
        for err in results["demo_errors"]:
            print(f"  {err}")

    all_ok = (
        results["shape_ok"]
        and results["rot_grip_ok"]
        and results["pos_max_error"] < 1e-8
        and results["violations_pos"] == 0
    )
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
