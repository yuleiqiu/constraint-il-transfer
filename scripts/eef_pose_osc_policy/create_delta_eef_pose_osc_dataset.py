"""
Create a delta-EEF full-pose OSC dataset for EEF-pose OSC policy training.

The action key is:
    delta_eef_pose_action[t] = [
        next_obs/robot0_eef_pos[t] - obs/robot0_eef_pos[t],
        axis_angle(R_next_site @ R_obs_site.T),
        actions[t, 6],
    ]

Quaternion inputs use robosuite's xyzw order directly. The copied dataset's
env_args metadata is updated to use built-in OSC_POSE in delta/world mode with
kp=500, identity scaling for the 6 EEF action dimensions, and desired-goal
delta accumulation. The desired-goal mode is required: achieved-goal delta
updates replay the local EEF displacement only and accumulate lag.

Run from repo root:
    uv run python scripts/eef_pose_osc_policy/create_delta_eef_pose_osc_dataset.py

For smoke tests without copying the full image dataset:
    uv run python scripts/eef_pose_osc_policy/create_delta_eef_pose_osc_dataset.py \
        --dest /tmp/image_v15_delta_eef_pose_osc_smoke.hdf5 --demo-limit 4 --overwrite
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

import robosuite.utils.transform_utils as T


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15.hdf5"
DEFAULT_DEST = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15_delta_eef_pose_osc.hdf5"
DEFAULT_ACTION_KEY = "delta_eef_pose_action"
DEFAULT_ORIENTATION_SOURCE = "robot0_eef_quat_site"


def sorted_demo_keys(data_group):
    return sorted(data_group.keys(), key=lambda x: int(x.split("_")[-1]))


def selected_demo_keys(src_file, demo_limit):
    demo_names = sorted_demo_keys(src_file["data"])
    if demo_limit is None:
        return demo_names
    if demo_limit <= 0:
        raise ValueError("--demo-limit must be positive")
    return demo_names[:demo_limit]


def delta_axis_angle(q0_xyzw, q1_xyzw):
    """Return rotvec dR such that dR @ R0 = R1."""
    r0 = T.quat2mat(q0_xyzw.copy())
    r1 = T.quat2mat(q1_xyzw.copy())
    return Rotation.from_matrix(r1 @ r0.T).as_rotvec()


def build_delta_eef_pose_action(group, orientation_source):
    n_steps = group["actions"].shape[0]
    action = np.zeros((n_steps, 7), dtype=np.float64)
    action[:, :3] = group["next_obs/robot0_eef_pos"][:] - group["obs/robot0_eef_pos"][:]
    action[:, 3:6] = np.asarray(
        [
            delta_axis_angle(q0, q1)
            for q0, q1 in zip(
                group[f"obs/{orientation_source}"][:],
                group[f"next_obs/{orientation_source}"][:],
            )
        ],
        dtype=np.float64,
    )
    action[:, 6] = group["actions"][:, 6]
    return action


def set_delta_osc_controller_metadata(data_group, kp):
    env_args = json.loads(data_group.attrs["env_args"])
    env_args["env_kwargs"]["controller_goal_update_mode"] = "desired"
    controller_configs = env_args["env_kwargs"]["controller_configs"]
    for part_name, body_part in controller_configs["body_parts"].items():
        if body_part.get("type") != "OSC_POSE":
            raise ValueError(f"Expected OSC_POSE controller for body part {part_name}, got {body_part.get('type')}")
        body_part["kp"] = kp
        body_part["input_type"] = "delta"
        body_part["input_ref_frame"] = "world"
        body_part["input_min"] = [-1.0] * 6
        body_part["input_max"] = [1.0] * 6
        body_part["output_min"] = [-1.0] * 6
        body_part["output_max"] = [1.0] * 6
        body_part["eef_pose_action_note"] = (
            "Delta full-pose EEF action: world dpos + site dR axis-angle + gripper; "
            "requires controller_goal_update_mode=desired"
        )
    data_group.attrs["env_args"] = json.dumps(env_args, indent=4)
    return env_args


def copy_subset(source, dest, demo_names):
    with h5py.File(source, "r") as src, h5py.File(dest, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        dst_data = dst.create_group("data")
        for key, value in src["data"].attrs.items():
            dst_data.attrs[key] = value

        total = 0
        for demo_name in demo_names:
            src.copy(src[f"data/{demo_name}"], dst_data, name=demo_name)
            total += int(src[f"data/{demo_name}"].attrs["num_samples"])
        dst_data.attrs["total"] = total

        if "mask" in src:
            dst_mask = dst.create_group("mask")
            selected = set(demo_names)
            for mask_name in src["mask"]:
                values = [
                    item.decode("utf-8")
                    for item in np.asarray(src[f"mask/{mask_name}"][:])
                    if item.decode("utf-8") in selected
                ]
                dst_mask.create_dataset(mask_name, data=np.asarray(values, dtype="S"))


def copy_source_dataset(source, dest, demo_names, demo_limit):
    if demo_limit is None:
        shutil.copy2(source, dest)
    else:
        copy_subset(source, dest, demo_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--action-key", type=str, default=DEFAULT_ACTION_KEY)
    parser.add_argument("--orientation-source", type=str, default=DEFAULT_ORIENTATION_SOURCE)
    parser.add_argument("--osc-kp", type=float, default=500.0)
    parser.add_argument("--demo-limit", type=int, default=None, help="Create a small subset dataset for smoke tests")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if dest.exists() and not args.overwrite:
        raise FileExistsError(f"{dest} already exists; pass --overwrite to replace it")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()

    with h5py.File(source, "r") as src:
        demo_names = selected_demo_keys(src, args.demo_limit)
        first_demo = src[f"data/{demo_names[0]}"]
        required = [
            "obs/robot0_eef_pos",
            "next_obs/robot0_eef_pos",
            f"obs/{args.orientation_source}",
            f"next_obs/{args.orientation_source}",
            "actions",
        ]
        for key in required:
            if key not in first_demo:
                raise KeyError(f"Missing required dataset key {key}")

    copy_source_dataset(source, dest, demo_names=demo_names, demo_limit=args.demo_limit)

    with h5py.File(dest, "r+") as f:
        env_args = set_delta_osc_controller_metadata(f["data"], kp=args.osc_kp)
        demo_names = sorted_demo_keys(f["data"])
        action_min = None
        action_max = None
        for demo_name in demo_names:
            group = f[f"data/{demo_name}"]
            if args.action_key in group:
                del group[args.action_key]
            action = build_delta_eef_pose_action(group, orientation_source=args.orientation_source)
            group.create_dataset(args.action_key, data=action, compression="gzip")
            action_min = action.min(axis=0) if action_min is None else np.minimum(action_min, action.min(axis=0))
            action_max = action.max(axis=0) if action_max is None else np.maximum(action_max, action.max(axis=0))

        f["data"].attrs[f"{args.action_key}_source"] = (
            f"next_obs/robot0_eef_pos - obs/robot0_eef_pos + "
            f"axis_angle(next_obs/{args.orientation_source} @ obs/{args.orientation_source}^-1) + actions[:, 6]"
        )
        f["data"].attrs["delta_eef_pose_action_key"] = args.action_key
        f["data"].attrs["delta_eef_pose_orientation_source"] = args.orientation_source
        f["data"].attrs["delta_eef_pose_quaternion_order"] = "robosuite xyzw, no reordering"
        f["data"].attrs["delta_eef_pose_controller"] = "OSC_POSE delta world, identity EEF scaling, desired goal update"
        f["data"].attrs["delta_eef_pose_osc_kp"] = args.osc_kp
        f["data"].attrs["delta_eef_pose_action_min"] = action_min
        f["data"].attrs["delta_eef_pose_action_max"] = action_max

    body_parts = env_args["env_kwargs"]["controller_configs"]["body_parts"]
    print(f"Wrote {dest}")
    print(f"Demos: {len(demo_names)}")
    print(f"Action key: {args.action_key} shape=(T, 7)")
    print(f"Orientation source: obs/next_obs/{args.orientation_source} (xyzw, no reorder)")
    print(f"Controller goal update mode: {env_args['env_kwargs']['controller_goal_update_mode']}")
    for part_name, body_part in body_parts.items():
        print(
            f"  {part_name}: type={body_part['type']} input_type={body_part['input_type']} "
            f"input_ref_frame={body_part['input_ref_frame']} kp={body_part['kp']} "
            "scale=identity"
        )
    print(f"Action min: {np.array2string(action_min, precision=4)}")
    print(f"Action max: {np.array2string(action_max, precision=4)}")


if __name__ == "__main__":
    main()
