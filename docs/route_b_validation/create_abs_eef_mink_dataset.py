"""
Create an absolute-EEF action dataset for Panda + WholeBodyMinkIK training.

The default full-pose action key is:
    abs_eef_pose_action[t] = [
        next_obs/robot0_eef_pos[t],
        quat2axisangle(next_obs/robot0_eef_quat_site[t]),
        actions[t, 6],
    ]

The copied dataset's env_args metadata is also updated to use the expanded
Panda Mink IK composite controller, so training rollouts and saved checkpoints
load the same controller interface as the action labels.

Run from repo root:
    uv run python docs/route_b_validation/create_abs_eef_mink_dataset.py
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

import robosuite
from robosuite.utils import transform_utils as T


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5"
DEFAULT_DEST = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15_abs_eef_pose_mink.hdf5"
DEFAULT_CONTROLLER_CONFIG = (
    REPO_ROOT / "third_party/robosuite/robosuite/controllers/config/default/composite/panda_mink_ik_full_pose.json"
)
DEFAULT_ACTION_KEY = "abs_eef_pose_action"
DEFAULT_ORIENTATION_SOURCE = "robot0_eef_quat_site"


def build_abs_eef_action(group, orientation_source):
    n_steps = group["actions"].shape[0]
    action = np.zeros((n_steps, 7), dtype=np.float64)
    action[:, :3] = group["next_obs/robot0_eef_pos"][:]
    if orientation_source != "none":
        action[:, 3:6] = np.asarray(
            [T.quat2axisangle(quat.copy()) for quat in group[f"next_obs/{orientation_source}"][:]],
            dtype=np.float64,
        )
    action[:, 6] = group["actions"][:, 6]
    return action


def update_env_args(data_group, controller_config_path):
    env_args = json.loads(data_group.attrs["env_args"])
    controller_config = robosuite.load_composite_controller_config(controller=str(controller_config_path))
    env_args["env_kwargs"]["controller_configs"] = controller_config
    data_group.attrs["env_args"] = json.dumps(env_args, indent=4)
    return env_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--controller-config", type=Path, default=DEFAULT_CONTROLLER_CONFIG)
    parser.add_argument("--action-key", type=str, default=DEFAULT_ACTION_KEY)
    parser.add_argument(
        "--orientation-source",
        choices=["robot0_eef_quat_site", "robot0_eef_quat", "none"],
        default=DEFAULT_ORIENTATION_SOURCE,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()
    controller_config_path = args.controller_config.resolve()

    if not source.exists():
        raise FileNotFoundError(source)
    if dest.exists() and not args.overwrite:
        raise FileExistsError(f"{dest} already exists; pass --overwrite to replace it")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.copy2(source, dest)

    with h5py.File(dest, "r+") as f:
        env_args = update_env_args(f["data"], controller_config_path)
        demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        for demo_name in demo_names:
            group = f[f"data/{demo_name}"]
            if args.action_key in group:
                del group[args.action_key]
            group.create_dataset(
                args.action_key,
                data=build_abs_eef_action(group, orientation_source=args.orientation_source),
                compression="gzip",
            )

        f["data"].attrs[f"{args.action_key}_source"] = (
            f"next_obs/robot0_eef_pos + next_obs/{args.orientation_source}->axis_angle + actions[:, 6]"
        )
        f["data"].attrs["abs_eef_controller_config"] = str(controller_config_path)
        f["data"].attrs["abs_eef_action_key"] = args.action_key
        f["data"].attrs["abs_eef_orientation_source"] = args.orientation_source

    print(f"Wrote {dest}")
    print(f"Action key: {args.action_key}")
    print(f"Orientation source: {args.orientation_source}")
    print(f"Controller type: {env_args['env_kwargs']['controller_configs']['type']}")
    print(
        "Orientation cost: "
        f"{env_args['env_kwargs']['controller_configs']['composite_controller_specific_configs']['ik_hand_ori_cost']}"
    )
    print(f"Body parts: {list(env_args['env_kwargs']['controller_configs']['body_parts'].keys())}")
    print(f"Demos: {len(demo_names)}")


if __name__ == "__main__":
    main()
