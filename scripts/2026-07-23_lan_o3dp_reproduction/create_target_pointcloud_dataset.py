"""
Create the slim point-cloud delta-EEF training dataset for LAN-O3DP Phase 1.

Every saved simulator state is restored and rendered from ``agentview`` at
256x256 with metric depth and oracle geom segmentation. Physical ``Can`` and
``VisualCan`` goal geoms are merged, unprojected to world XYZ, and
deterministically sampled to 512 points.

The output is resumable at demo boundaries. A partial demo is deleted and
regenerated on resume. Existing files require an identical configuration hash;
``--overwrite`` explicitly starts over.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.target_pointcloud_utils import (
    DEFAULT_POINTCLOUD_CONFIG,
    get_target_and_goal_geom_ids,
    normalize_target_pointcloud_config,
    render_target_pointcloud,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "third_party/robomimic/datasets/can/yq/image_v15_delta_eef_pose_osc.hdf5"
)
DEFAULT_DEST = (
    REPO_ROOT
    / "third_party/robomimic/datasets/can/yq/"
    "image_v15_delta_eef_pose_osc_pointcloud.hdf5"
)
LOW_DIM_OBS_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
DATASET_KEYS = ("delta_eef_pose_action", "rewards", "dones")
FORMAT_VERSION = 1


def sorted_demo_keys(data_group):
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def source_fingerprint(source):
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_config(args, source, demo_names):
    pointcloud = dict(DEFAULT_POINTCLOUD_CONFIG)
    pointcloud.update(
        {
            "camera_name": args.camera,
            "height": args.height,
            "width": args.width,
            "target_object": args.target_object,
            "include_visual_goal": True,
            "num_points": args.num_points,
            "max_geom_distance": args.max_geom_distance,
        }
    )
    pointcloud = normalize_target_pointcloud_config(pointcloud)
    return {
        "format_version": FORMAT_VERSION,
        "source": source_fingerprint(source),
        "demos": demo_names,
        "low_dim_obs_keys": list(LOW_DIM_OBS_KEYS),
        "dataset_keys": list(DATASET_KEYS),
        "pointcloud": pointcloud,
    }


def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def pointcloud_env_args(source_env_args, pointcloud_config):
    env_args = json.loads(source_env_args)
    env_kwargs = env_args["env_kwargs"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    env_kwargs["camera_depths"] = False
    env_kwargs["target_pointcloud"] = pointcloud_config
    env_args["pointcloud_observation"] = {
        "frame": "world",
        "source": "oracle geom segmentation + metric depth",
        "merged_geoms": [
            pointcloud_config["target_object"],
            "Visual{}".format(pointcloud_config["target_object"]),
        ],
    }
    return json.dumps(env_args, indent=4)


def initialize_output(output, source_file, demo_names, config, digest):
    output.attrs["pointcloud_dataset_format_version"] = FORMAT_VERSION
    output.attrs["pointcloud_config_json"] = json.dumps(config, sort_keys=True)
    output.attrs["pointcloud_config_sha256"] = digest
    output.attrs["pointcloud_complete"] = False
    data = output.create_group("data")
    for key, value in source_file["data"].attrs.items():
        data.attrs[key] = value
    data.attrs["env_args"] = pointcloud_env_args(
        source_file["data"].attrs["env_args"], config["pointcloud"]
    )
    data.attrs["total"] = sum(
        int(source_file["data/{}".format(name)].attrs["num_samples"])
        for name in demo_names
    )
    data.attrs["task_pointcloud_source"] = (
        "agentview metric depth + oracle Can/VisualCan geom segmentation; "
        "world XYZ; deterministic FPS with cyclic repeat padding"
    )

    mask = output.create_group("mask")
    selected = set(demo_names)
    if "mask" in source_file:
        for mask_name, source_mask in source_file["mask"].items():
            values = [
                item.decode("utf-8")
                for item in source_mask[:]
                if item.decode("utf-8") in selected
            ]
            mask.create_dataset(mask_name, data=np.asarray(values, dtype="S"))
    output.flush()


def validate_resume_file(output, digest):
    actual = output.attrs.get("pointcloud_config_sha256")
    if actual != digest:
        raise ValueError(
            "Existing output configuration hash {} does not match {}. "
            "Use --overwrite to regenerate.".format(actual, digest)
        )


def copy_dataset(source_group, dest_group, key, compression="gzip"):
    source = source_group[key]
    return dest_group.create_dataset(
        key,
        data=source[:],
        dtype=source.dtype,
        compression=compression,
    )


def process_demo(env, source_demo, output_data, demo_name, pointcloud_config):
    if demo_name in output_data:
        existing = output_data[demo_name]
        if bool(existing.attrs.get("pointcloud_complete", False)):
            return False
        del output_data[demo_name]

    demo = output_data.create_group(demo_name)
    for key, value in source_demo.attrs.items():
        demo.attrs[key] = value
    demo.attrs["pointcloud_complete"] = False
    for key in ("states",) + DATASET_KEYS:
        copy_dataset(source_demo, demo, key)
    obs = demo.create_group("obs")
    for key in LOW_DIM_OBS_KEYS:
        copy_dataset(source_demo["obs"], obs, key)

    states = source_demo["states"]
    num_samples = len(states)
    pointclouds = obs.create_dataset(
        pointcloud_config["obs_key"],
        shape=(num_samples, pointcloud_config["num_points"], 3),
        dtype=np.float32,
        chunks=(1, pointcloud_config["num_points"], 3),
        compression="gzip",
        compression_opts=4,
    )

    initial_state = {"states": states[0]}
    if "model_file" in source_demo.attrs:
        initial_state["model"] = source_demo.attrs["model_file"]
    if "ep_meta" in source_demo.attrs:
        initial_state["ep_meta"] = source_demo.attrs["ep_meta"]
    env.reset_to(initial_state)
    raw_env = env.env
    target_geom_ids = get_target_and_goal_geom_ids(
        raw_env=raw_env,
        target_object=pointcloud_config["target_object"],
        include_visual_goal=pointcloud_config["include_visual_goal"],
    )

    for timestep in tqdm(range(num_samples), desc=demo_name, leave=False):
        raw_env.sim.set_state_from_flattened(states[timestep])
        raw_env.sim.forward()
        pointclouds[timestep] = render_target_pointcloud(
            raw_env=raw_env,
            config=pointcloud_config,
            target_geom_ids=target_geom_ids,
        )
    demo.attrs["pointcloud_complete"] = True
    output_data.file.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--target-object", default="Can")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--max-geom-distance", type=float, default=0.15)
    parser.add_argument("--demo-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    if source == dest:
        raise ValueError("Source and destination must differ")
    if not source.exists():
        raise FileNotFoundError(source)
    if args.overwrite and dest.exists():
        dest.unlink()

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": [], "rgb": [], "depth": [], "scan": []}}
    )
    with h5py.File(source, "r") as source_file:
        demo_names = sorted_demo_keys(source_file["data"])
        if args.demo_limit is not None:
            if args.demo_limit <= 0:
                raise ValueError("--demo-limit must be positive")
            demo_names = demo_names[: args.demo_limit]
        config = build_config(args, source, demo_names)
        digest = config_hash(config)

        dest.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if dest.exists() else "w"
        with h5py.File(dest, mode) as output:
            if mode == "w":
                initialize_output(output, source_file, demo_names, config, digest)
            else:
                validate_resume_file(output, digest)

            # The processing environment does not enable the runtime provider;
            # the same extraction utility is called explicitly below.
            env_args = json.loads(source_file["data"].attrs["env_args"])
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_args,
                render=False,
                render_offscreen=True,
                use_image_obs=False,
                use_depth_obs=False,
            )
            generated = 0
            try:
                for demo_name in tqdm(demo_names, desc="demos"):
                    generated += int(
                        process_demo(
                            env=env,
                            source_demo=source_file["data/{}".format(demo_name)],
                            output_data=output["data"],
                            demo_name=demo_name,
                            pointcloud_config=config["pointcloud"],
                        )
                    )
                output.attrs["pointcloud_complete"] = True
                output.flush()
            finally:
                env.env.close()

    print("Wrote {}".format(dest))
    print("Config SHA256: {}".format(digest))
    print("Demos generated: {}; total present: {}".format(generated, len(demo_names)))


if __name__ == "__main__":
    main()
