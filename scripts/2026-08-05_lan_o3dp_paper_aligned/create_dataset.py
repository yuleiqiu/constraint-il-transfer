"""Build a paper-aligned LAN-O3DP dataset from executable absolute EEF data.

The output contains a fixed 40-demonstration subset, a 38/2 train/validation
split, a 6-D absolute EEF pose observation, a 7-D absolute EEF pose action,
and a target-object point cloud rendered from one calibrated front camera
with oracle simulator masks. The opt-in corrected profile also removes the
absolute rotation-vector branch discontinuity and raises raster density before
sampling 256 target points.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from robosuite.utils import transform_utils as T
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.target_pointcloud_utils import (
    DEFAULT_POINTCLOUD_CONFIG,
    get_target_and_goal_geom_ids,
    normalize_target_pointcloud_config,
    render_target_pointcloud,
)
from robomimic.utils.continuous_rotation_utils import (
    continuous_quaternion_sequence_to_rotation_vectors,
    normalize_quaternion_xyzw,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "third_party/robomimic/datasets/can/yq/"
    "image_v15_abs_eef_pose_osc.hdf5"
)
PROFILE_CONFIGS = {
    "paper256": {"num_points": 256, "padding_mode": "repeat"},
    "official512": {"num_points": 512, "padding_mode": "zero"},
    "paper256_hires_continuous": {
        "height": 256,
        "width": 256,
        "num_points": 256,
        "padding_mode": "repeat",
    },
}
SEED = 42
SUBSET_SIZE = 40
VALID_SIZE = 2
FORMAT_VERSION = 1
ACTION_KEY = "abs_eef_pose_action"
OBS_KEY = "agent_pos"
POINTCLOUD_KEY = "task_pointcloud"


def default_dest(profile):
    return (
        REPO_ROOT
        / "third_party/robomimic/datasets/can/yq/"
        / "image_v15_abs_eef_pose_osc_lan40_{}.hdf5".format(profile)
    )


def numeric_demo_sort(names):
    return sorted((str(name) for name in names), key=lambda name: int(name.split("_")[-1]))


def select_demo_split(names, seed=SEED):
    ordered = numeric_demo_sort(names)
    if len(ordered) < SUBSET_SIZE:
        raise ValueError("Need at least {} demonstrations".format(SUBSET_SIZE))
    selected = numeric_demo_sort(
        np.random.default_rng(seed).choice(
            ordered,
            SUBSET_SIZE,
            replace=False,
        )
    )
    valid = numeric_demo_sort(
        np.random.default_rng(seed).choice(
            selected,
            VALID_SIZE,
            replace=False,
        )
    )
    valid_set = set(valid)
    train = [name for name in selected if name not in valid_set]
    return selected, train, valid


def rotation_vector_config(profile, reference_quaternion=None):
    if profile != "paper256_hires_continuous":
        return {"mode": "principal"}
    if reference_quaternion is None:
        raise ValueError(
            "paper256_hires_continuous requires a reference quaternion"
        )
    return {
        "mode": "continuous",
        "reference_quaternion_xyzw": normalize_quaternion_xyzw(
            reference_quaternion
        ).tolist(),
    }


def quaternion_sequence_to_rotation_vectors(quaternions, rotation_config):
    if rotation_config["mode"] == "continuous":
        return continuous_quaternion_sequence_to_rotation_vectors(
            quaternions,
            rotation_config["reference_quaternion_xyzw"],
        )
    return np.asarray(
        [T.quat2axisangle(quaternion.copy()) for quaternion in quaternions],
        dtype=np.float64,
    )


def build_agent_pos(obs_group, rotation_config=None):
    positions = np.asarray(obs_group["robot0_eef_pos"], dtype=np.float32)
    quaternions = np.asarray(
        obs_group["robot0_eef_quat_site"],
        dtype=np.float64,
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("robot0_eef_pos must have shape [T, 3]")
    if quaternions.shape != (positions.shape[0], 4):
        raise ValueError("robot0_eef_quat_site must have shape [T, 4]")
    rotation_config = rotation_config or {"mode": "principal"}
    rotations = quaternion_sequence_to_rotation_vectors(
        quaternions,
        rotation_config,
    ).astype(np.float32)
    return np.concatenate((positions, rotations), axis=1)


def build_absolute_action(source_demo, rotation_config):
    positions = np.asarray(
        source_demo["next_obs/robot0_eef_pos"],
        dtype=np.float64,
    )
    quaternions = np.asarray(
        source_demo["next_obs/robot0_eef_quat_site"],
        dtype=np.float64,
    )
    gripper = np.asarray(source_demo["actions"][:, 6], dtype=np.float64)
    rotations = quaternion_sequence_to_rotation_vectors(
        quaternions,
        rotation_config,
    )
    return np.concatenate(
        (positions, rotations, gripper[:, None]),
        axis=1,
    )


def source_fingerprint(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_config(
    source,
    profile,
    selected,
    train,
    valid,
    reference_quaternion=None,
):
    pointcloud = dict(DEFAULT_POINTCLOUD_CONFIG)
    pointcloud.update(
        {
            "obs_key": POINTCLOUD_KEY,
            "camera_name": "agentview",
            "height": 120,
            "width": 120,
            "target_object": "Can",
            "include_visual_goal": True,
            **PROFILE_CONFIGS[profile],
        }
    )
    rotation_config = rotation_vector_config(
        profile,
        reference_quaternion=reference_quaternion,
    )
    config = {
        "format_version": (
            2 if rotation_config["mode"] == "continuous" else FORMAT_VERSION
        ),
        "source": source_fingerprint(source),
        "profile": profile,
        "selected": selected,
        "train": train,
        "valid": valid,
        "action_key": ACTION_KEY,
        "agent_pos": (
            "obs/robot0_eef_pos + "
            "axis_angle(obs/robot0_eef_quat_site_xyzw)"
        ),
        "pointcloud": normalize_target_pointcloud_config(pointcloud),
    }
    if rotation_config["mode"] == "continuous":
        config["rotation_vector"] = rotation_config
        config["agent_pos"] = (
            "obs/robot0_eef_pos + continuous_axis_angle("
            "obs/robot0_eef_quat_site_xyzw)"
        )
    return config


def config_digest(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def runtime_env_args(source_env_args, pointcloud, rotation_config=None):
    env_args = json.loads(source_env_args)
    env_kwargs = env_args["env_kwargs"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    env_kwargs["camera_depths"] = False
    env_kwargs["target_pointcloud"] = pointcloud
    env_kwargs["eef_pose_observation"] = {
        "enabled": True,
        "obs_key": OBS_KEY,
        "robot_prefix": "robot0",
    }
    if rotation_config is not None and rotation_config["mode"] == "continuous":
        env_kwargs["eef_pose_observation"].update(
            {
                "rotation_vector_mode": "continuous",
                "reference_quaternion_xyzw": rotation_config[
                    "reference_quaternion_xyzw"
                ],
            }
        )
    env_args["lan_o3dp_observation"] = {
        "camera": "single calibrated agentview",
        "point_frame": "world",
        "segmentation": "oracle visible geom mask",
        "agent_pos": (
            "world xyz + continuous site axis-angle"
            if rotation_config is not None
            and rotation_config["mode"] == "continuous"
            else "world xyz + site axis-angle"
        ),
    }
    return json.dumps(env_args, indent=4)


def initialize_output(output, source_file, config, digest):
    for key, value in source_file.attrs.items():
        output.attrs[key] = value
    output.attrs["lan_o3dp_paper_aligned_format_version"] = config[
        "format_version"
    ]
    output.attrs["lan_o3dp_config_json"] = json.dumps(config, sort_keys=True)
    output.attrs["lan_o3dp_config_sha256"] = digest
    output.attrs["lan_o3dp_complete"] = False

    data = output.create_group("data")
    for key, value in source_file["data"].attrs.items():
        data.attrs[key] = value
    data.attrs["total"] = sum(
        int(source_file["data/{}".format(name)].attrs["num_samples"])
        for name in config["selected"]
    )
    data.attrs["env_args"] = runtime_env_args(
        source_file["data"].attrs["env_args"],
        config["pointcloud"],
        config.get("rotation_vector"),
    )
    data.attrs["lan_o3dp_action_semantics"] = (
        "absolute world EEF next pose: xyz + {}site axis-angle + gripper".format(
            "continuous " if "rotation_vector" in config else ""
        )
    )

    mask = output.create_group("mask")
    for key in ("selected", "train", "valid"):
        mask.create_dataset(key, data=np.asarray(config[key], dtype="S"))
    output.flush()


def validate_resume(output, digest):
    if output.attrs.get("lan_o3dp_config_sha256") != digest:
        raise ValueError(
            "Existing output has a different configuration; use --overwrite"
        )


def copy_dataset(source_group, destination_group, key):
    dataset = source_group[key]
    destination_group.create_dataset(
        key,
        data=dataset[:],
        dtype=dataset.dtype,
        compression="gzip",
    )


def process_demo(env, source_demo, output_data, demo_name, config):
    if demo_name in output_data:
        existing = output_data[demo_name]
        if bool(existing.attrs.get("lan_o3dp_complete", False)):
            return False
        del output_data[demo_name]

    demo = output_data.create_group(demo_name)
    for key, value in source_demo.attrs.items():
        demo.attrs[key] = value
    demo.attrs["lan_o3dp_complete"] = False
    for key in ("states", "rewards", "dones"):
        copy_dataset(source_demo, demo, key)
    rotation_config = config.get("rotation_vector", {"mode": "principal"})
    if rotation_config["mode"] == "continuous":
        action = build_absolute_action(source_demo, rotation_config)
        demo.create_dataset(ACTION_KEY, data=action, compression="gzip")
    else:
        copy_dataset(source_demo, demo, ACTION_KEY)

    obs = demo.create_group("obs")
    agent_pos = build_agent_pos(source_demo["obs"], rotation_config)
    obs.create_dataset(OBS_KEY, data=agent_pos, compression="gzip")

    states = source_demo["states"]
    pointcloud = config["pointcloud"]
    clouds = obs.create_dataset(
        POINTCLOUD_KEY,
        shape=(len(states), pointcloud["num_points"], 3),
        dtype=np.float32,
        chunks=(1, pointcloud["num_points"], 3),
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
    target_ids = get_target_and_goal_geom_ids(
        raw_env,
        target_object=pointcloud["target_object"],
        include_visual_goal=pointcloud["include_visual_goal"],
    )
    for timestep in tqdm(range(len(states)), desc=demo_name, leave=False):
        raw_env.sim.set_state_from_flattened(states[timestep])
        raw_env.sim.forward()
        clouds[timestep] = render_target_pointcloud(
            raw_env,
            config=pointcloud,
            target_geom_ids=target_ids,
        )
    demo.attrs["lan_o3dp_complete"] = True
    output_data.file.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_CONFIGS),
        default="paper256",
    )
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stop-after-demos",
        type=int,
        help="Process only this many selected demos and leave a resumable incomplete output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.resolve()
    dest = (args.dest or default_dest(args.profile)).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source == dest:
        raise ValueError("Source and destination must differ")
    if args.stop_after_demos is not None and args.stop_after_demos <= 0:
        raise ValueError("--stop-after-demos must be positive")
    if args.overwrite and dest.exists():
        dest.unlink()

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": [], "rgb": [], "depth": [], "scan": []}}
    )
    with h5py.File(source, "r") as source_file:
        selected, train, valid = select_demo_split(source_file["data"].keys())
        reference_quaternion = None
        if args.profile == "paper256_hires_continuous":
            reference_quaternion = source_file[
                "data/{}/obs/robot0_eef_quat_site".format(selected[0])
            ][0]
        config = build_config(
            source,
            args.profile,
            selected,
            train,
            valid,
            reference_quaternion=reference_quaternion,
        )
        digest = config_digest(config)
        dest.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if dest.exists() else "w"
        with h5py.File(dest, mode) as output:
            if mode == "w":
                initialize_output(output, source_file, config, digest)
            else:
                validate_resume(output, digest)

            source_env_args = json.loads(source_file["data"].attrs["env_args"])
            env = EnvUtils.create_env_from_metadata(
                env_meta=source_env_args,
                render=False,
                render_offscreen=True,
                use_image_obs=False,
                use_depth_obs=False,
            )
            generated = 0
            try:
                for demo_name in tqdm(selected, desc="demos"):
                    generated += int(
                        process_demo(
                            env,
                            source_file["data/{}".format(demo_name)],
                            output["data"],
                            demo_name,
                            config,
                        )
                    )
                    if (
                        args.stop_after_demos is not None
                        and generated >= args.stop_after_demos
                    ):
                        break
                complete = all(
                    name in output["data"]
                    and bool(
                        output["data/{}".format(name)].attrs.get(
                            "lan_o3dp_complete", False
                        )
                    )
                    for name in selected
                )
                output.attrs["lan_o3dp_complete"] = complete
                output.flush()
            finally:
                env.env.close()

    print("Wrote {}".format(dest))
    print("Profile: {}; config SHA256: {}".format(args.profile, digest))
    print("Demos generated this run: {}; train/valid: 38/2".format(generated))
    print("Complete: {}".format(complete))


if __name__ == "__main__":
    main()
