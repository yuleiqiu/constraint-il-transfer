"""Verify Phase-1 point-cloud data and offline/runtime observation parity."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from robosuite.utils import camera_utils as CameraUtils
from scipy.spatial import cKDTree

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.target_pointcloud_utils import (
    get_target_and_goal_geom_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT
    / "third_party/robomimic/datasets/can/yq/"
    "image_v15_delta_eef_pose_osc_pointcloud.hdf5"
)
EXPECTED_LOW_DIM = {
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
}


def sorted_demo_keys(data_group):
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def verify_hdf5(dataset):
    with h5py.File(dataset, "r") as data:
        assert bool(data.attrs["pointcloud_complete"])
        config = json.loads(data.attrs["pointcloud_config_json"])
        pointcloud_config = config["pointcloud"]
        demos = sorted_demo_keys(data["data"])
        assert demos == config["demos"]
        total_states = sum(
            int(data["data/{}".format(name)].attrs["num_samples"]) for name in demos
        )
        assert int(data["data"].attrs["total"]) == total_states

        mask_sets = {}
        for mask_name, mask in data["mask"].items():
            values = {item.decode("utf-8") for item in mask[:]}
            assert values <= set(demos)
            mask_sets[mask_name] = values
        assert "train" in mask_sets and "valid" in mask_sets
        assert mask_sets["train"].isdisjoint(mask_sets["valid"])
        assert mask_sets["train"] | mask_sets["valid"] == set(demos)

        for demo_name in demos:
            demo = data["data/{}".format(demo_name)]
            assert bool(demo.attrs["pointcloud_complete"])
            length = int(demo.attrs["num_samples"])
            assert set(demo["obs"]) == EXPECTED_LOW_DIM | {"task_pointcloud"}
            assert demo["states"].shape[0] == length
            assert demo["delta_eef_pose_action"].shape == (length, 7)
            assert demo["rewards"].shape[0] == length
            assert demo["dones"].shape[0] == length
            pointcloud = demo["obs/task_pointcloud"]
            assert pointcloud.shape == (length, pointcloud_config["num_points"], 3)
            assert pointcloud.dtype == np.float32
            for start in range(0, length, 128):
                assert np.isfinite(pointcloud[start : start + 128]).all()
        return config, demos, total_states


def verify_runtime_parity(dataset, demo_name, timestep):
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {
            "obs": {
                "low_dim": list(EXPECTED_LOW_DIM),
                "rgb": [],
                "depth": [],
                "scan": ["task_pointcloud"],
            }
        }
    )
    with h5py.File(dataset, "r") as data:
        env_meta = json.loads(data["data"].attrs["env_args"])
        demo = data["data/{}".format(demo_name)]
        state = demo["states"][timestep]
        offline = demo["obs/task_pointcloud"][timestep]

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
    )
    try:
        assert env.env.has_offscreen_renderer
        runtime_obs = env.reset_to({"states": state})
        runtime = runtime_obs["task_pointcloud"]
        # Separate EGL environment instances can differ by a few float ULPs.
        # Those perturb FPS tie ordering, so compare the two unordered clouds.
        offline_to_runtime = cKDTree(runtime).query(offline, k=1)[0]
        runtime_to_offline = cKDTree(offline).query(runtime, k=1)[0]
        assert max(offline_to_runtime.max(), runtime_to_offline.max()) < 5e-5
        assert max(offline_to_runtime.mean(), runtime_to_offline.mean()) < 1e-5

        target_ids = get_target_and_goal_geom_ids(env.env, "Can", True)
        names = {env.env.sim.model.geom_id2name(geom_id) for geom_id in target_ids}
        assert any(name.startswith("Can_") for name in names)
        assert any(name.startswith("VisualCan_") for name in names)
        assert all(
            name.startswith("Can_") or name.startswith("VisualCan_") for name in names
        )

        config = env.target_pointcloud_config
        centers = np.asarray([env.env.sim.data.geom_xpos[geom_id] for geom_id in target_ids])
        nearest = np.linalg.norm(runtime[:, None, :] - centers[None, :, :], axis=-1).min(axis=1)
        assert nearest.max() <= config["max_geom_distance"] + 1e-5

        transform = CameraUtils.get_camera_transform_matrix(
            env.env.sim,
            camera_name=config["camera_name"],
            camera_height=config["height"],
            camera_width=config["width"],
        )
        projected = CameraUtils.project_points_from_world_to_camera(
            runtime,
            transform,
            camera_height=config["height"],
            camera_width=config["width"],
        )
        assert np.all(projected[:, 0] >= 0) and np.all(projected[:, 0] < config["height"])
        assert np.all(projected[:, 1] >= 0) and np.all(projected[:, 1] < config["width"])
    finally:
        env.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--demo")
    parser.add_argument("--timestep", type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    _, demos, total_states = verify_hdf5(dataset)
    demo = args.demo or demos[0]
    verify_runtime_parity(dataset, demo, args.timestep)
    print("Phase-1 dataset verification passed")
    print("dataset={}".format(dataset))
    print("demos={} total_states={}".format(len(demos), total_states))
    print("offline/runtime parity: {} timestep {}".format(demo, args.timestep))


if __name__ == "__main__":
    main()
