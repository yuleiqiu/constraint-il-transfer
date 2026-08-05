"""Verify the offline LAN-O3DP contract and its online observation parity."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT
    / "third_party/robomimic/datasets/can/yq/"
    "image_v15_abs_eef_pose_osc_lan40_paper256.hdf5"
)


def decode_mask(mask):
    return {item.decode("utf-8") for item in np.asarray(mask[:])}


def nearest_neighbor_error(first, second):
    first_to_second = cKDTree(second).query(first, k=1)[0]
    second_to_first = cKDTree(first).query(second, k=1)[0]
    return (
        max(first_to_second.max(), second_to_first.max()),
        max(first_to_second.mean(), second_to_first.mean()),
    )


def verify_structure(path):
    with h5py.File(path, "r") as dataset:
        if not bool(dataset.attrs.get("lan_o3dp_complete", False)):
            raise ValueError("Dataset is incomplete; resume generation first")
        config = json.loads(dataset.attrs["lan_o3dp_config_json"])
        data = dataset["data"]
        demos = set(data.keys())
        selected = decode_mask(dataset["mask/selected"])
        train = decode_mask(dataset["mask/train"])
        valid = decode_mask(dataset["mask/valid"])
        assert demos == selected
        assert train.isdisjoint(valid)
        assert train | valid == selected
        assert (len(selected), len(train), len(valid)) == (40, 38, 2)
        assert valid == {"demo_18", "demo_164"}

        expected_points = int(config["pointcloud"]["num_points"])
        total = 0
        for demo_name in sorted(demos):
            demo = data[demo_name]
            length = int(demo.attrs["num_samples"])
            total += length
            assert bool(demo.attrs["lan_o3dp_complete"])
            assert demo["states"].shape[0] == length
            assert demo["abs_eef_pose_action"].shape == (length, 7)
            assert demo["obs/agent_pos"].shape == (length, 6)
            assert demo["obs/task_pointcloud"].shape == (
                length,
                expected_points,
                3,
            )
            assert np.isfinite(demo["obs/agent_pos"][:]).all()
            for start in range(0, length, 128):
                cloud = demo["obs/task_pointcloud"][start : start + 128]
                assert np.isfinite(cloud).all()
        assert total == int(data.attrs["total"])
        return config, sorted(demos), total


def verify_runtime_parity(path, demo_name, timestep):
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {
            "obs": {
                "low_dim": ["agent_pos"],
                "rgb": [],
                "depth": [],
                "scan": ["task_pointcloud"],
            }
        }
    )
    with h5py.File(path, "r") as dataset:
        env_meta = json.loads(dataset["data"].attrs["env_args"])
        demo = dataset["data/{}".format(demo_name)]
        state = demo["states"][timestep]
        offline_agent = demo["obs/agent_pos"][timestep]
        offline_cloud = demo["obs/task_pointcloud"][timestep]

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
    )
    try:
        runtime = env.reset_to({"states": state})
        np.testing.assert_allclose(runtime["agent_pos"], offline_agent, atol=1e-6)
        maximum, mean = nearest_neighbor_error(
            runtime["task_pointcloud"],
            offline_cloud,
        )
        if maximum >= 5e-5 or mean >= 1e-5:
            raise AssertionError(
                "Runtime point cloud differs: max={:.3g}, mean={:.3g}".format(
                    maximum, mean
                )
            )
        controller = env.env.robots[0].part_controllers["right"]
        assert controller.input_type == "absolute"
        assert controller.input_ref_frame == "world"
        assert env.action_dimension == 7
        return maximum, mean
    finally:
        env.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--demo")
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Only run online parity on an intentionally partial smoke dataset",
    )
    args = parser.parse_args()
    path = args.dataset.resolve()
    if args.allow_incomplete:
        with h5py.File(path, "r") as dataset:
            demos = sorted(dataset["data"].keys())
            config = json.loads(dataset.attrs["lan_o3dp_config_json"])
            total = sum(int(dataset["data/{}".format(k)].attrs["num_samples"]) for k in demos)
    else:
        config, demos, total = verify_structure(path)
    if not demos:
        raise ValueError("Dataset has no completed demos")
    demo_name = args.demo or demos[0]
    maximum, mean = verify_runtime_parity(path, demo_name, args.timestep)
    print("Verified {}".format(path))
    print("Profile: {}; demos: {}; samples: {}".format(config["profile"], len(demos), total))
    print("Runtime parity {}[{}]: max={:.3g}, mean={:.3g}".format(demo_name, args.timestep, maximum, mean))


if __name__ == "__main__":
    main()
