"""Validate the Tabletop point-cloud / absolute-EEF demonstration dataset."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from scipy.spatial import cKDTree


OBS_KEYS = ("agent_pos", "task_pointcloud")
ACTION_KEY = "abs_eef_pose_action"


def sorted_demo_names(data):
    return sorted(data.keys(), key=lambda name: int(name.split("_")[-1]))


def decode_mask(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def validate_dataset(path):
    with h5py.File(path, "r") as input_file:
        data = input_file["data"]
        demos = sorted_demo_names(data)
        if len(demos) < 2:
            raise AssertionError("dataset needs at least two demos")
        total = 0
        for name in demos:
            demo = data[name]
            length = int(demo.attrs["num_samples"])
            total += length
            assert bool(demo.attrs["teacher_success"])
            assert demo["states"].shape[0] == length
            assert demo["actions"].shape == (length, 7)
            assert demo[ACTION_KEY].shape == (length, 7)
            assert demo["rewards"].shape == (length,)
            assert demo["dones"].shape == (length,)
            assert demo["dones"][-1] == 1
            assert np.all(np.isfinite(demo[ACTION_KEY][:]))
            np.testing.assert_allclose(
                demo[ACTION_KEY][:, :6], demo["next_obs/agent_pos"][:], atol=1e-6
            )
            np.testing.assert_allclose(
                demo[ACTION_KEY][:, 6], demo["actions"][:, 6], atol=1e-6
            )
            assert demo["obs/agent_pos"].shape == (length, 6)
            assert demo["obs/task_pointcloud"].shape == (length, 256, 3)
            assert np.all(np.isfinite(demo["obs/task_pointcloud"][:]))
        assert total == int(data.attrs["total"])
        train = decode_mask(input_file["mask/train"][:])
        valid = decode_mask(input_file["mask/valid"][:])
        assert set(train).isdisjoint(valid)
        assert set(train) | set(valid) == set(demos)
        env_meta = json.loads(data.attrs["env_args"])
        first = data[demos[0]]
        replay_state = {
            "states": first["states"][0],
            "model": first.attrs["model_file"],
        }
        expected = {key: first[f"obs/{key}"][0] for key in OBS_KEYS}

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["agent_pos"], "rgb": [], "depth": [], "scan": ["task_pointcloud"]}}
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )
    try:
        replayed = env.reset_to(replay_state)
        agent_error = float(np.max(np.abs(replayed["agent_pos"] - expected["agent_pos"])))
        replay_cloud = replayed["task_pointcloud"]
        expected_cloud = expected["task_pointcloud"]
        cloud_error = float(
            max(
                cKDTree(replay_cloud).query(expected_cloud)[0].max(),
                cKDTree(expected_cloud).query(replay_cloud)[0].max(),
            )
        )
        if agent_error > 1e-5 or cloud_error > 1e-4:
            raise AssertionError(
                f"initial observation replay mismatch agent={agent_error} cloud={cloud_error}"
            )
        next_obs, _, _, _ = env.step(np.asarray(first_action(path), dtype=np.float32))
        assert next_obs["agent_pos"].shape == (6,)
        assert next_obs["task_pointcloud"].shape == (256, 3)
    finally:
        env.env.close()
    return {
        "demos": len(demos),
        "transitions": total,
        "train_demos": len(train),
        "valid_demos": len(valid),
        "agent_replay_max_abs_error": agent_error,
        "pointcloud_replay_symmetric_set_error": cloud_error,
    }


def first_action(path):
    with h5py.File(path, "r") as input_file:
        first_name = sorted_demo_names(input_file["data"])[0]
        return input_file[f"data/{first_name}/actions"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset.resolve()), indent=2))


if __name__ == "__main__":
    main()
