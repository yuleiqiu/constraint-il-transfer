"""CPU-only tests for paired-correction action and time alignment."""

import tempfile
from pathlib import Path

import h5py
import numpy as np

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.dataset import SequenceDataset

import paired_correction as Pair


def _synthetic_episode(length=24):
    actions = np.arange(length * 7, dtype=np.float32).reshape(length, 7)
    queries = 1000.0 + actions
    controller = np.zeros(length, dtype=np.uint8)
    controller[4:21] = Pair.TEACHER_CONTROLLER
    intervention_id = np.full(length, -1, dtype=np.int32)
    intervention_id[4:21] = 3
    query_valid = controller == Pair.TEACHER_CONTROLLER
    queries[~query_valid] = np.nan
    return {
        "obs": {
            "state": np.arange(length * 2, dtype=np.float32).reshape(length, 2)
        },
        "delta_eef_pose_action": actions,
        "controller": controller,
        "intervention_id": intervention_id,
        "robot_query_action": queries,
        "robot_query_valid": query_valid,
    }


def test_pair_extraction_uses_shared_history_and_per_step_queries():
    episode = _synthetic_episode()
    pairs = Pair.derive_episode_pairs(episode, obs_keys=["state"])
    assert pairs["anchor_step"].tolist() == [4, 5, 6]
    for index, anchor in enumerate(pairs["anchor_step"]):
        positive = pairs["positive_actions"][index]
        negative = pairs["negative_actions"][index]
        assert np.array_equal(positive[0], episode["delta_eef_pose_action"][anchor - 1])
        assert np.array_equal(negative[0], positive[0])
        assert np.array_equal(
            positive[1:], episode["delta_eef_pose_action"][anchor : anchor + 15]
        )
        assert np.array_equal(
            negative[1:], episode["robot_query_action"][anchor : anchor + 15]
        )
        assert np.array_equal(
            pairs["obs"]["state"][index],
            episode["obs"]["state"][anchor - 1 : anchor + 1],
        )


def test_pair_extraction_rejects_mixed_or_missing_query_windows():
    episode = _synthetic_episode()
    episode["robot_query_valid"][12] = False
    episode["intervention_id"][18] = 4
    pairs = Pair.derive_episode_pairs(episode, obs_keys=["state"])
    assert len(pairs["anchor_step"]) == 0


def test_realized_delta_action_uses_achieved_pose_and_commanded_gripper():
    obs = {
        "robot0_eef_pos": np.array([[0.0, 0.0, 0.0], [0.1, -0.1, 0.2]]),
        "robot0_eef_quat_site": np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        ),
    }
    next_obs = {
        "robot0_eef_pos": np.array(
            [[0.1, -0.1, 0.2], [0.11, -0.08, 0.17]]
        ),
        "robot0_eef_quat_site": np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        ),
    }
    command = np.array([0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0])
    action = Pair.realized_delta_eef_pose_action(obs, next_obs, command)
    assert np.allclose(action[:3], [0.01, 0.02, -0.03])
    assert np.allclose(action[3:6], 0.0)
    assert action[6] == 1.0


def _write_tiny_sequence_dataset(path):
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        demo = data.create_group("demo_0")
        length = 24
        demo.attrs["num_samples"] = length
        obs = demo.create_group("obs")
        obs.create_dataset(
            "state", data=np.arange(length, dtype=np.float32)[:, None]
        )
        demo.create_dataset(
            "delta_eef_pose_action",
            data=np.repeat(
                np.arange(length, dtype=np.float32)[:, None], 7, axis=1
            ),
        )


def test_alignment_matches_robomimic_sequence_dataset():
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": ["state"]})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.hdf5"
        _write_tiny_sequence_dataset(path)
        dataset = SequenceDataset(
            hdf5_path=str(path),
            obs_keys=("state",),
            action_keys=("delta_eef_pose_action",),
            dataset_keys=("delta_eef_pose_action",),
            action_config={
                "delta_eef_pose_action": {"normalization": None}
            },
            frame_stack=2,
            seq_length=16,
            pad_frame_stack=True,
            pad_seq_length=True,
            hdf5_cache_mode=None,
            load_next_obs=False,
        )
        anchor = 4
        item = dataset[anchor]
        raw_obs = {
            "state": np.arange(24, dtype=np.float32)[:, None]
        }
        raw_actions = np.repeat(
            np.arange(24, dtype=np.float32)[:, None], 7, axis=1
        )
        expected_obs, expected_actions = Pair.aligned_source_sample(
            raw_obs, raw_actions, anchor
        )
        assert np.array_equal(item["obs"]["state"][:2], expected_obs["state"])
        assert np.array_equal(item["actions"][:16], expected_actions)
        assert np.array_equal(
            item["actions"][1:9], raw_actions[anchor : anchor + 8]
        )


def test_raw_to_pair_hdf5_round_trip():
    synthetic = _synthetic_episode()
    length = len(synthetic["controller"])
    episode = {
        **synthetic,
        "next_obs": {
            "state": synthetic["obs"]["state"] + 1.0,
        },
        "states": np.arange(length * 3, dtype=np.float64).reshape(length, 3),
        "actions": synthetic["delta_eef_pose_action"].copy(),
        "rewards": np.zeros(length, dtype=np.float32),
        "dones": np.zeros(length, dtype=bool),
        "robot_query_seed": np.where(
            synthetic["robot_query_valid"], np.arange(length), -1
        ),
        "target_contact": np.zeros(length, dtype=bool),
        "distractor_contact": np.zeros(length, dtype=bool),
    }
    with tempfile.TemporaryDirectory() as directory:
        raw_path = Path(directory) / "raw.hdf5"
        pair_path = Path(directory) / "pairs.hdf5"
        with h5py.File(raw_path, "w") as raw:
            Pair.initialize_raw_file(
                raw,
                source_checkpoint="checkpoint.pth",
                source_checkpoint_sha256="checkpoint-hash",
                env_metadata={"env_name": "Synthetic"},
                policy_obs_keys=["state"],
                raw_obs_keys=["state"],
                action_offset=np.zeros(7),
                action_scale=np.ones(7),
            )
            Pair.append_raw_episode(
                raw,
                episode_id=0,
                episode=episode,
                attrs={
                    "layout_id": "layout-0",
                    "selected_yaw_deg": 20.0,
                    "train_eligible": True,
                    "split": "train",
                },
            )
        summary = Pair.extract_pair_file(raw_path, pair_path)
        assert summary["pairs"] == 3
        assert summary["train_pairs"] == 3
        with h5py.File(pair_path, "r") as pairs:
            assert pairs.attrs["schema_version"] == Pair.PAIR_SCHEMA_VERSION
            assert pairs["pairs/positive_actions"].shape == (3, 16, 7)
            assert pairs["pairs/obs/state"].shape == (3, 2, 2)
            assert [
                value.decode("utf-8")
                for value in pairs["mask/train_episode_ids"][()]
            ] == ["demo_0"]


def main():
    test_pair_extraction_uses_shared_history_and_per_step_queries()
    test_pair_extraction_rejects_mixed_or_missing_query_windows()
    test_realized_delta_action_uses_achieved_pose_and_commanded_gripper()
    test_alignment_matches_robomimic_sequence_dataset()
    test_raw_to_pair_hdf5_round_trip()
    print("paired-correction tests passed")


if __name__ == "__main__":
    main()
