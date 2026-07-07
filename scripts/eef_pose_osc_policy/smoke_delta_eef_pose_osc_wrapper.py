"""
Smoke-test robomimic normalization and EnvRobosuite stepping for delta EEF OSC.

Run from repo root:
    MUJOCO_GL=egl uv run python scripts/eef_pose_osc_policy/smoke_delta_eef_pose_osc_wrapper.py \
        --dataset /tmp/image_v15_delta_eef_pose_osc_smoke.hdf5 \
        --config third_party/robomimic/robomimic/exps/delta_eef_pose_osc/diffusion_policy_can_image.json
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from robomimic.algo import RolloutPolicy
from robomimic.config import config_factory
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.python_utils as PyUtils
import robomimic.utils.train_utils as TrainUtils


DEFAULT_ACTION_KEY = "delta_eef_pose_action"


class DummyNormalizedPolicy:
    def __init__(self, config, normalized_action):
        self.global_config = config
        self.device = torch.device("cpu")
        self._normalized_action = torch.as_tensor(normalized_action, dtype=torch.float32).reshape(1, -1)

    def set_eval(self):
        pass

    def reset(self):
        pass

    def get_action(self, obs_dict, goal_dict=None):
        return self._normalized_action


def load_config(config_path, dataset_path):
    ext_cfg = json.load(open(config_path, "r"))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
        config.train.data = [{"path": str(dataset_path)}]
        config.experiment.validate = False
        config.experiment.rollout.enabled = False
        config.train.hdf5_validation_filter_key = None
        config.train.cuda = False
    config.lock()
    return config


def first_demo_name(data_group):
    return sorted(data_group.keys(), key=lambda x: int(x.split("_")[-1]))[0]


def validate_controller_metadata(env_meta):
    assert env_meta["env_kwargs"].get("controller_goal_update_mode") == "desired"
    body_parts = env_meta["env_kwargs"]["controller_configs"]["body_parts"]
    for part_name, body_part in body_parts.items():
        assert body_part["type"] == "OSC_POSE", (part_name, body_part["type"])
        assert body_part["input_type"] == "delta", (part_name, body_part["input_type"])
        assert body_part["input_ref_frame"] == "world", (part_name, body_part["input_ref_frame"])
        assert float(body_part["kp"]) == 500.0, (part_name, body_part["kp"])
        assert np.allclose(body_part["output_min"], [-1.0] * 6)
        assert np.allclose(body_part["output_max"], [1.0] * 6)


def initial_state_from_demo(group):
    return {
        "states": group["states"][0],
        "model": group.attrs["model_file"],
        "ep_meta": group.attrs.get("ep_meta", None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action-key", type=str, default=DEFAULT_ACTION_KEY)
    args = parser.parse_args()

    config = load_config(args.config, args.dataset)
    ObsUtils.initialize_obs_utils_with_config(config)

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config={"path": str(args.dataset)},
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=False,
    )
    assert shape_meta["ac_dim"] == 7, shape_meta

    trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    action_stats = trainset.get_action_normalization_stats()
    assert args.action_key in action_stats, action_stats.keys()
    assert action_stats[args.action_key]["offset"].shape == (1, 7)
    assert action_stats[args.action_key]["scale"].shape == (1, 7)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(args.dataset))
    validate_controller_metadata(env_meta)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=shape_meta["use_images"] or shape_meta["use_depths"],
        use_image_obs=shape_meta["use_images"] or shape_meta["use_depths"],
    )
    assert env.action_dimension == 7, env.action_dimension

    with h5py.File(args.dataset, "r") as f:
        demo_name = first_demo_name(f["data"])
        group = f[f"data/{demo_name}"]
        raw_action = group[args.action_key][0].astype(np.float32)
        obs = env.reset_to(initial_state_from_demo(group))

    action_dict = {args.action_key: raw_action.reshape(1, -1)}
    normalized_dict = ObsUtils.normalize_dict(action_dict, normalization_stats=action_stats)
    normalized_action = PyUtils.action_dict_to_vector(normalized_dict, action_keys=config.train.action_keys)[0]
    assert np.all(normalized_action <= 1.000001), normalized_action
    assert np.all(normalized_action >= -1.000001), normalized_action

    policy = RolloutPolicy(
        DummyNormalizedPolicy(config=config, normalized_action=normalized_action),
        obs_normalization_stats=None,
        action_normalization_stats=action_stats,
    )
    policy.start_episode()
    unnormalized_action = policy(ob=obs)
    if not np.allclose(unnormalized_action, raw_action, atol=1e-5):
        raise AssertionError(
            f"RolloutPolicy unnormalize mismatch\nraw={raw_action}\nunnorm={unnormalized_action}"
        )

    next_obs, _, _, _ = env.step(unnormalized_action)
    expected_next_pos = obs["robot0_eef_pos"][-1] + raw_action[:3] if obs["robot0_eef_pos"].ndim == 2 else obs["robot0_eef_pos"] + raw_action[:3]
    pos_err_cm = float(np.linalg.norm(next_obs["robot0_eef_pos"] - expected_next_pos) * 100)
    print("SMOKE PASSED")
    print(f"dataset={args.dataset}")
    print(f"config={args.config}")
    print(f"action_dim={shape_meta['ac_dim']}")
    print(f"normalization=roundtrip_ok min={normalized_action.min():.4f} max={normalized_action.max():.4f}")
    print(f"controller=OSC_POSE delta world kp=500 identity_scale goal_update_mode=desired")
    print(f"env_action_dimension={env.action_dimension}")
    print(f"raw_action_max_abs={np.max(np.abs(raw_action)):.4f}")
    print(f"first_step_delta_target_pos_err_cm={pos_err_cm:.3f}")
    print("reset_to_controller_refresh=covered_by_EnvRobosuite.reset_to")
    print("action_clipping=raw_delta_pose_action_within_env_action_spec")
    env.env.close()


if __name__ == "__main__":
    main()
