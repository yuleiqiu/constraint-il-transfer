"""
Replay stored delta-EEF full-pose OSC actions through the dataset env.

This verifies that delta_eef_pose_action and env metadata are jointly
executable. The dataset metadata must request OSC_POSE delta/world with
identity EEF scaling and controller_goal_update_mode=desired.

Run from repo root:
    MUJOCO_GL=egl uv run python scripts/eef_pose_osc_policy/verify_delta_eef_pose_osc_dataset.py \
        --dataset /tmp/image_v15_delta_eef_pose_osc_smoke.hdf5 --n-demos 2
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite.utils.control_utils as CU
import robosuite.utils.transform_utils as T


DEFAULT_ACTION_KEY = "delta_eef_pose_action"


def sorted_demo_keys(data_group):
    return sorted(data_group.keys(), key=lambda x: int(x.split("_")[-1]))


def quat_angular_distance(q1_xyzw, q2_xyzw):
    m1 = T.quat2mat(q1_xyzw)
    m2 = T.quat2mat(q2_xyzw)
    return np.linalg.norm(CU.orientation_error(m2, m1))


def initial_state_from_demo(group, states, is_robosuite_env):
    initial_state = dict(states=states[0])
    if is_robosuite_env:
        initial_state["model"] = group.attrs["model_file"]
        initial_state["ep_meta"] = group.attrs.get("ep_meta", None)
    return initial_state


def task_success(env):
    return bool(env.is_success().get("task", False))


def validate_controller_metadata(env_meta):
    assert env_meta["env_kwargs"].get("controller_goal_update_mode") == "desired"
    body_parts = env_meta["env_kwargs"]["controller_configs"]["body_parts"]
    for part_name, body_part in body_parts.items():
        assert body_part["type"] == "OSC_POSE", (part_name, body_part["type"])
        assert body_part["input_type"] == "delta", (part_name, body_part["input_type"])
        assert body_part["input_ref_frame"] == "world", (part_name, body_part["input_ref_frame"])
        assert float(body_part["kp"]) == 500.0, (part_name, body_part["kp"])
        assert np.allclose(body_part["input_min"], [-1.0] * 6)
        assert np.allclose(body_part["input_max"], [1.0] * 6)
        assert np.allclose(body_part["output_min"], [-1.0] * 6)
        assert np.allclose(body_part["output_max"], [1.0] * 6)


def replay_demo(env, group, demo_name, action_key, is_robosuite_env, max_steps=None):
    states = group["states"][:]
    actions = group[action_key][:]
    if max_steps is not None:
        actions = actions[:max_steps]

    obs = env.reset_to(initial_state_from_demo(group, states, is_robosuite_env))
    pos_errs = []
    ori_errs = []
    any_success = task_success(env)
    max_abs_action = float(np.max(np.abs(actions)))

    for i, action in enumerate(actions):
        obs, _, _, _ = env.step(action)
        any_success = any_success or task_success(env)
        pos_errs.append(float(np.linalg.norm(obs["robot0_eef_pos"] - group["next_obs/robot0_eef_pos"][i])))
        if "robot0_eef_quat_site" in obs:
            ori_errs.append(
                float(quat_angular_distance(obs["robot0_eef_quat_site"], group["next_obs/robot0_eef_quat_site"][i]) * 180 / np.pi)
            )

    return {
        "demo": demo_name,
        "steps": int(actions.shape[0]),
        "any_success": bool(any_success),
        "final_success": task_success(env),
        "max_abs_action": max_abs_action,
        "pos_err_mean_cm": float(np.mean(pos_errs) * 100),
        "pos_err_max_cm": float(np.max(pos_errs) * 100),
        "ori_err_mean_deg": float(np.mean(ori_errs)) if ori_errs else None,
        "ori_err_max_deg": float(np.max(ori_errs)) if ori_errs else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--action-key", type=str, default=DEFAULT_ACTION_KEY)
    parser.add_argument("--n-demos", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs=dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=[]))
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(args.dataset))
    validate_controller_metadata(env_meta)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    results = []
    with h5py.File(args.dataset, "r") as f:
        demo_names = sorted_demo_keys(f["data"])[:args.n_demos]
        for demo_name in demo_names:
            group = f[f"data/{demo_name}"]
            if args.action_key not in group:
                raise KeyError(f"{args.action_key} missing in {demo_name}")
            if group[args.action_key].shape[1] != 7:
                raise ValueError(f"{args.action_key} must be 7D, got {group[args.action_key].shape}")
            result = replay_demo(env, group, demo_name, args.action_key, is_robosuite_env, args.max_steps)
            results.append(result)
            print(
                f"{demo_name}: steps={result['steps']} final_success={int(result['final_success'])} "
                f"pos_mean={result['pos_err_mean_cm']:.2f}cm pos_max={result['pos_err_max_cm']:.2f}cm "
                f"ori_mean={result['ori_err_mean_deg']:.2f}deg max_abs_action={result['max_abs_action']:.3f}"
            )

    summary = {
        "num_demos": len(results),
        "final_success_count": int(sum(r["final_success"] for r in results)),
        "final_success_rate": float(np.mean([r["final_success"] for r in results])) if results else 0.0,
        "pos_err_mean_cm": float(np.mean([r["pos_err_mean_cm"] for r in results])) if results else 0.0,
        "pos_err_max_cm": float(np.max([r["pos_err_max_cm"] for r in results])) if results else 0.0,
        "ori_err_mean_deg": float(np.mean([r["ori_err_mean_deg"] for r in results])) if results else 0.0,
        "max_abs_action": float(np.max([r["max_abs_action"] for r in results])) if results else 0.0,
    }
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"Wrote {args.output_json}")

    env.env.close()


if __name__ == "__main__":
    main()
