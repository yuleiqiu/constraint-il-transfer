"""
Diagnose whether a delta-EEF-pose diffusion policy action chunk can be
converted back into the EEF pose trajectory observed during rollout.

For each policy resampling point, this script:
  1. samples a full action chunk from the diffusion policy,
  2. unnormalizes it with the checkpoint action stats,
  3. reconstructs predicted position and orientation trajectories from the
     delta action chunk,
  4. executes the chunk in the environment, and
  5. compares predicted EEF pose against actual EEF pose after each step.

Run from repo root:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 ROBOMIMIC_GPU_ID=0 \
    uv run python scripts/eef_pose_osc_policy/diagnose_delta_eef_policy_traj.py \
      --agent outputs/robomimic/train/dp_can_delta_pose_osc/20260707222943/models/model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth \
      --n-rollouts 2 \
      --output outputs/eef_pose_osc_policy/delta_policy_traj_diagnostic/epoch260.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robosuite.utils.transform_utils as T


DEFAULT_AGENT = (
    "outputs/robomimic/train/dp_can_delta_pose_osc/20260707222943/models/"
    "model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth"
)


def as_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: as_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_serializable(v) for v in value]
    return value


def quat_to_mat_xyzw(quat):
    return T.quat2mat(np.asarray(quat, dtype=np.float64).copy())


def mat_to_quat_xyzw(mat):
    return T.mat2quat(np.asarray(mat, dtype=np.float64))


def rotvec_to_mat(rotvec):
    return T.quat2mat(T.axisangle2quat(np.asarray(rotvec, dtype=np.float64)))


def angular_distance_deg(q_pred_xyzw, q_actual_xyzw):
    pred = quat_to_mat_xyzw(q_pred_xyzw)
    actual = quat_to_mat_xyzw(q_actual_xyzw)
    delta = actual @ pred.T
    cos_angle = (np.trace(delta) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def get_pose_obs(obs, quat_key):
    if "robot0_eef_pos" not in obs:
        raise KeyError("Observation is missing robot0_eef_pos")
    if quat_key not in obs:
        raise KeyError(f"Observation is missing {quat_key}")
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    quat = np.asarray(obs[quat_key], dtype=np.float64)
    if pos.ndim >= 2:
        pos = pos[-1]
    if quat.ndim >= 2:
        quat = quat[-1]
    return (
        pos.reshape(3),
        quat.reshape(4),
    )


def choose_quat_key(obs, requested_key):
    if requested_key in obs:
        return requested_key
    fallback = "robot0_eef_quat"
    if fallback in obs:
        print(f"WARNING: {requested_key} missing; falling back to {fallback}", flush=True)
        return fallback
    raise KeyError(f"Neither {requested_key} nor {fallback} is present in observations")


def get_action_scale_offset(policy):
    stats = getattr(policy, "action_normalization_stats", None)
    if stats is None:
        return None, None
    action_keys = policy.policy.global_config.train.action_keys
    scales = []
    offsets = []
    for key in action_keys:
        if key not in stats:
            raise KeyError(f"Action normalization stats missing key {key}")
        scales.append(np.asarray(stats[key]["scale"], dtype=np.float32).reshape(-1))
        offsets.append(np.asarray(stats[key]["offset"], dtype=np.float32).reshape(-1))
    return np.concatenate(scales, axis=0), np.concatenate(offsets, axis=0)


def unnormalize_action_chunk(policy, chunk_norm):
    chunk = np.asarray(chunk_norm, dtype=np.float32)
    scale, offset = get_action_scale_offset(policy)
    if scale is None:
        return chunk
    return chunk * scale.reshape(1, -1) + offset.reshape(1, -1)


def sample_action_chunk(policy, obs):
    prepared_obs = policy._prepare_observation(obs)
    with torch.no_grad():
        chunk_norm = policy.policy._get_action_trajectory(obs_dict=prepared_obs)[0]
    return unnormalize_action_chunk(policy, chunk_norm.detach().cpu().numpy())


def env_action_limits(env, action_dim):
    raw_env = getattr(env, "unwrapped", env)
    robosuite_env = getattr(raw_env, "env", raw_env)
    if not hasattr(robosuite_env, "action_spec"):
        low = -np.ones(action_dim, dtype=np.float32)
        high = np.ones(action_dim, dtype=np.float32)
        return low, high
    low, high = robosuite_env.action_spec
    low = np.asarray(low, dtype=np.float32).reshape(-1)
    high = np.asarray(high, dtype=np.float32).reshape(-1)
    if low.shape[0] != action_dim or high.shape[0] != action_dim:
        raise ValueError(f"Env action spec shape mismatch: low={low.shape} high={high.shape} action_dim={action_dim}")
    return low, high


def clip_action_chunk(chunk, low, high):
    return np.clip(chunk, low.reshape(1, -1), high.reshape(1, -1))


def predict_pose_traj(start_pos, start_quat, action_chunk):
    pos = np.asarray(start_pos, dtype=np.float64).copy()
    rot = quat_to_mat_xyzw(start_quat)
    pred_pos = []
    pred_quat = []
    for action in action_chunk:
        pos = pos + np.asarray(action[:3], dtype=np.float64)
        rot = rotvec_to_mat(action[3:6]) @ rot
        pred_pos.append(pos.copy())
        pred_quat.append(mat_to_quat_xyzw(rot))
    return np.asarray(pred_pos), np.asarray(pred_quat)


def pose_error_summary(pos_err_cm, ori_err_deg):
    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return dict(mean=None, median=None, p90=None, max=None)
        return dict(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            p90=float(np.percentile(values, 90)),
            max=float(np.max(values)),
        )

    return {
        "pos_cm": stats(pos_err_cm),
        "ori_deg": stats(ori_err_deg),
    }


def task_success(env):
    return bool(env.is_success().get("task", False))


def rollout_with_chunk_diagnostics(policy, env, horizon, quat_key, terminate_on_success):
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)
    quat_key = choose_quat_key(obs, quat_key)

    low, high = env_action_limits(env, action_dim=policy.policy.ac_dim)
    chunks = []
    all_raw_pos_err = []
    all_raw_ori_err = []
    all_clipped_pos_err = []
    all_clipped_ori_err = []
    total_reward = 0.0
    step_i = 0
    success = task_success(env)

    while step_i < horizon:
        start_step = step_i
        start_pos, start_quat = get_pose_obs(obs, quat_key)
        raw_chunk = sample_action_chunk(policy, obs)
        clipped_chunk = clip_action_chunk(raw_chunk, low, high)
        action_horizon = int(raw_chunk.shape[0])

        raw_pred_pos, raw_pred_quat = predict_pose_traj(start_pos, start_quat, raw_chunk)
        clipped_pred_pos, clipped_pred_quat = predict_pose_traj(start_pos, start_quat, clipped_chunk)

        actual_pos = []
        actual_quat = []
        rewards = []
        dones = []
        executed_actions = []
        for chunk_i in range(action_horizon):
            if step_i >= horizon:
                break
            action = raw_chunk[chunk_i]
            next_obs, reward, done, _ = env.step(action)
            total_reward += float(reward)
            success = success or task_success(env)

            pos, quat = get_pose_obs(next_obs, quat_key)
            actual_pos.append(pos)
            actual_quat.append(quat)
            rewards.append(float(reward))
            dones.append(bool(done))
            executed_actions.append(action.copy())

            obs = next_obs
            step_i += 1
            if done or (terminate_on_success and success):
                break

        actual_pos = np.asarray(actual_pos, dtype=np.float64)
        actual_quat = np.asarray(actual_quat, dtype=np.float64)
        executed_len = int(actual_pos.shape[0])

        raw_pos_err = np.linalg.norm(raw_pred_pos[:executed_len] - actual_pos, axis=1) * 100.0
        clipped_pos_err = np.linalg.norm(clipped_pred_pos[:executed_len] - actual_pos, axis=1) * 100.0
        raw_ori_err = np.asarray(
            [
                angular_distance_deg(raw_pred_quat[i], actual_quat[i])
                for i in range(executed_len)
            ],
            dtype=np.float64,
        )
        clipped_ori_err = np.asarray(
            [
                angular_distance_deg(clipped_pred_quat[i], actual_quat[i])
                for i in range(executed_len)
            ],
            dtype=np.float64,
        )

        all_raw_pos_err.extend(raw_pos_err.tolist())
        all_raw_ori_err.extend(raw_ori_err.tolist())
        all_clipped_pos_err.extend(clipped_pos_err.tolist())
        all_clipped_ori_err.extend(clipped_ori_err.tolist())

        clip_abs_delta = np.abs(raw_chunk - clipped_chunk)
        chunk_record = {
            "start_step": int(start_step),
            "executed_len": executed_len,
            "action_horizon": action_horizon,
            "start_pos": start_pos,
            "start_quat_xyzw": start_quat,
            "raw_action_chunk": raw_chunk,
            "clipped_action_chunk": clipped_chunk,
            "action_clip_max_abs_delta": float(np.max(clip_abs_delta)) if clip_abs_delta.size else 0.0,
        "action_clip_count": int(np.sum(clip_abs_delta > 1e-5)),
            "raw_pred_pos": raw_pred_pos[:executed_len],
            "raw_pred_quat_xyzw": raw_pred_quat[:executed_len],
            "clipped_pred_pos": clipped_pred_pos[:executed_len],
            "clipped_pred_quat_xyzw": clipped_pred_quat[:executed_len],
            "actual_pos": actual_pos,
            "actual_quat_xyzw": actual_quat,
            "raw_pos_err_cm": raw_pos_err,
            "raw_ori_err_deg": raw_ori_err,
            "clipped_pos_err_cm": clipped_pos_err,
            "clipped_ori_err_deg": clipped_ori_err,
            "rewards": rewards,
            "dones": dones,
        }
        chunks.append(chunk_record)

        if executed_len == 0 or bool(dones[-1]) or (terminate_on_success and success):
            break

    summary = {
        "horizon": int(step_i),
        "success": bool(success),
        "return": float(total_reward),
        "num_chunks": len(chunks),
        "raw": pose_error_summary(all_raw_pos_err, all_raw_ori_err),
        "clipped": pose_error_summary(all_clipped_pos_err, all_clipped_ori_err),
        "action_clip_count": int(sum(c["action_clip_count"] for c in chunks)),
        "action_clip_max_abs_delta": float(max([c["action_clip_max_abs_delta"] for c in chunks], default=0.0)),
        "quat_obs_key": quat_key,
    }
    return summary, chunks


def aggregate_rollout_summaries(rollouts):
    raw_pos = []
    raw_ori = []
    clipped_pos = []
    clipped_ori = []
    for rollout in rollouts:
        for chunk in rollout["chunks"]:
            raw_pos.extend(chunk["raw_pos_err_cm"])
            raw_ori.extend(chunk["raw_ori_err_deg"])
            clipped_pos.extend(chunk["clipped_pos_err_cm"])
            clipped_ori.extend(chunk["clipped_ori_err_deg"])
    return {
        "num_rollouts": len(rollouts),
        "success_rate": float(np.mean([r["summary"]["success"] for r in rollouts])) if rollouts else 0.0,
        "mean_horizon": float(np.mean([r["summary"]["horizon"] for r in rollouts])) if rollouts else 0.0,
        "raw": pose_error_summary(raw_pos, raw_ori),
        "clipped": pose_error_summary(clipped_pos, clipped_ori),
        "action_clip_count": int(sum(r["summary"]["action_clip_count"] for r in rollouts)),
        "action_clip_max_abs_delta": float(max([r["summary"]["action_clip_max_abs_delta"] for r in rollouts], default=0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=Path(DEFAULT_AGENT))
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--n-rollouts", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quat-obs-key", type=str, default="robot0_eef_quat_site")
    parser.add_argument("--terminate-on-success", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/eef_pose_osc_policy/delta_policy_traj_diagnostic/epoch260.json"))
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(args.agent), device=device, verbose=False)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=False,
        verbose=False,
    )

    rollouts = []
    for rollout_i in range(args.n_rollouts):
        print(f"Rollout {rollout_i + 1}/{args.n_rollouts} start", flush=True)
        summary, chunks = rollout_with_chunk_diagnostics(
            policy=policy,
            env=env,
            horizon=args.horizon,
            quat_key=args.quat_obs_key,
            terminate_on_success=args.terminate_on_success,
        )
        rollouts.append({"summary": summary, "chunks": chunks})
        print(
            "Rollout {}/{} done: success={} horizon={} chunks={} "
            "raw_pos_mean={:.3f}cm raw_ori_mean={:.3f}deg "
            "clip_count={}".format(
                rollout_i + 1,
                args.n_rollouts,
                int(summary["success"]),
                summary["horizon"],
                summary["num_chunks"],
                summary["raw"]["pos_cm"]["mean"],
                summary["raw"]["ori_deg"]["mean"],
                summary["action_clip_count"],
            ),
            flush=True,
        )

    aggregate = aggregate_rollout_summaries(rollouts)
    result = {
        "agent": str(args.agent),
        "env": args.env,
        "horizon": args.horizon,
        "n_rollouts": args.n_rollouts,
        "terminate_on_success": args.terminate_on_success,
        "aggregate": aggregate,
        "rollouts": rollouts,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(as_serializable(result), f, indent=2)
    print(json.dumps(as_serializable({"aggregate": aggregate}), indent=2))
    print(f"Wrote {args.output}")

    raw_env = getattr(env, "unwrapped", env)
    robosuite_env = getattr(raw_env, "env", raw_env)
    close_fn = getattr(robosuite_env, "close", None)
    if close_fn is not None:
        close_fn()


if __name__ == "__main__":
    main()
