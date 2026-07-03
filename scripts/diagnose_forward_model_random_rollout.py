"""Validate OSC forward model on random-action closed-loop rollouts.

This diagnostic separates forward-model accuracy from obstacle guidance:
it executes random actions in the real robosuite environment, records the
executed EEF trajectory, and compares sliding-window forward-model
predictions against the recorded future EEF positions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.obstacle_guidance_utils import get_raw_env
from robomimic.scripts.run_obstacle_guided_agent import (
    env_from_checkpoint_for_guidance,
    get_forward_model_state_from_obs,
    get_latest_obs_value,
)
from robomimic.utils.osc_forward_model_utils import load_osc_forward_model


DEFAULT_AGENT = (
    "outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/"
    "model_epoch_140_image_v15_can_mask_success_1.0.pth"
)
DEFAULT_FORWARD_MODEL = "outputs/forward_model/osc_eef_forward_image_v15/model.pth"
DEFAULT_OUTPUT_DIR = "outputs/forward_model/random_rollout_validation"
DEFAULT_STATE_OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def get_action_bounds(env):
    low, high = get_raw_env(env).action_spec
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def sample_action(rng, low, high, action_scale):
    center = (low + high) * 0.5
    half_range = (high - low) * 0.5 * float(action_scale)
    return rng.uniform(center - half_range, center + half_range).astype(np.float32)


def cumsum_abs_traj(current_eef, action_chunk):
    return current_eef[None, :] + np.cumsum(action_chunk[:, :3] * 0.05, axis=0)


def prediction_metrics(pred, target):
    err = pred - target
    l2 = np.linalg.norm(err, axis=-1)
    per_step_rmse = np.sqrt(np.mean(np.sum(err**2, axis=-1), axis=0))
    return {
        "n_windows": int(pred.shape[0]),
        "traj_rmse_cm": float(np.sqrt(np.mean(np.sum(err**2, axis=-1))) * 100.0),
        "traj_mae_cm": float(np.mean(np.abs(err)) * 100.0),
        "terminal_error_mean_cm": float(np.mean(l2[:, -1]) * 100.0),
        "terminal_error_median_cm": float(np.median(l2[:, -1]) * 100.0),
        "terminal_error_p90_cm": float(np.quantile(l2[:, -1], 0.9) * 100.0),
        "per_step_rmse_cm": [float(x * 100.0) for x in per_step_rmse],
    }


def evaluate_rollout(records, forward_model, state_obs_keys, device):
    horizon = forward_model.horizon
    states = records["states"]
    actions = records["actions"]
    next_eef_pos = records["next_eef_pos"]
    n_windows = len(actions) - horizon + 1
    if n_windows <= 0:
        raise ValueError(f"Need at least horizon={horizon} actions, got {len(actions)}")

    state_batch = []
    action_batch = []
    target_batch = []
    cumsum_batch = []
    for start in range(n_windows):
        state = states[start]
        action_chunk = actions[start : start + horizon]
        current_eef = state[:3]
        target = next_eef_pos[start : start + horizon]
        state_batch.append(state)
        action_batch.append(action_chunk)
        target_batch.append(target)
        cumsum_batch.append(cumsum_abs_traj(current_eef, action_chunk))

    state_t = torch.from_numpy(np.asarray(state_batch, dtype=np.float32)).to(device)
    action_t = torch.from_numpy(np.asarray(action_batch, dtype=np.float32)).to(device)
    with torch.no_grad():
        pred = forward_model.predict_abs_traj(state_t, action_t).detach().cpu().numpy()

    target_arr = np.asarray(target_batch, dtype=np.float32)
    cumsum_arr = np.asarray(cumsum_batch, dtype=np.float32)
    return {
        "model": prediction_metrics(pred, target_arr),
        "cumsum_action_scale_0p05": prediction_metrics(cumsum_arr, target_arr),
    }


def collect_rollout(env, rng, steps, action_scale, state_obs_keys):
    obs = env.reset()
    low, high = get_action_bounds(env)
    records = {
        "states": [],
        "actions": [],
        "eef_pos": [],
        "eef_quat": [],
        "next_eef_pos": [],
        "next_eef_quat": [],
    }

    for _ in range(steps):
        records["states"].append(get_forward_model_state_from_obs(obs, state_obs_keys))
        records["eef_pos"].append(get_latest_obs_value(obs, "robot0_eef_pos"))
        records["eef_quat"].append(get_latest_obs_value(obs, "robot0_eef_quat"))

        action = sample_action(rng=rng, low=low, high=high, action_scale=action_scale)
        next_obs, _, _, _ = env.step(action)
        records["actions"].append(action)
        records["next_eef_pos"].append(get_latest_obs_value(next_obs, "robot0_eef_pos"))
        records["next_eef_quat"].append(get_latest_obs_value(next_obs, "robot0_eef_quat"))
        obs = next_obs

    return {key: np.asarray(value, dtype=np.float32) for key, value in records.items()}


def summarize(metrics_by_rollout):
    out = {}
    for predictor in ("model", "cumsum_action_scale_0p05"):
        out[predictor] = {}
        for key in (
            "traj_rmse_cm",
            "traj_mae_cm",
            "terminal_error_mean_cm",
            "terminal_error_median_cm",
            "terminal_error_p90_cm",
        ):
            values = [m[predictor][key] for m in metrics_by_rollout]
            out[predictor][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(x) for x in values],
            }
        per_step = np.asarray([m[predictor]["per_step_rmse_cm"] for m in metrics_by_rollout], dtype=np.float64)
        out[predictor]["per_step_rmse_cm_mean"] = [float(x) for x in per_step.mean(axis=0)]
    return out


def write_summary(output_dir, result):
    aggregate = result["aggregate"]
    lines = [
        "# Forward Model Random Rollout Validation",
        "",
        f"Environment: `{result['env_name']}`",
        f"Rollouts: {result['n_rollouts']} x {result['steps']} random actions",
        f"Action scale: {result['action_scale']}",
        f"Forward-model horizon: {result['horizon']} steps",
        "",
        "This validates `state + action chunk -> executed EEF xyz trajectory` directly in env rollouts.",
        "Guidance is not involved.",
        "",
        "## Aggregate Metrics",
        "",
        "| predictor | traj RMSE cm | terminal mean cm | terminal p90 cm |",
        "|---|---:|---:|---:|",
    ]
    for predictor in ("model", "cumsum_action_scale_0p05"):
        row = aggregate[predictor]
        lines.append(
            "| {} | {:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} |".format(
                predictor,
                row["traj_rmse_cm"]["mean"],
                row["traj_rmse_cm"]["std"],
                row["terminal_error_mean_cm"]["mean"],
                row["terminal_error_mean_cm"]["std"],
                row["terminal_error_p90_cm"]["mean"],
                row["terminal_error_p90_cm"]["std"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-Rollout Metrics",
            "",
            "| rollout | model traj RMSE cm | cumsum traj RMSE cm | model terminal mean cm | cumsum terminal mean cm |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result["rollout_metrics"]:
        lines.append(
            "| {rollout} | {m_traj:.3f} | {c_traj:.3f} | {m_term:.3f} | {c_term:.3f} |".format(
                rollout=item["rollout"],
                m_traj=item["model"]["traj_rmse_cm"],
                c_traj=item["cumsum_action_scale_0p05"]["traj_rmse_cm"],
                m_term=item["model"]["terminal_error_mean_cm"],
                c_term=item["cumsum_action_scale_0p05"]["terminal_error_mean_cm"],
            )
        )

    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {key: make_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [make_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--forward_model_path", default=DEFAULT_FORWARD_MODEL)
    parser.add_argument("--env", default="PickPlaceCan")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_rollouts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state_obs_keys", nargs="+", default=list(DEFAULT_STATE_OBS_KEYS))
    args = parser.parse_args()

    device = torch.device(args.device)
    forward_model = load_osc_forward_model(resolve_path(args.forward_model_path), device=device)
    ckpt_dict = FileUtils.load_dict_from_checkpoint(str(resolve_path(args.agent)))
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=False)
    ObsUtils.initialize_obs_utils_with_config(config)
    env = env_from_checkpoint_for_guidance(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=False,
    )

    rng = np.random.default_rng(args.seed)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_metrics = []
    raw_rollouts = []
    for rollout_idx in range(args.n_rollouts):
        records = collect_rollout(
            env=env,
            rng=rng,
            steps=args.steps,
            action_scale=args.action_scale,
            state_obs_keys=args.state_obs_keys,
        )
        metrics = evaluate_rollout(
            records=records,
            forward_model=forward_model,
            state_obs_keys=args.state_obs_keys,
            device=device,
        )
        metrics["rollout"] = rollout_idx
        rollout_metrics.append(metrics)
        raw_rollouts.append(records)
        np.savez_compressed(output_dir / f"rollout_{rollout_idx:03d}.npz", **records)
        print(
            "rollout {}: model traj_rmse={:.3f}cm, cumsum traj_rmse={:.3f}cm".format(
                rollout_idx,
                metrics["model"]["traj_rmse_cm"],
                metrics["cumsum_action_scale_0p05"]["traj_rmse_cm"],
            )
        )

    result = {
        "agent": str(resolve_path(args.agent)),
        "forward_model_path": str(resolve_path(args.forward_model_path)),
        "env_name": args.env,
        "n_rollouts": args.n_rollouts,
        "steps": args.steps,
        "action_scale": args.action_scale,
        "seed": args.seed,
        "state_obs_keys": list(args.state_obs_keys),
        "horizon": forward_model.horizon,
        "rollout_metrics": rollout_metrics,
        "aggregate": summarize(rollout_metrics),
    }
    (output_dir / "metrics.json").write_text(json.dumps(make_jsonable(result), indent=2))
    write_summary(output_dir, make_jsonable(result))
    print(json.dumps(make_jsonable(result["aggregate"]), indent=2))


if __name__ == "__main__":
    main()
