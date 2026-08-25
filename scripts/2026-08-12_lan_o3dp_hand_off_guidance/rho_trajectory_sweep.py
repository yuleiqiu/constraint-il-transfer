"""Measure executed EEF trajectory separation across XY guidance scales."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.paper_lan_guidance_utils import (
    paper_guided_policy_from_checkpoint,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = (
    SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
)
SPEC = importlib.util.spec_from_file_location("paper_lan_paired_eval", PAIRED_EVAL_PATH)
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument(
        "--rhos",
        nargs="+",
        type=float,
        default=(0.01, 0.03, 0.1, 0.3, 1.0),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def longest_true_run(mask):
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def trajectory_separation(baseline, guided):
    common = min(len(baseline), len(guided))
    baseline = np.asarray(baseline[:common], dtype=np.float64)
    guided = np.asarray(guided[:common], dtype=np.float64)
    delta = guided - baseline
    xy = np.linalg.norm(delta[:, :2], axis=1)
    z = np.abs(delta[:, 2])
    xyz = np.linalg.norm(delta, axis=1)
    result = {
        "common_samples": common,
        "common_executed_steps": max(0, common - 1),
        "max_xy_separation_m": float(np.max(xy)),
        "max_z_separation_m": float(np.max(z)),
        "max_xyz_separation_m": float(np.max(xyz)),
        "mean_xy_separation_m": float(np.mean(xy)),
        "p95_xy_separation_m": float(np.percentile(xy, 95)),
        "step_of_max_xy_separation": int(np.argmax(xy)),
    }
    for threshold_m in (0.005, 0.01, 0.02):
        suffix = "{}mm".format(round(1000 * threshold_m))
        active = xy >= threshold_m
        result["samples_xy_ge_{}".format(suffix)] = int(np.count_nonzero(active))
        result["longest_run_xy_ge_{}".format(suffix)] = longest_true_run(active)
        indices = np.flatnonzero(active)
        result["first_step_xy_ge_{}".format(suffix)] = (
            int(indices[0]) if len(indices) else None
        )
    return result


def record_rollout(policy, env, state, policy_seed, run_args, condition, rho, q):
    trajectory = []
    steps = []

    def capture(frame_env, observation, step, success, collision):
        del frame_env, success, collision
        steps.append(int(step))
        trajectory.append(
            np.asarray(
                PAIRED_EVAL.rollout_common.latest_obs_value(
                    observation, "agent_pos"
                )[:3],
                dtype=np.float64,
            )
        )

    metrics = PAIRED_EVAL.rollout_condition(
        policy,
        env,
        state,
        condition,
        policy_seed,
        run_args,
        guidance_scale=rho,
        safety_distance=q if condition == "guided" else None,
        frame_callback=capture,
    )
    if steps != list(range(len(steps))):
        raise RuntimeError("Trajectory callback did not capture every executed step")
    return metrics, np.asarray(trajectory, dtype=np.float32)


def compact_outcome(metrics):
    return {
        key: metrics[key]
        for key in (
            "success",
            "safe_success",
            "collision_any",
            "ncr",
            "hand_obstacle_collision_any",
            "finger_obstacle_collision_any",
            "arm_obstacle_collision_any",
            "target_obstacle_collision_any",
            "horizon",
            "guidance_trigger_rate",
            "guidance_update_norm_mean",
            "guidance_update_norm_max",
            "first_guidance_active_step",
            "first_collision_step",
            "action_clip_count",
            "action_clip_max_abs",
            "position_action_clip_count",
            "position_action_clip_max_abs",
        )
    }


def plot_episode(path, episode, trajectories, rows):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    baseline = trajectories["baseline"]
    axes[0].plot(baseline[:, 0], baseline[:, 1], color="black", linewidth=2, label="baseline")
    axes[1].plot(np.arange(len(baseline)), baseline[:, 2], color="black", linewidth=2, label="baseline")
    for row in rows:
        rho = row["rho"]
        guided = trajectories["rho_{}".format(rho)]
        label = "rho={:g}".format(rho)
        line = axes[0].plot(guided[:, 0], guided[:, 1], linewidth=1.4, label=label)[0]
        color = line.get_color()
        axes[1].plot(np.arange(len(guided)), guided[:, 2], linewidth=1.2, color=color, label=label)
        common = min(len(baseline), len(guided))
        xy = np.linalg.norm(guided[:common, :2] - baseline[:common, :2], axis=1)
        axes[2].plot(np.arange(common), 100 * xy, linewidth=1.4, color=color, label=label)
    axes[0].set_title("Executed EEF XY")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].axis("equal")
    axes[1].set_title("Executed EEF Z vs step")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("z (m)")
    axes[2].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="1 cm gate")
    axes[2].axhline(2.0, color="gray", linestyle=":", linewidth=1, label="2 cm gate")
    axes[2].set_title("XY separation from baseline")
    axes[2].set_xlabel("common executed step")
    axes[2].set_ylabel("separation (cm)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("D2 hand-off episode {}: XY-only rho sweep".format(episode))
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_report(path, rows):
    lines = [
        "# XY-only rho trajectory sweep",
        "",
        "All runs use D2 hand-off, H600, Huberized hinge, `Q=0.04 m`, "
        "`epsilon=0.01 m`. Separation compares executed EEF positions at the "
        "same environment step against the single baseline rollout for that state.",
        "",
        "| Episode | rho | Task | Collision | Trigger | Max XY | P95 XY | Longest >=1 cm | Longest >=2 cm | Pos clips |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        outcome = row["outcome"]
        separation = row["separation"]
        lines.append(
            "| {} | {:g} | {} | {} | {:.1%} | {:.2f} cm | {:.2f} cm | {} | {} | {} |".format(
                row["episode"],
                row["rho"],
                int(outcome["success"]),
                int(outcome["collision_any"]),
                outcome["guidance_trigger_rate"],
                100 * separation["max_xy_separation_m"],
                100 * separation["p95_xy_separation_m"],
                separation["longest_run_xy_ge_10mm"],
                separation["longest_run_xy_ge_20mm"],
                outcome["position_action_clip_count"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Output already exists: {}".format(args.out_dir))
    if any(rho <= 0 for rho in args.rhos):
        raise ValueError("Every rho must be positive")
    args.out_dir.mkdir(parents=True)

    manifest = json.loads((args.eval_dir / "manifest.json").read_text())
    if manifest["envs"] != ["PickPlaceBreadCerealCan"]:
        raise ValueError("This diagnostic is frozen to D2")
    if manifest["hand_collision_enabled"]:
        raise ValueError("This diagnostic expects the hand-off evaluation")
    state_bank_path = Path(manifest["state_bank_source"])
    state_bank = PAIRED_EVAL.load_state_bank(state_bank_path)
    seed = manifest["seeds"][0]
    q = manifest["safety_distances"][0]

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = paper_guided_policy_from_checkpoint(
        ckpt_path=manifest["agent"], device=device
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=manifest["envs"][0],
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    run_args = SimpleNamespace(
        horizon=manifest["horizon"],
        terminate_on_success=True,
        terminate_on_collision=False,
        target_object_name="Can",
        xy_only=True,
        disable_hand_collision=True,
        obstacle_cost_type="huber_hinge",
        obstacle_cost_smoothing=manifest["obstacle_cost_smoothing"],
        obstacle_cost_smoothing_ratio=None,
    )
    rows = []
    try:
        for episode in args.episodes:
            key = "{}:{}:{}".format(manifest["envs"][0], seed, episode)
            record = state_bank["states"][key]
            state = record["state"]
            if PAIRED_EVAL.state_hash(state) != record["state_hash"]:
                raise RuntimeError("State hash mismatch for {}".format(key))
            policy_seed = seed * 1_000_000 + episode
            baseline_metrics, baseline = record_rollout(
                policy, env, state, policy_seed, run_args, "baseline", 0.0, q
            )
            trajectories = {"baseline": baseline}
            episode_rows = []
            for rho in args.rhos:
                guided_metrics, guided = record_rollout(
                    policy, env, state, policy_seed, run_args, "guided", rho, q
                )
                trajectories["rho_{}".format(rho)] = guided
                row = {
                    "episode": episode,
                    "rho": rho,
                    "initial_state_hash": record["state_hash"],
                    "policy_seed": policy_seed,
                    "baseline_outcome": compact_outcome(baseline_metrics),
                    "outcome": compact_outcome(guided_metrics),
                    "separation": trajectory_separation(baseline, guided),
                }
                rows.append(row)
                episode_rows.append(row)
                print(
                    "episode={} rho={:g} task={} collision={} max_xy={:.2f}cm run1cm={}".format(
                        episode,
                        rho,
                        guided_metrics["success"],
                        guided_metrics["collision_any"],
                        100 * row["separation"]["max_xy_separation_m"],
                        row["separation"]["longest_run_xy_ge_10mm"],
                    ),
                    flush=True,
                )
            np.savez_compressed(
                args.out_dir / "episode_{:03d}_trajectories.npz".format(episode),
                **trajectories,
            )
            plot_episode(
                args.out_dir / "episode_{:03d}_trajectory.png".format(episode),
                episode,
                trajectories,
                episode_rows,
            )
    finally:
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env).close()

    (args.out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_report(args.out_dir / "report.md", rows)
    output_manifest = {
        "source_eval": str(args.eval_dir.resolve()),
        "state_bank": str(state_bank_path.resolve()),
        "agent": manifest["agent"],
        "environment": manifest["envs"][0],
        "seed": seed,
        "episodes": args.episodes,
        "rhos": args.rhos,
        "horizon": manifest["horizon"],
        "xy_only": True,
        "hand_collision_enabled": False,
        "obstacle_cost_type": "huber_hinge",
        "safety_distance_m": q,
        "obstacle_cost_smoothing_m": manifest["obstacle_cost_smoothing"],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
