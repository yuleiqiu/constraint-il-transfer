"""Measure Milk-guidance effects on executed Tabletop trajectories."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.paper_lan_guidance_utils import paper_guided_policy_from_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
SPEC = importlib.util.spec_from_file_location("tabletop_paper_lan_paired_eval", PAIRED_EVAL_PATH)
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)
ENV_NAME = "PickPlaceCanTabletopRaisedMilk"


def longest_true_run(mask):
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def trajectory_separation(baseline, guided):
    common = min(len(baseline), len(guided))
    delta = np.asarray(guided[:common], dtype=np.float64) - np.asarray(
        baseline[:common], dtype=np.float64
    )
    xy = np.linalg.norm(delta[:, :2], axis=1)
    result = {
        "common_samples": common,
        "max_xy_separation_m": float(np.max(xy)),
        "mean_xy_separation_m": float(np.mean(xy)),
        "p95_xy_separation_m": float(np.percentile(xy, 95)),
        "step_of_max_xy_separation": int(np.argmax(xy)),
    }
    for threshold in (0.005, 0.01, 0.02):
        suffix = f"{int(1000 * threshold)}mm"
        active = xy >= threshold
        indices = np.flatnonzero(active)
        result[f"samples_xy_ge_{suffix}"] = int(np.count_nonzero(active))
        result[f"longest_run_xy_ge_{suffix}"] = longest_true_run(active)
        result[f"first_step_xy_ge_{suffix}"] = int(indices[0]) if len(indices) else None
    return result


def compact_metrics(metrics):
    keys = (
        "success",
        "safe_success",
        "collision_any",
        "hand_obstacle_collision_any",
        "finger_obstacle_collision_any",
        "target_obstacle_collision_any",
        "horizon",
        "minimum_eef_obstacle_clearance_m",
        "guidance_trigger_rate",
        "guidance_update_norm_mean",
        "guidance_update_norm_max",
        "tangent_side_first",
        "tangent_side_counts",
        "first_guidance_active_step",
        "first_collision_step",
    )
    return {key: metrics[key] for key in keys}


def add_caption(frame, lines, collision):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    box_height = 10 + 15 * len(lines)
    fill = (130, 0, 0, 205) if collision else (0, 0, 0, 185)
    draw.rectangle((0, 0, image.width, box_height), fill=fill)
    for index, line in enumerate(lines):
        draw.text((7, 5 + 15 * index), line, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image)


def record_rollout(policy, env, state, policy_seed, args, condition, rho, render):
    samples = {"eef": [], "can": [], "obstacle": [], "steps": []}
    frames = []

    def capture(frame_env, observation, step, success, collision):
        raw_env = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(frame_env)
        samples["steps"].append(int(step))
        samples["eef"].append(
            np.asarray(
                PAIRED_EVAL.rollout_common.latest_obs_value(observation, "agent_pos")[:3],
                dtype=np.float64,
            )
        )
        samples["can"].append(
            np.asarray(raw_env.sim.data.body_xpos[raw_env.can_body_id], dtype=np.float64).copy()
        )
        samples["obstacle"].append(
            np.asarray(raw_env.sim.data.body_xpos[raw_env.obstacle_body_id], dtype=np.float64).copy()
        )
        if render and (step % render.frame_skip == 0 or success or step == args.horizon):
            frame = frame_env.render(
                mode="rgb_array",
                width=render.width,
                height=render.height,
                camera_name="agentview",
            )
            frames.append(
                add_caption(
                    frame,
                    [
                        f"{condition.upper()} | RaisedMilk | guidance {'OFF' if condition == 'baseline' else 'ON'}",
                        f"step {step} | task {'SUCCESS' if success else 'running'} | collision ever {'YES' if collision else 'NO'}",
                        (
                            "rho n/a | Q n/a"
                            if condition == "baseline"
                            else f"rho {rho:g} | Q {args.safety_distance:.2f} m | tangent {args.tangent_ratio:g}"
                        ),
                    ],
                    collision,
                )
            )

    metrics = PAIRED_EVAL.rollout_condition(
        policy,
        env,
        state,
        condition,
        policy_seed,
        args,
        guidance_scale=rho,
        safety_distance=args.safety_distance if condition == "guided" else None,
        frame_callback=capture,
    )
    if samples["steps"] != list(range(len(samples["steps"]))):
        raise RuntimeError("trajectory callback missed an executed step")
    return (
        metrics,
        {key: np.asarray(value) for key, value in samples.items() if key != "steps"},
        frames,
    )


def write_paired_video(path, baseline, guided, fps):
    count = max(len(baseline), len(guided))
    if count == 0:
        raise RuntimeError("no rendered frames were captured")
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for index in range(count):
            left = baseline[min(index, len(baseline) - 1)]
            right = guided[min(index, len(guided) - 1)]
            writer.append_data(np.concatenate((left, right), axis=1))


def plot_episode(path, episode, trajectories, rows, obstacle_radius):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    baseline = trajectories["baseline"]
    obstacle_xy = baseline["obstacle"][0, :2]
    circle = plt.Circle(obstacle_xy, obstacle_radius, color="tab:red", alpha=0.25, label="Milk footprint")
    axes[0].add_patch(circle)
    axes[0].plot(baseline["eef"][:, 0], baseline["eef"][:, 1], color="black", linewidth=2, label="baseline EEF")
    axes[0].plot(baseline["can"][:, 0], baseline["can"][:, 1], color="gray", linewidth=1.3, label="baseline Can")
    for row in rows:
        key = f"rho_{row['rho']:g}"
        guided = trajectories[key]
        line = axes[0].plot(guided["eef"][:, 0], guided["eef"][:, 1], linewidth=1.3, label=key)[0]
        common = min(len(baseline["eef"]), len(guided["eef"]))
        separation = np.linalg.norm(guided["eef"][:common, :2] - baseline["eef"][:common, :2], axis=1)
        axes[1].plot(100 * separation, color=line.get_color(), linewidth=1.3, label=key)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Episode {episode}: executed XY")
    axes[0].set_xlabel("world x (m)")
    axes[0].set_ylabel("world y (m)")
    axes[1].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("EEF separation from paired baseline")
    axes[1].set_xlabel("environment step")
    axes[1].set_ylabel("XY separation (cm)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--rhos", nargs="+", type=float, required=True)
    parser.add_argument("--safety-distance", type=float, default=0.04)
    parser.add_argument("--smoothing", type=float, default=0.01)
    parser.add_argument("--tangent-ratio", type=float, default=0.0)
    parser.add_argument(
        "--guidance-phase",
        choices=("always", "target_lifted"),
        default="always",
    )
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--video-width", type=int, default=480)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    cli = parser.parse_args()
    if cli.out_dir.exists():
        raise FileExistsError(f"output already exists: {cli.out_dir}")
    if cli.tangent_ratio < 0:
        raise ValueError("tangent ratio must be non-negative")
    cli.out_dir.mkdir(parents=True)
    bank = PAIRED_EVAL.load_state_bank(cli.state_bank)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = paper_guided_policy_from_checkpoint(
        ckpt_path=str(cli.checkpoint.resolve()), device=device
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=ENV_NAME,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    run_args = SimpleNamespace(
        horizon=cli.horizon,
        terminate_on_success=True,
        terminate_on_collision=False,
        target_object_name="Can",
        xy_only=True,
        disable_hand_collision=False,
        obstacle_cost_type="huber_hinge",
        obstacle_cost_smoothing=cli.smoothing,
        obstacle_cost_smoothing_ratio=None,
        tangent_ratio=cli.tangent_ratio,
        safety_distance=cli.safety_distance,
        guidance_phase=cli.guidance_phase,
    )
    rows = []
    render = (
        SimpleNamespace(
            frame_skip=cli.frame_skip,
            width=cli.video_width,
            height=cli.video_height,
        )
        if cli.render_video
        else None
    )
    try:
        raw_env = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
        obstacle_radius = float(raw_env.obstacle.horizontal_radius)
        for episode in cli.episodes:
            key = f"{ENV_NAME}:{cli.seed}:{episode}"
            record = bank["states"][key]
            if PAIRED_EVAL.state_hash(record["state"]) != record["state_hash"]:
                raise RuntimeError(f"state hash mismatch for {key}")
            policy_seed = cli.seed * 1_000_000 + episode
            baseline_metrics, baseline, baseline_frames = record_rollout(
                policy, env, record["state"], policy_seed, run_args, "baseline", 0.0, render
            )
            trajectories = {"baseline": baseline}
            episode_rows = []
            for rho in cli.rhos:
                guided_metrics, guided, guided_frames = record_rollout(
                    policy, env, record["state"], policy_seed, run_args, "guided", rho, render
                )
                separation = trajectory_separation(baseline["eef"], guided["eef"])
                row = {
                    "episode": episode,
                    "rho": rho,
                    "tangent_ratio": cli.tangent_ratio,
                    "policy_seed": policy_seed,
                    "state_hash": record["state_hash"],
                    "baseline": compact_metrics(baseline_metrics),
                    "guided": compact_metrics(guided_metrics),
                    "separation": separation,
                }
                rows.append(row)
                episode_rows.append(row)
                trajectories[f"rho_{rho:g}"] = guided
                if render is not None:
                    write_paired_video(
                        cli.out_dir / f"episode_{episode:03d}_rho_{rho:g}_paired.mp4",
                        baseline_frames,
                        guided_frames,
                        cli.video_fps,
                    )
                print(
                    f"episode={episode} rho={rho:g} trigger={guided_metrics['guidance_trigger_rate']:.1%} "
                    f"max_xy={100 * separation['max_xy_separation_m']:.2f}cm "
                    f"task={guided_metrics['success']} collision={guided_metrics['collision_any']}",
                    flush=True,
                )
            arrays = {}
            for condition, values in trajectories.items():
                for name, value in values.items():
                    arrays[f"{condition}_{name}"] = value
            np.savez_compressed(cli.out_dir / f"episode_{episode:03d}_trajectories.npz", **arrays)
            plot_episode(
                cli.out_dir / f"episode_{episode:03d}_trajectory.png",
                episode,
                trajectories,
                episode_rows,
                obstacle_radius,
            )
    finally:
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env).close()
    (cli.out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    manifest = {
        "checkpoint": str(cli.checkpoint.resolve()),
        "state_bank": str(cli.state_bank.resolve()),
        "environment": ENV_NAME,
        "seed": cli.seed,
        "episodes": cli.episodes,
        "rhos": cli.rhos,
        "horizon": cli.horizon,
        "xy_only": True,
        "safety_distance_m": cli.safety_distance,
        "cost": "huber_hinge",
        "smoothing_m": cli.smoothing,
        "tangent_ratio": cli.tangent_ratio,
        "hand_collision_enabled": True,
        "guidance_phase": cli.guidance_phase,
    }
    (cli.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
