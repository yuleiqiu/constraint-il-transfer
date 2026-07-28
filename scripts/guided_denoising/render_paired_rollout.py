"""Replay matched baseline / guided episodes and render synchronized videos."""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from robosuite.utils import camera_utils as CameraUtils

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import common  # noqa: E402
import paired_rollout_eval  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument(
        "--pair",
        nargs=2,
        type=int,
        action="append",
        metavar=("SEED", "EPISODE"),
        required=True,
    )
    parser.add_argument(
        "--warmup-layout",
        nargs=2,
        type=int,
        action="append",
        metavar=("SEED", "EPISODE"),
        default=[],
        help="Advance environment reset/model state without running or rendering a pair.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument(
        "--pointcloud",
        action="store_true",
        help="Add a LAN Fig. 6-style agentview projection of the sampled point cloud.",
    )
    parser.add_argument(
        "--require-recorded-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require state hash and task/collision outcomes to match the recorded pilot.",
    )
    return parser.parse_args()


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def outcome(metrics):
    if metrics["safe_success"]:
        return "SAFE SUCCESS"
    if metrics["success_with_collision"]:
        return "SUCCESS + COLLISION"
    if metrics["collision_failure"]:
        return "COLLISION FAILURE"
    return "NO-COLLISION FAILURE"


def render_frame(env, height, width, camera_name):
    return np.asarray(
        env.render(
            mode="rgb_array",
            height=height,
            width=width,
            camera_name=camera_name,
        ),
        dtype=np.uint8,
    )


def pointcloud_history(obs):
    value = np.asarray(obs["task_pointcloud"], dtype=np.float32)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(
            "Expected task_pointcloud with shape (T,N,3), got {}".format(value.shape)
        )
    return value.copy()


def project_world_points(env, points, height, width, camera_name):
    raw_env = common.raw_env_from_wrapper(env)
    intrinsic = CameraUtils.get_camera_intrinsic_matrix(
        raw_env.sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    camera_pose = CameraUtils.get_camera_extrinsic_matrix(
        raw_env.sim, camera_name=camera_name
    )
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1))], axis=1
    )
    camera_points = homogeneous @ np.linalg.inv(camera_pose).T
    depth = camera_points[:, 2]
    columns = intrinsic[0, 0] * camera_points[:, 0] / depth + intrinsic[0, 2]
    rows = intrinsic[1, 1] * camera_points[:, 1] / depth + intrinsic[1, 2]
    pixels = np.stack([rows, columns], axis=1)
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (depth > 0)
        & (rows >= 0)
        & (rows < height)
        & (columns >= 0)
        & (columns < width)
    )
    return pixels[valid], points[valid, 2]


def annotate(frame, condition, step, collision, final_outcome, held):
    bar_height = 44
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + bar_height), "black")
    image.paste(Image.fromarray(frame), (0, bar_height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = "{} | step {:03d}{}".format(
        condition.upper(), step, " | FINAL FRAME HELD" if held else ""
    )
    status = "{} | collision so far: {}".format(
        final_outcome, "YES" if collision else "NO"
    )
    draw.text((8, 6), title, fill="white", font=font)
    draw.text(
        (8, 23),
        status,
        fill=(255, 110, 110) if collision else (130, 255, 130),
        font=font,
    )
    return np.asarray(image)


def height_color(value):
    value = float(np.clip(value, 0.0, 1.0))
    anchors = (
        (0.00, (45, 65, 255)),
        (0.25, (35, 220, 235)),
        (0.50, (55, 230, 75)),
        (0.75, (255, 225, 40)),
        (1.00, (245, 45, 45)),
    )
    for (left_x, left_color), (right_x, right_color) in zip(anchors, anchors[1:]):
        if value <= right_x:
            weight = (value - left_x) / (right_x - left_x)
            return tuple(
                int(round((1.0 - weight) * left + weight * right))
                for left, right in zip(left_color, right_color)
            )
    return anchors[-1][1]


def camera_pointcloud_panel(frame, z_range):
    bar_height = 44
    rgb = Image.fromarray(frame["image"]).convert("L").convert("RGB")
    neutral = Image.new("RGB", rgb.size, (155, 155, 155))
    rgb = Image.blend(neutral, rgb, 0.32)
    image = Image.new("RGB", (rgb.width, rgb.height + bar_height), "black")
    image.paste(rgb, (0, bar_height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    query = frame["step"] % 8 == 0
    draw.text(
        (8, 7),
        "AGENTVIEW | PROJECTED 512-POINT CLOUD",
        fill=(255, 220, 80) if query else "white",
        font=font,
    )
    draw.text(
        (8, 23),
        "LATEST OF 2 FRAMES | Z COLOR{}".format(" | QUERY" if query else ""),
        fill=(215, 220, 230),
        font=font,
    )
    z_min, z_max = z_range
    z_span = max(z_max - z_min, 1e-6)
    for (row, column), world_z in zip(
        frame["projected_pixels"], frame["projected_world_z"]
    ):
        color = height_color((world_z - z_min) / z_span)
        x = int(round(column))
        y = int(round(row)) + bar_height
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    return np.asarray(image)


def record_condition(policy, env, initial_state, condition, policy_seed, run_args, args):
    frames = []

    def capture(frame_env, obs, step, success, collision):
        if step % args.frame_skip == 0 or success or step == run_args.horizon:
            history = pointcloud_history(obs)
            projected_pixels, projected_world_z = project_world_points(
                frame_env,
                history[-1],
                args.height,
                args.width,
                args.camera_name,
            )
            frames.append(
                {
                    "step": step,
                    "collision": bool(collision),
                    "image": render_frame(
                        frame_env, args.height, args.width, args.camera_name
                    ),
                    "pointcloud": history,
                    "projected_pixels": projected_pixels,
                    "projected_world_z": projected_world_z,
                }
            )

    metrics = paired_rollout_eval.rollout_condition(
        policy=policy,
        env=env,
        initial_state=initial_state,
        condition=condition,
        guidance_scale=run_args.guidance_scale,
        policy_seed=policy_seed,
        args=run_args,
        frame_callback=capture,
    )
    return metrics, frames


def write_side_by_side(
    path,
    baseline,
    guided,
    baseline_metrics,
    guided_metrics,
    fps,
    include_pointcloud=False,
):
    frame_count = max(len(baseline), len(guided))
    z_range = None
    if include_pointcloud:
        world_z = np.concatenate(
            [
                frame["projected_world_z"]
                for frames in (baseline, guided)
                for frame in frames
            ]
        )
        z_range = (float(world_z.min()), float(world_z.max()))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for index in range(frame_count):
            baseline_index = min(index, len(baseline) - 1)
            guided_index = min(index, len(guided) - 1)
            baseline_frame = baseline[baseline_index]
            guided_frame = guided[guided_index]
            left = annotate(
                baseline_frame["image"],
                "baseline",
                baseline_frame["step"],
                baseline_frame["collision"],
                outcome(baseline_metrics),
                index >= len(baseline),
            )
            right = annotate(
                guided_frame["image"],
                "guided",
                guided_frame["step"],
                guided_frame["collision"],
                outcome(guided_metrics),
                index >= len(guided),
            )
            if include_pointcloud:
                left = np.concatenate(
                    [left, camera_pointcloud_panel(baseline_frame, z_range)],
                    axis=0,
                )
                right = np.concatenate(
                    [right, camera_pointcloud_panel(guided_frame, z_range)],
                    axis=0,
                )
            writer.append_data(np.concatenate([left, right], axis=1))
    return z_range


def main():
    args = parse_args()
    if args.frame_skip <= 0:
        raise ValueError("--frame-skip must be positive")
    manifest = json.loads((args.eval_dir / "manifest.json").read_text())
    rows = load_rows(args.eval_dir / "episode_metrics.jsonl")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt = common.load_guided_policy_and_checkpoint(manifest["agent"], device)
    env_name = manifest["envs"][0]
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt,
        env_name=env_name,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    run_args = SimpleNamespace(
        horizon=manifest["horizon"],
        clearance_margin=manifest["clearance_margin"],
        cost_type=manifest["cost_type"],
        max_guidance_timestep=manifest["max_guidance_timestep"],
        max_waypoint_displacement_m=manifest["max_waypoint_displacement_m"],
        target_object_name=args.target_object_name,
        terminate_on_success=manifest["terminate_on_success"],
        guidance_scale=manifest["guidance_scale"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "agent": manifest["agent"],
        "environment": env_name,
        "camera_name": args.camera_name,
        "fps": args.fps,
        "frame_skip": args.frame_skip,
        "pointcloud": args.pointcloud,
        "warmup_layouts": args.warmup_layout,
        "require_recorded_replay": args.require_recorded_replay,
        "pairs": [],
    }

    for seed, episode in args.warmup_layout:
        layout_seed = seed * 1000 + episode
        random.seed(layout_seed)
        np.random.seed(layout_seed)
        env.reset()

    for seed, episode in args.pair:
        expected = {
            row["condition"]: row
            for row in rows
            if row["seed"] == seed and row["episode"] == episode
        }
        if set(expected) != {"baseline", "guided"}:
            raise ValueError("Pair seed={} episode={} is absent or incomplete".format(seed, episode))
        if (
            expected["baseline"]["initial_state_hash"]
            != expected["guided"]["initial_state_hash"]
        ):
            raise ValueError("Recorded pair has mismatched initial-state hashes")

        layout_seed = seed * 1000 + episode
        policy_seed = seed * 1_000_000 + episode
        random.seed(layout_seed)
        np.random.seed(layout_seed)
        env.reset()
        initial_state = env.get_state()
        replay_hash = paired_rollout_eval.state_hash(initial_state)
        expected_hash = expected["baseline"]["initial_state_hash"]
        state_hash_matches_record = replay_hash == expected_hash
        if args.require_recorded_replay and not state_hash_matches_record:
            raise ValueError(
                "Replay state hash mismatch for seed={} episode={}: {} != {}".format(
                    seed, episode, replay_hash, expected_hash
                )
            )
        if not state_hash_matches_record:
            print(
                "warning: replay state hash differs from recorded pilot for "
                "seed={} episode={}".format(seed, episode),
                flush=True,
            )

        baseline_metrics, baseline_frames = record_condition(
            policy,
            env,
            initial_state,
            "baseline",
            policy_seed,
            run_args,
            args,
        )
        guided_metrics, guided_frames = record_condition(
            policy,
            env,
            initial_state,
            "guided",
            policy_seed,
            run_args,
            args,
        )
        outcome_matches_record = {}
        for condition, replay_metrics in (
            ("baseline", baseline_metrics),
            ("guided", guided_metrics),
        ):
            expected_metrics = expected[condition]
            replay_signature = (
                replay_metrics["success"],
                replay_metrics["collision_any"],
                replay_metrics["observation_failure"],
            )
            expected_signature = (
                expected_metrics["success"],
                expected_metrics["collision_any"],
                expected_metrics["observation_failure"],
            )
            outcome_matches_record[condition] = replay_signature == expected_signature
            if args.require_recorded_replay and not outcome_matches_record[condition]:
                raise ValueError(
                    "{} outcome mismatch for seed={} episode={}: {} != {}".format(
                        condition, seed, episode, replay_signature, expected_signature
                    )
                )

        suffix = "_camera_pointcloud" if args.pointcloud else ""
        video_path = args.out_dir / "seed{}_episode{}_baseline_vs_guided{}.mp4".format(
            seed, episode, suffix
        )
        z_range = write_side_by_side(
            video_path,
            baseline_frames,
            guided_frames,
            baseline_metrics,
            guided_metrics,
            args.fps,
            include_pointcloud=args.pointcloud,
        )
        pair_metadata = {
            "seed": seed,
            "episode": episode,
            "layout_seed": layout_seed,
            "policy_seed": policy_seed,
            "shared_replay_state_hash": replay_hash,
            "recorded_pilot_state_hash": expected_hash,
            "state_hash_matches_recorded_pilot": state_hash_matches_record,
            "baseline_and_guided_share_replay_state": True,
            "outcome_matches_recorded_pilot": outcome_matches_record,
            "baseline": common.json_safe(baseline_metrics),
            "guided": common.json_safe(guided_metrics),
            "video": video_path.name,
        }
        if z_range is not None:
            pair_metadata["pointcloud_world_z_range"] = list(z_range)
            pair_metadata["pointcloud_projection"] = (
                "Exact latest-frame 512-point task_pointcloud projected into "
                "the agentview image with the runtime camera calibration."
            )
        metadata["pairs"].append(pair_metadata)
        print("wrote {}".format(video_path), flush=True)

    metadata_name = (
        "camera_pointcloud_metadata.json" if args.pointcloud else "metadata.json"
    )
    (args.out_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("wrote {}".format(args.out_dir / metadata_name), flush=True)


if __name__ == "__main__":
    main()
