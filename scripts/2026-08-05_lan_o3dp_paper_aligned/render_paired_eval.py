"""Render exact-state baseline versus paper-guided LAN-O3DP rollouts."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.paper_lan_guidance_utils import (
    paper_guided_policy_from_checkpoint,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = SCRIPT_DIR / "paired_eval.py"
SPEC = importlib.util.spec_from_file_location("paper_lan_paired_eval", PAIRED_EVAL_PATH)
paired_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paired_eval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--safety-distance", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def outcome(metrics):
    if metrics["safe_success"]:
        return "SAFE SUCCESS"
    if metrics["success_with_collision"]:
        return "SUCCESS + COLLISION"
    if metrics.get("horizon_truncated", False):
        if metrics["collision_any"]:
            return "HORIZON TRUNCATED + COLLISION"
        return "HORIZON TRUNCATED"
    if metrics["collision_failure"]:
        return "COLLISION FAILURE"
    return "NO-COLLISION FAILURE"


def render_frame(env, args):
    return np.asarray(
        env.render(
            mode="rgb_array",
            height=args.height,
            width=args.width,
            camera_name=args.camera_name,
        ),
        dtype=np.uint8,
    )


def record_condition(
    policy,
    env,
    initial_state,
    condition,
    policy_seed,
    run_args,
    guidance_scale,
    safety_distance,
    args,
):
    frames = []

    def capture(frame_env, observation, step, success, collision):
        del observation
        if step % args.frame_skip == 0 or success or collision or step == run_args.horizon:
            if frames and frames[-1]["step"] == step:
                return
            frames.append(
                {
                    "step": int(step),
                    "collision": bool(collision),
                    "image": render_frame(frame_env, args),
                }
            )

    metrics = paired_eval.rollout_condition(
        policy,
        env,
        initial_state,
        condition,
        policy_seed,
        run_args,
        guidance_scale=guidance_scale,
        safety_distance=safety_distance,
        frame_callback=capture,
    )
    return metrics, frames


def annotate(frame, label, metrics, held=False):
    bar_height = 62
    image = Image.new("RGB", (frame["image"].shape[1], frame["image"].shape[0] + bar_height), "black")
    image.paste(Image.fromarray(frame["image"]), (0, bar_height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (8, 6),
        "{} | step {:03d}{}".format(
            label.upper(), frame["step"], " | FINAL HELD" if held else ""
        ),
        fill="white",
        font=font,
    )
    draw.text(
        (8, 24),
        "{} | collision: {}".format(
            outcome(metrics), "YES" if frame["collision"] else "NO"
        ),
        fill=(255, 105, 105) if frame["collision"] else (125, 255, 135),
        font=font,
    )
    draw.text(
        (8, 42),
        "trigger={:.1%} | update_max={:.2e}".format(
            metrics["guidance_trigger_rate"], metrics["guidance_update_norm_max"]
        ),
        fill=(210, 215, 230),
        font=font,
    )
    return np.asarray(image)


def write_side_by_side(path, baseline_frames, guided_frames, baseline_metrics, guided_metrics, fps):
    frame_count = max(len(baseline_frames), len(guided_frames))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for index in range(frame_count):
            left_index = min(index, len(baseline_frames) - 1)
            right_index = min(index, len(guided_frames) - 1)
            left = annotate(
                baseline_frames[left_index],
                "baseline",
                baseline_metrics,
                held=index >= len(baseline_frames),
            )
            right = annotate(
                guided_frames[right_index],
                "guided",
                guided_metrics,
                held=index >= len(guided_frames),
            )
            writer.append_data(np.concatenate([left, right], axis=1))


def expected_row(rows, seed, episode, condition, guidance_scale=None, safety_distance=None):
    selected = [
        row
        for row in rows
        if row["seed"] == seed
        and row["episode"] == episode
        and row["condition"] == condition
        and (
            condition == "baseline"
            or (
                row["guidance_scale"] == guidance_scale
                and row["safety_distance"] == safety_distance
            )
        )
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one recorded row for seed={} episode={} condition={}, got {}".format(
                seed, episode, condition, len(selected)
            )
        )
    return selected[0]


def verify_replay(expected, actual, initial_hash):
    if expected["initial_state_hash"] != initial_hash:
        raise RuntimeError("Replay initial-state hash does not match recorded evaluation")
    for key in ("success", "collision_any"):
        if expected[key] != actual[key]:
            raise RuntimeError(
                "Replay mismatch for {}: expected {}, got {}".format(
                    key, expected[key], actual[key]
                )
            )
    if expected["horizon"] != actual["horizon"]:
        print(
            "Replay horizon differs while outcome matches: expected {}, got {}".format(
                expected["horizon"], actual["horizon"]
            ),
            flush=True,
        )


def main():
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Output already exists: {}".format(args.out_dir))
    if args.frame_skip <= 0:
        raise ValueError("--frame-skip must be positive")
    args.out_dir.mkdir(parents=True)

    manifest = json.loads((args.eval_dir / "manifest.json").read_text())
    rows = load_rows(args.eval_dir / "episode_metrics.jsonl")
    if len(manifest["envs"]) != 1:
        raise ValueError("Video rendering expects exactly one environment")
    env_name = manifest["envs"][0]
    seed = args.seed if args.seed is not None else manifest["seeds"][0]
    state_bank_path = Path(manifest["state_bank_source"])
    state_bank = paired_eval.load_state_bank(state_bank_path)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = paper_guided_policy_from_checkpoint(
        ckpt_path=manifest["agent"],
        device=device,
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=env_name,
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
    )
    rendered = []
    try:
        for episode in args.episodes:
            state_key = "{}:{}:{}".format(env_name, seed, episode)
            record = state_bank["states"][state_key]
            initial_state = record["state"]
            initial_hash = paired_eval.state_hash(initial_state)
            if initial_hash != record["state_hash"]:
                raise RuntimeError("State-bank hash mismatch for {}".format(state_key))
            policy_seed = seed * 1_000_000 + episode

            baseline_metrics, baseline_frames = record_condition(
                policy,
                env,
                initial_state,
                "baseline",
                policy_seed,
                run_args,
                0.0,
                None,
                args,
            )
            guided_metrics, guided_frames = record_condition(
                policy,
                env,
                initial_state,
                "guided",
                policy_seed,
                run_args,
                args.guidance_scale,
                args.safety_distance,
                args,
            )
            verify_replay(
                expected_row(rows, seed, episode, "baseline"),
                baseline_metrics,
                initial_hash,
            )
            verify_replay(
                expected_row(
                    rows,
                    seed,
                    episode,
                    "guided",
                    args.guidance_scale,
                    args.safety_distance,
                ),
                guided_metrics,
                initial_hash,
            )

            video_path = args.out_dir / "episode_{:03d}_baseline_vs_guided.mp4".format(
                episode
            )
            write_side_by_side(
                video_path,
                baseline_frames,
                guided_frames,
                baseline_metrics,
                guided_metrics,
                args.fps,
            )
            rendered.append(
                {
                    "episode": episode,
                    "initial_state_hash": initial_hash,
                    "video": str(video_path.resolve()),
                    "baseline": baseline_metrics,
                    "guided": guided_metrics,
                }
            )
            print("Wrote {}".format(video_path), flush=True)
    finally:
        paired_eval.rollout_common.raw_env_from_wrapper(env).close()

    video_manifest = {
        "eval_dir": str(args.eval_dir.resolve()),
        "state_bank": str(state_bank_path.resolve()),
        "seed": seed,
        "guidance_scale": args.guidance_scale,
        "safety_distance": args.safety_distance,
        "camera_name": args.camera_name,
        "frame_skip": args.frame_skip,
        "fps": args.fps,
        "rendered": rendered,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(paired_eval.rollout_common.json_safe(video_manifest), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
