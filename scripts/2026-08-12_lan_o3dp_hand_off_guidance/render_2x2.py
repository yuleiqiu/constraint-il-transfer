"""Render exact-state full-physics/hand-off x baseline/guided comparisons."""

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
PAIRED_EVAL_PATH = (
    SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
)
SPEC = importlib.util.spec_from_file_location("paper_lan_paired_eval", PAIRED_EVAL_PATH)
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-eval-dir", type=Path, required=True)
    parser.add_argument("--hand-off-eval-dir", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def expected_row(rows, seed, episode, condition):
    selected = [
        row
        for row in rows
        if row["seed"] == seed
        and row["episode"] == episode
        and row["condition"] == condition
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one row for seed={} episode={} condition={}, got {}".format(
                seed, episode, condition, len(selected)
            )
        )
    return selected[0]


def outcome(metrics):
    if metrics["safe_success"]:
        return "SAFE SUCCESS"
    if metrics["success_with_collision"]:
        return "SUCCESS + COLLISION"
    if metrics["collision_failure"]:
        return "FAILURE + COLLISION"
    return "FAILURE, NO COLLISION"


def contact_label(metrics):
    active = []
    for name, key in (
        ("hand", "hand_obstacle_collision_any"),
        ("finger", "finger_obstacle_collision_any"),
        ("arm", "arm_obstacle_collision_any"),
        ("target", "target_obstacle_collision_any"),
    ):
        if metrics.get(key, False):
            active.append(name)
    return ", ".join(active) if active else "none"


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
    policy_seed,
    manifest,
    *,
    condition,
    hand_off,
    args,
):
    frames = []

    def capture(frame_env, observation, step, success, collision):
        del observation
        if step % args.frame_skip == 0 or success or collision or step == manifest["horizon"]:
            if frames and frames[-1]["step"] == step:
                return
            frames.append(
                {
                    "step": int(step),
                    "collision": bool(collision),
                    "image": render_frame(frame_env, args),
                }
            )

    run_args = SimpleNamespace(
        horizon=manifest["horizon"],
        terminate_on_success=True,
        terminate_on_collision=False,
        target_object_name="Can",
        xy_only=manifest["xy_only"],
        disable_hand_collision=hand_off,
        obstacle_cost_type=manifest["obstacle_cost_type"],
        obstacle_cost_smoothing=manifest["obstacle_cost_smoothing"],
        obstacle_cost_smoothing_ratio=manifest["obstacle_cost_smoothing_ratio"],
    )
    guided = condition == "guided"
    metrics = PAIRED_EVAL.rollout_condition(
        policy,
        env,
        initial_state,
        condition,
        policy_seed,
        run_args,
        guidance_scale=manifest["guidance_scales"][0] if guided else 0.0,
        safety_distance=manifest["safety_distances"][0] if guided else None,
        frame_callback=capture,
    )
    return metrics, frames


def verify_replay(expected, actual, initial_hash):
    if expected["initial_state_hash"] != initial_hash:
        raise RuntimeError("Initial-state hash mismatch")
    mismatches = []
    for key in (
        "success",
        "collision_any",
        "hand_obstacle_collision_any",
        "finger_obstacle_collision_any",
        "arm_obstacle_collision_any",
        "target_obstacle_collision_any",
    ):
        if expected.get(key, False) != actual.get(key, False):
            mismatches.append(
                {
                    "field": key,
                    "formal": expected.get(key),
                    "replay": actual.get(key),
                }
            )
    return mismatches


def annotate(frame, label, metrics, held=False):
    bar_height = 82
    height, width = frame["image"].shape[:2]
    image = Image.new("RGB", (width, height + bar_height), "black")
    image.paste(Image.fromarray(frame["image"]), (0, bar_height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (8, 6),
        "{} | step {:03d}{}".format(label, frame["step"], " | FINAL HELD" if held else ""),
        fill="white",
        font=font,
    )
    draw.text(
        (8, 24),
        "{} | collision seen: {}".format(
            outcome(metrics), "YES" if frame["collision"] else "NO"
        ),
        fill=(255, 105, 105) if frame["collision"] else (125, 255, 135),
        font=font,
    )
    draw.text(
        (8, 42),
        "final contacts: {} | first collision: {}".format(
            contact_label(metrics), metrics.get("first_collision_step")
        ),
        fill=(225, 210, 160),
        font=font,
    )
    draw.text(
        (8, 60),
        "trigger={:.1%} | update max={:.2e}".format(
            metrics["guidance_trigger_rate"], metrics["guidance_update_norm_max"]
        ),
        fill=(210, 215, 230),
        font=font,
    )
    return np.asarray(image)


def write_grid(path, records, fps):
    frame_count = max(len(item["frames"]) for item in records)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for index in range(frame_count):
            panels = []
            for item in records:
                frames = item["frames"]
                frame_index = min(index, len(frames) - 1)
                panels.append(
                    annotate(
                        frames[frame_index],
                        item["label"],
                        item["metrics"],
                        held=index >= len(frames),
                    )
                )
            top = np.concatenate(panels[:2], axis=1)
            bottom = np.concatenate(panels[2:], axis=1)
            writer.append_data(np.concatenate((top, bottom), axis=0))


def main():
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Output already exists: {}".format(args.out_dir))
    if args.frame_skip <= 0:
        raise ValueError("frame-skip must be positive")
    args.out_dir.mkdir(parents=True)

    full_manifest = json.loads((args.full_eval_dir / "manifest.json").read_text())
    hand_manifest = json.loads((args.hand_off_eval_dir / "manifest.json").read_text())
    if full_manifest["envs"] != hand_manifest["envs"] or len(full_manifest["envs"]) != 1:
        raise ValueError("Evaluation directories must contain the same single environment")
    for key in (
        "agent",
        "seeds",
        "horizon",
        "guidance_scales",
        "safety_distances",
        "xy_only",
        "obstacle_cost_type",
        "obstacle_cost_smoothing",
    ):
        if full_manifest[key] != hand_manifest[key]:
            raise ValueError("Manifest mismatch for {}".format(key))
    if not full_manifest["hand_collision_enabled"] or hand_manifest["hand_collision_enabled"]:
        raise ValueError("The supplied physics directories are reversed or invalid")

    rows_by_physics = {
        "full": load_rows(args.full_eval_dir / "episode_metrics.jsonl"),
        "hand_off": load_rows(args.hand_off_eval_dir / "episode_metrics.jsonl"),
    }
    state_bank_path = Path(full_manifest["state_bank_source"])
    if state_bank_path.resolve() != Path(hand_manifest["state_bank_source"]).resolve():
        raise ValueError("Evaluation directories do not share a state bank")
    state_bank = PAIRED_EVAL.load_state_bank(state_bank_path)
    env_name = full_manifest["envs"][0]
    seed = full_manifest["seeds"][0]

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = paper_guided_policy_from_checkpoint(
        ckpt_path=full_manifest["agent"], device=device
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=env_name,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    rendered = []
    conditions = (
        ("full", "baseline", False, "FULL PHYSICS / BASELINE"),
        ("full", "guided", False, "FULL PHYSICS / GUIDED"),
        ("hand_off", "baseline", True, "HAND OFF / BASELINE"),
        ("hand_off", "guided", True, "HAND OFF / GUIDED"),
    )
    try:
        for episode in args.episodes:
            key = "{}:{}:{}".format(env_name, seed, episode)
            state_record = state_bank["states"][key]
            initial_state = state_record["state"]
            initial_hash = PAIRED_EVAL.state_hash(initial_state)
            if initial_hash != state_record["state_hash"]:
                raise RuntimeError("State-bank hash mismatch for {}".format(key))
            policy_seed = seed * 1_000_000 + episode
            records = []
            for physics, condition, hand_off, label in conditions:
                manifest = hand_manifest if hand_off else full_manifest
                metrics, frames = record_condition(
                    policy,
                    env,
                    initial_state,
                    policy_seed,
                    manifest,
                    condition=condition,
                    hand_off=hand_off,
                    args=args,
                )
                expected = expected_row(
                    rows_by_physics[physics], seed, episode, condition
                )
                mismatches = verify_replay(expected, metrics, initial_hash)
                if mismatches:
                    print(
                        "Replay label drift for episode {} {}: {}".format(
                            episode, label, mismatches
                        ),
                        flush=True,
                    )
                records.append(
                    {
                        "label": label,
                        "metrics": metrics,
                        "frames": frames,
                        "formal_metrics": expected,
                        "formal_replay_mismatches": mismatches,
                    }
                )
            video_path = args.out_dir / "episode_{:03d}_2x2.mp4".format(episode)
            write_grid(video_path, records, args.fps)
            rendered.append(
                {
                    "episode": episode,
                    "initial_state_hash": initial_hash,
                    "video": str(video_path.resolve()),
                    "conditions": [
                        {
                            "label": item["label"],
                            "metrics": item["metrics"],
                            "formal_metrics": item["formal_metrics"],
                            "formal_replay_mismatches": item[
                                "formal_replay_mismatches"
                            ],
                        }
                        for item in records
                    ],
                }
            )
            print("Wrote {}".format(video_path), flush=True)
    finally:
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env).close()

    output_manifest = {
        "full_eval_dir": str(args.full_eval_dir.resolve()),
        "hand_off_eval_dir": str(args.hand_off_eval_dir.resolve()),
        "state_bank": str(state_bank_path.resolve()),
        "environment": env_name,
        "seed": seed,
        "camera_name": args.camera_name,
        "frame_skip": args.frame_skip,
        "fps": args.fps,
        "replay_note": (
            "Same serialized state and policy seed as the formal evaluation, but "
            "CUDA rollout is not bitwise deterministic. Panel annotations report "
            "the actual rendered replay; formal/replay label drift is retained per "
            "condition below."
        ),
        "rendered": rendered,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(PAIRED_EVAL.rollout_common.json_safe(output_manifest), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
