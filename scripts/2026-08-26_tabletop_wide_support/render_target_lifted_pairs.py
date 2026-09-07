"""Render matched baseline / target-lifted guidance RaisedMilk rollouts."""

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
from robomimic.utils.paper_lan_guidance_utils import paper_guided_policy_from_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = (
    SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
)
SPEC = importlib.util.spec_from_file_location("lan_paper_paired_eval", PAIRED_EVAL_PATH)
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)


ENV_NAME = "PickPlaceCanTabletopRaisedMilk"


def read_jsonl(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {row["episode"]: row for row in rows}


def outcome(metrics):
    if metrics["safe_success"]:
        return "SAFE SUCCESS"
    if metrics["success_with_collision"]:
        return "SUCCESS + COLLISION"
    if metrics["collision_failure"]:
        return "COLLISION FAILURE"
    return "NO-COLLISION NON-COMPLETION"


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


def record_condition(policy, env, initial_state, condition, policy_seed, run_args, cli):
    frames = []

    def capture(frame_env, observation, step, success, collision):
        del observation
        if step % cli.frame_skip == 0 or success or collision or step == cli.horizon:
            if frames and frames[-1]["step"] == step:
                return
            frames.append(
                {
                    "step": int(step),
                    "collision": bool(collision),
                    "image": render_frame(
                        frame_env, cli.height, cli.width, cli.camera_name
                    ),
                }
            )

    metrics = PAIRED_EVAL.rollout_condition(
        policy=policy,
        env=env,
        initial_state=initial_state,
        condition=condition,
        policy_seed=policy_seed,
        args=run_args,
        guidance_scale=cli.guidance_scale if condition == "guided" else 0.0,
        safety_distance=cli.safety_distance if condition == "guided" else None,
        frame_callback=capture,
    )
    return metrics, frames


def annotate(frame, label, metrics, held):
    bar_height = 60
    array = frame["image"]
    canvas = Image.new("RGB", (array.shape[1], array.shape[0] + bar_height), "black")
    canvas.paste(Image.fromarray(array), (0, bar_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, 7),
        f"{label} | step {frame['step']:03d}" + (" | FINAL HELD" if held else ""),
        fill="white",
        font=font,
    )
    draw.text(
        (8, 27),
        f"{outcome(metrics)} | collision: {'YES' if frame['collision'] else 'NO'}",
        fill=(255, 105, 105) if frame["collision"] else (125, 255, 135),
        font=font,
    )
    if label == "TARGET-LIFTED GUIDANCE":
        detail = f"first active: {metrics['first_guidance_active_step']}"
    else:
        detail = "no guidance"
    draw.text((8, 45), detail, fill=(210, 215, 230), font=font)
    return np.asarray(canvas)


def write_video(path, baseline_frames, guided_frames, baseline_metrics, guided_metrics, fps):
    count = max(len(baseline_frames), len(guided_frames))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for index in range(count):
            left_index = min(index, len(baseline_frames) - 1)
            right_index = min(index, len(guided_frames) - 1)
            left = annotate(
                baseline_frames[left_index],
                "BASELINE",
                baseline_metrics,
                index >= len(baseline_frames),
            )
            right = annotate(
                guided_frames[right_index],
                "TARGET-LIFTED GUIDANCE",
                guided_metrics,
                index >= len(guided_frames),
            )
            writer.append_data(np.concatenate((left, right), axis=1))


def replay_check(expected, actual, initial_hash):
    return {
        "initial_state_hash": expected["initial_state_hash"] == initial_hash,
        "success": expected["success"] == actual["success"],
        "collision_any": expected["collision_any"] == actual["collision_any"],
        "safe_success": expected["safe_success"] == actual["safe_success"],
        "expected_horizon": expected["horizon"],
        "actual_horizon": actual["horizon"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--guided", type=Path, required=True)
    parser.add_argument("--bank-seed", type=int, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--guidance-scale", type=float, default=45.0)
    parser.add_argument("--safety-distance", type=float, default=0.08)
    parser.add_argument("--smoothing", type=float, default=0.01)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    cli = parser.parse_args()
    if cli.out_dir.exists():
        raise FileExistsError(f"output already exists: {cli.out_dir}")
    cli.out_dir.mkdir(parents=True)

    baseline_rows = read_jsonl(cli.baseline)
    guided_rows = read_jsonl(cli.guided)
    bank = PAIRED_EVAL.load_state_bank(cli.state_bank.resolve())
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
        tangent_ratio=0.0,
        guidance_phase="target_lifted",
    )
    rendered = []
    try:
        for episode in cli.episodes:
            key = f"{ENV_NAME}:{cli.bank_seed}:{episode}"
            record = bank["states"][key]
            initial_state = record["state"]
            initial_hash = PAIRED_EVAL.state_hash(initial_state)
            if initial_hash != record["state_hash"]:
                raise RuntimeError(f"state hash mismatch for {key}")
            policy_seed = cli.bank_seed * 1_000_000 + episode
            baseline_metrics, baseline_frames = record_condition(
                policy, env, initial_state, "baseline", policy_seed, run_args, cli
            )
            guided_metrics, guided_frames = record_condition(
                policy, env, initial_state, "guided", policy_seed, run_args, cli
            )
            video_path = cli.out_dir / f"episode_{episode:02d}_baseline_vs_target_lifted.mp4"
            write_video(
                video_path,
                baseline_frames,
                guided_frames,
                baseline_metrics,
                guided_metrics,
                cli.fps,
            )
            item = {
                "episode": episode,
                "layout_seed": cli.bank_seed * 1000 + episode,
                "policy_seed": policy_seed,
                "initial_state_hash": initial_hash,
                "video": str(video_path.resolve()),
                "baseline_expected": baseline_rows[episode],
                "baseline_replay": baseline_metrics,
                "baseline_replay_check": replay_check(
                    baseline_rows[episode], baseline_metrics, initial_hash
                ),
                "guided_expected": guided_rows[episode],
                "guided_replay": guided_metrics,
                "guided_replay_check": replay_check(
                    guided_rows[episode], guided_metrics, initial_hash
                ),
            }
            rendered.append(item)
            print(
                json.dumps(
                    {
                        "episode": episode,
                        "video": str(video_path),
                        "baseline_replay": {
                            "success": baseline_metrics["success"],
                            "collision_any": baseline_metrics["collision_any"],
                        },
                        "guided_replay": {
                            "success": guided_metrics["success"],
                            "collision_any": guided_metrics["collision_any"],
                        },
                    }
                ),
                flush=True,
            )
    finally:
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env).close()

    manifest = {
        "checkpoint": str(cli.checkpoint.resolve()),
        "state_bank": str(cli.state_bank.resolve()),
        "baseline": str(cli.baseline.resolve()),
        "guided": str(cli.guided.resolve()),
        "bank_seed": cli.bank_seed,
        "episodes": cli.episodes,
        "horizon": cli.horizon,
        "guidance": {
            "phase": "target_lifted",
            "rho": cli.guidance_scale,
            "Q_m": cli.safety_distance,
            "cost": "huber_hinge",
            "smoothing_m": cli.smoothing,
        },
        "camera_name": cli.camera_name,
        "frame_skip": cli.frame_skip,
        "fps": cli.fps,
        "rendered": rendered,
    }
    (cli.out_dir / "manifest.json").write_text(
        json.dumps(PAIRED_EVAL.rollout_common.json_safe(manifest), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
