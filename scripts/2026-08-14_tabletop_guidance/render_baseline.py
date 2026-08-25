"""Render fixed-seed Tabletop LAN-O3DP baseline rollouts."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


DEFAULT_EPISODES = (0, 7, 9, 23, 34)


def close_wrapped_env(env):
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        close = getattr(current, "close", None)
        if callable(close):
            close()
            return
        current = getattr(current, "env", None)
        if current is None:
            return


def raw_env_from_wrapper(env):
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "sim") and hasattr(current, "can_body_id"):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    raise ValueError("could not find raw Tabletop robosuite environment")


def add_caption(frame, lines):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    line_height = 15
    box_height = 12 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 175))
    for index, line in enumerate(lines):
        draw.text((8, 6 + index * line_height), line, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image)


def render_frame(env, *, width, height, episode, seed, step, expected, actual=None):
    frame = env.render(
        mode="rgb_array",
        width=width,
        height=height,
        camera_name="agentview",
    )
    lines = [
        f"epoch 100 | episode {episode} | diffusion seed {seed} | step {step}",
        f"formal record: {expected}",
    ]
    if actual is not None:
        lines.append(f"rendered replay: {actual}")
    return add_caption(frame, lines)


def load_formal_records(path):
    payload = json.loads(path.read_text())
    return {int(record["episode"]): record for record in payload["records"]}


def advance_unrendered_reset(env, record):
    """Advance the environment RNG exactly as the sequential formal eval did."""

    seed = int(record["noise_seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.reset()


def render_rollout(policy, env, record, args):
    episode = int(record["episode"])
    seed = int(record["noise_seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy.start_episode()
    obs = env.reset()
    expected = "SUCCESS" if record["success"] else "FAILURE"
    frames = [
        render_frame(
            env,
            width=args.width,
            height=args.height,
            episode=episode,
            seed=seed,
            step=0,
            expected=expected,
        )
    ]
    success = False
    for step in range(1, args.horizon + 1):
        with torch.inference_mode():
            action = np.asarray(policy(ob=obs), dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise AssertionError(f"invalid action at episode {episode}, step {step}: {action}")
        obs, _, _, info = env.step(action)
        success = bool(info["is_success"]["task"])
        terminal = success or step == args.horizon
        if step % args.frame_skip == 0 or terminal:
            frames.append(
                render_frame(
                    env,
                    width=args.width,
                    height=args.height,
                    episode=episode,
                    seed=seed,
                    step=step,
                    expected=expected,
                    actual=("SUCCESS" if success else "FAILURE") if terminal else None,
                )
            )
        if success:
            break

    raw_env = raw_env_from_wrapper(env)
    can_pos = np.asarray(raw_env.sim.data.body_xpos[raw_env.can_body_id]).copy()
    eef_pos = np.asarray(
        raw_env.sim.data.site_xpos[raw_env.robots[0].eef_site_id[raw_env.robots[0].arms[0]]]
    ).copy()
    eef_distance = float(np.linalg.norm(eef_pos - can_pos))
    r_reach = float(1.0 - np.tanh(10.0 * eef_distance))
    result = {
        "episode": episode,
        "noise_seed": seed,
        "formal_success": bool(record["success"]),
        "rendered_success": success,
        "steps": step,
        "can_pos": can_pos.tolist(),
        "eef_pos": eef_pos.tolist(),
        "goal_xy": np.asarray(raw_env.goal_xy).tolist(),
        "can_in_goal": bool(raw_env._can_in_goal()),
        "eef_distance_to_can": eef_distance,
        "r_reach": r_reach,
    }
    path = args.out_dir / f"episode_{episode:02d}_seed_{seed}_{'success' if success else 'failure'}.mp4"
    with imageio.get_writer(path, fps=args.fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    result["video"] = str(path)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--formal-result", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(DEFAULT_EPISODES))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.formal_result = args.formal_result.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_formal_records(args.formal_result)
    missing = sorted(set(args.episodes) - set(records))
    if missing:
        raise ValueError(f"episodes missing from formal result: {missing}")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if device.type != "cuda":
        raise RuntimeError("baseline rendering requires CUDA")
    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=device,
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        render=False,
        render_offscreen=True,
    )
    selected_episodes = sorted(set(args.episodes))
    results = []
    previous_episode = -1
    try:
        for episode in selected_episodes:
            for skipped_episode in range(previous_episode + 1, episode):
                advance_unrendered_reset(env, records[skipped_episode])
            results.append(render_rollout(policy, env, records[episode], args))
            previous_episode = episode
    finally:
        close_wrapped_env(env)
    manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
