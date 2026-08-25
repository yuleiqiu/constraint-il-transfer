"""Evaluate a trained Tabletop LAN-O3DP checkpoint on the clean source task."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


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
        if hasattr(current, "sim") and hasattr(current, "robots"):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    raise ValueError("could not find raw robosuite environment")


def evaluate(checkpoint_path, episodes, horizon, seed):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if device.type != "cuda":
        raise RuntimeError("baseline evaluation requires CUDA")
    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint_path),
        device=device,
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        render=False,
        render_offscreen=True,
    )
    raw_env = raw_env_from_wrapper(env)
    records = []
    try:
        for episode in range(episodes):
            noise_seed = seed + episode
            np.random.seed(noise_seed)
            torch.manual_seed(noise_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(noise_seed)
            policy.start_episode()
            obs = env.reset()
            success = False
            action_max_abs = 0.0
            for step in range(1, horizon + 1):
                with torch.inference_mode():
                    action = np.asarray(policy(ob=obs), dtype=np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise AssertionError(f"invalid policy action at episode {episode}, step {step}: {action}")
                action_max_abs = max(action_max_abs, float(np.max(np.abs(action))))
                obs, _, _, info = env.step(action)
                success = bool(info["is_success"]["task"])
                if success:
                    break
            can_pos = np.asarray(raw_env.sim.data.body_xpos[raw_env.can_body_id]).copy()
            record = {
                "episode": episode,
                "noise_seed": noise_seed,
                "success": success,
                "steps": step,
                "can_pos": can_pos.tolist(),
                "goal_xy": np.asarray(raw_env.goal_xy).tolist(),
                "action_max_abs": action_max_abs,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        close_wrapped_env(env)
    return {
        "checkpoint": str(checkpoint_path),
        "episodes": episodes,
        "successes": sum(item["success"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])),
        "mean_steps_success": (
            float(np.mean([item["steps"] for item in records if item["success"]]))
            if any(item["success"] for item in records)
            else None
        ),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=26000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0 or args.horizon <= 0:
        raise ValueError("--episodes and --horizon must be positive")
    result = evaluate(args.checkpoint.resolve(), args.episodes, args.horizon, args.seed)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
