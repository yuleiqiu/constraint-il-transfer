"""Pair Tabletop source checkpoints on an exact saved initial-state bank."""

import argparse
import gc
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = (
    SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
)
SPEC = importlib.util.spec_from_file_location("lan_paper_paired_eval", PAIRED_EVAL_PATH)
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)


def parse_checkpoint(value):
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH") from exc
    if not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label, Path(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def close_wrapped_env(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        close = getattr(current, "close", None)
        if callable(close):
            close()
            return
        current = getattr(current, "env", None)


def evaluate_checkpoint(
    label,
    checkpoint_path,
    state_bank,
    env_name,
    bank_seed,
    episodes,
    horizon,
    out_dir,
):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if device.type != "cuda":
        raise RuntimeError("fixed-state checkpoint screening requires CUDA")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint_path),
        device=device,
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=env_name,
        render=False,
        render_offscreen=True,
    )
    rows = []
    jsonl_path = out_dir / f"{label}.jsonl"
    try:
        with jsonl_path.open("w") as stream:
            for episode in range(episodes):
                key = f"{env_name}:{bank_seed}:{episode}"
                if key not in state_bank["states"]:
                    raise KeyError(f"state bank does not contain {key}")
                saved = state_bank["states"][key]
                expected_layout_seed = bank_seed * 1000 + episode
                if saved["layout_seed"] != expected_layout_seed:
                    raise ValueError(f"layout seed mismatch for {key}")
                initial_state = saved["state"]
                initial_hash = PAIRED_EVAL.state_hash(initial_state)
                if initial_hash != saved["state_hash"]:
                    raise ValueError(f"saved state hash mismatch for {key}")

                policy_seed = bank_seed * 1_000_000 + episode
                seed_everything(policy_seed)
                policy.start_episode()
                obs = env.reset_to(initial_state)
                restored_hash = PAIRED_EVAL.state_hash(env.get_state())
                if restored_hash != initial_hash:
                    raise RuntimeError(
                        f"restored state hash mismatch for {key}: "
                        f"{restored_hash} != {initial_hash}"
                    )

                success = False
                action_max_abs = 0.0
                for step in range(1, horizon + 1):
                    with torch.inference_mode():
                        action = np.asarray(policy(ob=obs), dtype=np.float32)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise AssertionError(
                            f"invalid action for {label}, episode {episode}, step {step}: {action}"
                        )
                    action_max_abs = max(action_max_abs, float(np.max(np.abs(action))))
                    obs, _, _, info = env.step(action)
                    success = bool(info["is_success"]["task"])
                    if success:
                        break

                row = {
                    "checkpoint": label,
                    "episode": episode,
                    "layout_seed": expected_layout_seed,
                    "policy_seed": policy_seed,
                    "initial_state_hash": initial_hash,
                    "restored_state_hash": restored_hash,
                    "success": success,
                    "steps": step,
                    "action_max_abs": action_max_abs,
                }
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
    finally:
        close_wrapped_env(env)
        del env, policy, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "successes": int(sum(row["success"] for row in rows)),
        "episodes": episodes,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps_success": (
            float(np.mean([row["steps"] for row in rows if row["success"]]))
            if any(row["success"] for row in rows)
            else None
        ),
        "records": rows,
    }


def paired_summary(left, right):
    left_by_episode = {row["episode"]: row for row in left["records"]}
    right_by_episode = {row["episode"]: row for row in right["records"]}
    if left_by_episode.keys() != right_by_episode.keys():
        raise ValueError("checkpoint episode sets differ")
    both_success = 0
    left_only = 0
    right_only = 0
    both_failure = 0
    for episode in sorted(left_by_episode):
        lhs = left_by_episode[episode]
        rhs = right_by_episode[episode]
        if (
            lhs["initial_state_hash"] != rhs["initial_state_hash"]
            or lhs["policy_seed"] != rhs["policy_seed"]
        ):
            raise ValueError(f"pairing mismatch at episode {episode}")
        if lhs["success"] and rhs["success"]:
            both_success += 1
        elif lhs["success"]:
            left_only += 1
        elif rhs["success"]:
            right_only += 1
        else:
            both_failure += 1
    return {
        "left": left["label"],
        "right": right["label"],
        "both_success": both_success,
        "left_only_success": left_only,
        "right_only_success": right_only,
        "both_failure": both_failure,
        "right_failure_to_success": right_only,
        "right_success_to_failure": left_only,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--bank-seed", type=int, required=True)
    parser.add_argument("--env", default="PickPlaceCanTabletop")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        help="Provide one or two times as LABEL=PATH",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoint) not in (1, 2):
        raise ValueError("one or two --checkpoint arguments are required")
    if args.episodes <= 0 or args.horizon <= 0:
        raise ValueError("episodes and horizon must be positive")
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    bank = PAIRED_EVAL.load_state_bank(args.state_bank.resolve())
    results = []
    for label, path in args.checkpoint:
        results.append(
            evaluate_checkpoint(
                label=label,
                checkpoint_path=path,
                state_bank=bank,
                env_name=args.env,
                bank_seed=args.bank_seed,
                episodes=args.episodes,
                horizon=args.horizon,
                out_dir=args.out_dir,
            )
        )
    summary = {
        "environment": args.env,
        "state_bank": str(args.state_bank.resolve()),
        "bank_seed": args.bank_seed,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "paired_initial_states": True,
        "paired_policy_seeds": True,
        "checkpoints": [
            {key: value for key, value in result.items() if key != "records"}
            for result in results
        ],
        "paired_outcomes": (
            paired_summary(results[0], results[1]) if len(results) == 2 else None
        ),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
