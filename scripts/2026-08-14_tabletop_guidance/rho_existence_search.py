"""Sequential single-state rho search that stops at the first safe success."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.paper_lan_guidance_utils import paper_guided_policy_from_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
SPEC = importlib.util.spec_from_file_location("tabletop_paper_lan_paired_eval", PAIRED_EVAL_PATH)
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)
ENV_NAME = "PickPlaceCanTabletopRaisedMilk"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--rhos", nargs="+", type=float, required=True)
    parser.add_argument("--safety-distance", type=float, default=0.08)
    parser.add_argument("--smoothing", type=float, default=0.01)
    parser.add_argument("--tangent-ratio", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    if args.tangent_ratio < 0:
        raise ValueError("tangent ratio must be non-negative")
    args.out_dir.mkdir(parents=True)

    bank = PAIRED_EVAL.load_state_bank(args.state_bank)
    key = f"{ENV_NAME}:{args.seed}:{args.episode}"
    record = bank["states"][key]
    if PAIRED_EVAL.state_hash(record["state"]) != record["state_hash"]:
        raise RuntimeError(f"state hash mismatch for {key}")

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = paper_guided_policy_from_checkpoint(
        ckpt_path=str(args.checkpoint.resolve()), device=device
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=ENV_NAME,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    rollout_args = SimpleNamespace(
        horizon=args.horizon,
        terminate_on_success=True,
        terminate_on_collision=False,
        target_object_name="Can",
        xy_only=True,
        disable_hand_collision=False,
        obstacle_cost_type="huber_hinge",
        obstacle_cost_smoothing=args.smoothing,
        obstacle_cost_smoothing_ratio=None,
        tangent_ratio=args.tangent_ratio,
        guidance_phase="always",
    )
    policy_seed = args.seed * 1_000_000 + args.episode
    attempts = []
    try:
        for rho in args.rhos:
            metrics = PAIRED_EVAL.rollout_condition(
                policy,
                env,
                record["state"],
                "guided",
                policy_seed,
                rollout_args,
                guidance_scale=rho,
                safety_distance=args.safety_distance,
            )
            row = {"rho": rho, **metrics}
            attempts.append(row)
            print(
                f"rho={rho:g} task={metrics['success']} collision={metrics['collision_any']} "
                f"safe_success={metrics['safe_success']} first_active={metrics['first_guidance_active_step']} "
                f"first_collision={metrics['first_collision_step']}",
                flush=True,
            )
            if metrics["safe_success"]:
                break
    finally:
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env).close()

    result = {
        "environment": ENV_NAME,
        "seed": args.seed,
        "episode": args.episode,
        "state_hash": record["state_hash"],
        "policy_seed": policy_seed,
        "safety_distance": args.safety_distance,
        "smoothing": args.smoothing,
        "tangent_ratio": args.tangent_ratio,
        "horizon": args.horizon,
        "requested_rhos": args.rhos,
        "stopped_at_first_safe_success": True,
        "attempts": attempts,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
