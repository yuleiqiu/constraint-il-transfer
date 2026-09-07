"""Evaluate one baseline or guided condition on a fixed RaisedMilk state bank."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def summarize(rows):
    count = len(rows)
    if count == 0:
        raise ValueError("cannot summarize zero rows")
    return {
        "condition": rows[0]["condition"],
        "episodes": count,
        "task_success": int(sum(row["success"] for row in rows)),
        "task_sr": float(np.mean([row["success"] for row in rows])),
        "safe_success": int(sum(row["safe_success"] for row in rows)),
        "safe_sr": float(np.mean([row["safe_success"] for row in rows])),
        "collision_any": int(sum(row["collision_any"] for row in rows)),
        "cr": float(np.mean([row["collision_any"] for row in rows])),
        "ncr_count": int(sum(row["ncr"] for row in rows)),
        "ncr": float(np.mean([row["ncr"] for row in rows])),
        "four_way": {
            "safe_success": int(sum(row["safe_success"] for row in rows)),
            "success_with_collision": int(
                sum(row["success_with_collision"] for row in rows)
            ),
            "collision_failure": int(sum(row["collision_failure"] for row in rows)),
            "collision_free_non_completion": int(sum(row["ncr"] for row in rows)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--bank-seed", type=int, required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--condition", choices=("baseline", "guided"), required=True)
    parser.add_argument(
        "--guidance-phase",
        choices=("always", "target_lifted"),
        default="always",
    )
    parser.add_argument("--guidance-scale", type=float, default=45.0)
    parser.add_argument("--safety-distance", type=float, default=0.08)
    parser.add_argument("--smoothing", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, required=True)
    cli = parser.parse_args()
    if cli.episode_start < 0:
        raise ValueError("episode-start must be non-negative")
    if cli.episodes <= 0 or cli.horizon <= 0:
        raise ValueError("episodes and horizon must be positive")
    if cli.out_dir.exists():
        raise FileExistsError(f"output already exists: {cli.out_dir}")
    cli.out_dir.mkdir(parents=True)

    bank = PAIRED_EVAL.load_state_bank(cli.state_bank.resolve())
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if device.type != "cuda":
        raise RuntimeError("RaisedMilk evaluation requires CUDA")
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
    args = SimpleNamespace(
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
        guidance_phase=cli.guidance_phase,
    )
    rows = []
    try:
        with (cli.out_dir / f"{cli.condition}.jsonl").open("w") as stream:
            for episode in range(cli.episode_start, cli.episode_start + cli.episodes):
                key = f"{ENV_NAME}:{cli.bank_seed}:{episode}"
                if key not in bank["states"]:
                    raise KeyError(f"state bank does not contain {key}")
                record = bank["states"][key]
                expected_layout_seed = cli.bank_seed * 1000 + episode
                if record["layout_seed"] != expected_layout_seed:
                    raise ValueError(f"layout seed mismatch for {key}")
                initial_state = record["state"]
                initial_hash = PAIRED_EVAL.state_hash(initial_state)
                if initial_hash != record["state_hash"]:
                    raise ValueError(f"saved state hash mismatch for {key}")
                env.reset_to(initial_state)
                restored_hash = PAIRED_EVAL.state_hash(env.get_state())
                if restored_hash != initial_hash:
                    raise RuntimeError(f"restored state hash mismatch for {key}")

                policy_seed = cli.bank_seed * 1_000_000 + episode
                metrics = PAIRED_EVAL.rollout_condition(
                    policy=policy,
                    env=env,
                    initial_state=initial_state,
                    condition=cli.condition,
                    policy_seed=policy_seed,
                    args=args,
                    guidance_scale=(
                        cli.guidance_scale if cli.condition == "guided" else 0.0
                    ),
                    safety_distance=(
                        cli.safety_distance if cli.condition == "guided" else None
                    ),
                )
                row = {
                    "episode": episode,
                    "layout_seed": expected_layout_seed,
                    "policy_seed": policy_seed,
                    "initial_state_hash": initial_hash,
                    "restored_state_hash": restored_hash,
                    **metrics,
                }
                rows.append(row)
                stream.write(json.dumps(PAIRED_EVAL.rollout_common.json_safe(row)) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            "condition": cli.condition,
                            "episode": episode,
                            "success": row["success"],
                            "safe_success": row["safe_success"],
                            "collision_any": row["collision_any"],
                            "ncr": row["ncr"],
                            "horizon": row["horizon"],
                            "initial_state_hash": initial_hash,
                        }
                    ),
                    flush=True,
                )
    finally:
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env).close()

    summary = summarize(rows)
    (cli.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "checkpoint": str(cli.checkpoint.resolve()),
        "environment": ENV_NAME,
        "state_bank": str(cli.state_bank.resolve()),
        "bank_seed": cli.bank_seed,
        "episode_start": cli.episode_start,
        "episodes": cli.episodes,
        "horizon": cli.horizon,
        "condition": cli.condition,
        "guidance": (
            {
                "rho": cli.guidance_scale,
                "Q_m": cli.safety_distance,
                "cost": "huber_hinge",
                "smoothing_m": cli.smoothing,
                "xy_only": True,
                "tangent_ratio": 0.0,
                "phase": cli.guidance_phase,
            }
            if cli.condition == "guided"
            else None
        ),
    }
    (cli.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
