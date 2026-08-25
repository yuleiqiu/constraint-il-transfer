"""Generate a fresh collision-failure Tabletop state bank from scratch.

The archived state bank is intentionally not read. This script resets the
current RaisedMilk environment, runs the full unguided checkpoint rollout, and
keeps the initial state only when that rollout is a collision failure. The
saved model XML therefore already contains this branch's Backview camera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import random

import robomimic.utils.torch_utils as TorchUtils

from ellipsoid_counterfactual_rollout import (  # noqa: E402
    CF,
    PAIRED_EVAL,
    combined_frame_callback,
)
from robomimic.utils.ellipsoid_guidance_utils import (
    ellipsoid_guided_policy_from_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layouts", type=int, default=12)
    parser.add_argument("--target-failures", type=int, default=3)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--image-height", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=160)
    args = parser.parse_args()
    if args.layouts <= 0 or args.target_failures <= 0:
        raise ValueError("layouts and target_failures must be positive")
    if args.out_dir.exists():
        raise FileExistsError("output already exists: {}".format(args.out_dir))
    args.out_dir.mkdir(parents=True)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = ellipsoid_guided_policy_from_checkpoint(
        ckpt_path=str(args.checkpoint.resolve()),
        device=device,
    )
    env_kwargs = checkpoint["env_metadata"]["env_kwargs"]
    env_kwargs["camera_names"] = ["agentview", "backview"]
    env_kwargs["camera_heights"] = args.image_height
    env_kwargs["camera_widths"] = args.image_width
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["has_renderer"] = False
    env, _ = PAIRED_EVAL.FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=CF.OBSTACLE_ENV,
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
        obstacle_cost_smoothing=0.01,
        obstacle_cost_smoothing_ratio=None,
        tangent_ratio=0.0,
        guidance_phase="always",
    )

    bank = {"format_version": 1, "states": {}}
    rows = []
    failures = 0
    try:
        for episode in range(args.layouts):
            layout_seed = args.seed * 1000 + episode
            # This mirrors the archived generator convention: seed global
            # Python/numpy, then reset and snapshot the realized initial state.
            random.seed(layout_seed)
            np.random.seed(layout_seed)
            env.reset()
            initial_state = env.get_state()
            initial_hash = PAIRED_EVAL.state_hash(initial_state)
            policy.start_episode()
            audit = CF.ContactPairAudit()
            metrics = PAIRED_EVAL.rollout_condition(
                policy,
                env,
                initial_state,
                "baseline",
                args.seed * 1_000_000 + episode,
                rollout_args,
                frame_callback=combined_frame_callback(
                    audit,
                    lambda *_args: None,
                ),
            )
            row = {
                "episode": episode,
                "layout_seed": layout_seed,
                "state_hash": initial_hash,
                "success": metrics["success"],
                "collision_any": metrics["collision_any"],
                "collision_failure": metrics["collision_failure"],
                "safe_success": metrics["safe_success"],
                "ncr": metrics["ncr"],
                "horizon": metrics["horizon"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if metrics["collision_failure"]:
                key = "{0}:{1}:{2}".format(CF.OBSTACLE_ENV, args.seed, episode)
                bank["states"][key] = {
                    "layout_seed": layout_seed,
                    "state": initial_state,
                    "state_hash": initial_hash,
                }
                failures += 1
                PAIRED_EVAL.save_state_bank(
                    args.out_dir / "fresh_initial_state_bank.pkl",
                    bank,
                )
                if failures >= args.target_failures:
                    break
    finally:
        raw = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
        raw.close()

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "layouts_attempted": len(rows),
        "collision_failures_kept": failures,
        "seed": args.seed,
        "horizon": args.horizon,
        "state_bank": str((args.out_dir / "fresh_initial_state_bank.pkl").resolve()),
        "rows": rows,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if failures == 0:
        raise RuntimeError("no collision-failure state found; increase layouts")


if __name__ == "__main__":
    main()
