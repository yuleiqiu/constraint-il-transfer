"""
Diagnose failure modes in multi-object pick-place rollouts.

Records per-step EEF-to-obstacle distances to distinguish between
"collision failure" and "non-collision failure" (e.g., visual occlusion).

Usage:
    uv run python scripts/diagnose_collisions.py \
        --agent /path/to/model.pth \
        --env PickPlaceBreadCerealMilkCan \
        --n_rollouts 12 --horizon 400 --seed 600
"""

import argparse
import json
import sys
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils


def get_raw_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env


def rollout_with_diag(policy, env, horizon, target_object="Can", collision_threshold=0.03, verbose=True):
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    total_reward = 0.0
    collision_dists = []
    raw_env = get_raw_env(env)

    for step_i in range(horizon):
        act = policy(ob=obs)
        next_obs, r, done, _ = env.step(act)
        total_reward += r
        success = env.is_success()["task"]

        try:
            centers, radii, _, _, _ = ObstacleGuidanceUtils.get_oracle_obstacle_geometry(
                env=raw_env, target_object_name=target_object, obstacle_names=None)
            if len(centers) > 0:
                eef = np.array(next_obs.get("robot0_eef_pos", np.array([0, 0, 0])), dtype=np.float32)
                if eef.ndim == 2:
                    eef = eef[-1]
                dists = np.linalg.norm(centers[:, :2] - eef[:2], axis=1) - radii
                collision_dists.append(float(np.min(dists)))
        except Exception:
            collision_dists.append(float('nan'))

        if done or success:
            break
        obs = next_obs

    stats = dict(Horizon=step_i + 1, Success=int(success), Return=float(total_reward))

    valid = [d for d in collision_dists if not np.isnan(d) and np.isfinite(d)]
    stats["Collision_Min_Distance"] = float(np.min(valid)) if valid else float('nan')
    stats["Collision_Detected"] = any(d < collision_threshold for d in valid)
    stats["Collision_Hit_Count"] = sum(1 for d in valid if d < collision_threshold)
    stats["Collision_Hit_Ratio"] = (
        float(stats["Collision_Hit_Count"]) / len(valid) if valid else 0.0
    )

    if verbose:
        status = "SUCCESS" if success else "FAIL"
        coll_flag = "COLLISION" if stats["Collision_Detected"] else "CLEAN"
        print(
            "[{}] horizon={:3d}  coll_min={:+.4f}m  {}  hit_ratio={:.1%}".format(
                status, stats["Horizon"], stats["Collision_Min_Distance"],
                coll_flag, stats["Collision_Hit_Ratio"],
            ),
            flush=True,
        )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="path to saved checkpoint")
    parser.add_argument("--env", type=str, default=None, help="environment name override")
    parser.add_argument("--n_rollouts", type=int, default=12, help="number of rollouts")
    parser.add_argument("--horizon", type=int, default=400, help="rollout horizon")
    parser.add_argument("--seed", type=int, default=600, help="rollout seed")
    parser.add_argument("--collision_target", type=str, default="Can", help="target object name")
    parser.add_argument("--collision_threshold", type=float, default=0.03, help="collision distance threshold (m)")
    parser.add_argument("--output", type=str, default=None, help="optional JSON output path")
    args = parser.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=False)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, env_name=args.env, render=False, render_offscreen=False, verbose=False)

    if args.seed is not None:
        np.random.seed(args.seed)

    rollouts = []
    for i in range(args.n_rollouts):
        stats = rollout_with_diag(
            policy=policy, env=env, horizon=args.horizon,
            target_object=args.collision_target, collision_threshold=args.collision_threshold,
        )
        rollouts.append(stats)

    n_success = sum(r["Success"] for r in rollouts)
    success_rate = n_success / len(rollouts)
    failures = [r for r in rollouts if not r["Success"]]
    collision_fails = [r for r in failures if r["Collision_Detected"]]
    non_collision_fails = [r for r in failures if not r["Collision_Detected"]]

    print("\n=== Summary ===")
    print(f"Total:   {len(rollouts)}")
    print(f"Success: {n_success} ({success_rate:.1%})")
    print(f"Fail:    {len(failures)}")
    if failures:
        print(f"  - collision:    {len(collision_fails)}")
        print(f"  - non-collision: {len(non_collision_fails)}")

    if args.output:
        payload = dict(args=vars(args), summary=dict(
            total=len(rollouts), success=n_success, success_rate=success_rate,
            num_failures=len(failures),
            num_collision_failures=len(collision_fails),
            num_non_collision_failures=len(non_collision_fails),
        ), rollouts=rollouts)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
