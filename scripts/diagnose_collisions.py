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


def get_raw_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env


def sim_geom_id(sim, geom_name):
    try:
        return sim.model.geom_name2id(geom_name)
    except Exception:
        return None


def sim_body_pos(sim, body_name):
    try:
        body_id = sim.model.body_name2id(body_name)
    except Exception:
        return None
    return np.asarray(sim.data.body_xpos[body_id], dtype=np.float32)


def object_body_name(obj):
    for attr in ("root_body", "body_name", "root_body_name"):
        value = getattr(obj, attr, None)
        if value is not None:
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                value = value[0]
            return value
    name = getattr(obj, "name", None)
    return "{}_main".format(name) if name is not None else None


def object_geom_names(obj):
    geom_names = []
    for attr in ("visual_geoms", "contact_geoms"):
        for geom_name in getattr(obj, attr, []):
            if geom_name not in geom_names:
                geom_names.append(geom_name)
    return geom_names


def object_is_active_in_scene(sim, obj):
    body_name = object_body_name(obj)
    if body_name is None:
        return True
    try:
        body_id = sim.model.body_name2id(body_name)
    except Exception:
        return True
    return bool(sim.model.body_pos[body_id][2] > -10.0)


def geom_xy_radius(sim, geom_id):
    size = np.asarray(sim.model.geom_size[geom_id], dtype=np.float32)
    geom_type = int(sim.model.geom_type[geom_id])
    if geom_type in (2, 3, 5):
        return float(size[0])
    if geom_type in (4, 6):
        return float(np.linalg.norm(size[:2]))
    return float(np.linalg.norm(size[:2])) if size.shape[0] >= 2 else float(size[0])


def object_center_and_radius(sim, obj):
    centers = []
    radii = []
    for geom_name in object_geom_names(obj):
        geom_id = sim_geom_id(sim, geom_name)
        if geom_id is None:
            continue
        centers.append(np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float32))
        radii.append(geom_xy_radius(sim, geom_id))
    if centers:
        center = np.mean(np.stack(centers, axis=0), axis=0)
        radius = max(radii)
        return center, radius

    body_name = object_body_name(obj)
    center = sim_body_pos(sim, body_name) if body_name is not None else None
    if center is None:
        return None, None
    return center, 0.0


def get_oracle_obstacle_geometry(env, target_object_name=None, obstacle_names=None):
    raw_env = get_raw_env(env)
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        raise ValueError("Collision diagnosis requires simulator access")

    target_lower = target_object_name.lower() if target_object_name is not None else None
    obstacle_name_set = set(name.lower() for name in obstacle_names) if obstacle_names else None
    centers = []
    radii = []
    names = []
    for obj in getattr(raw_env, "objects", []):
        name = getattr(obj, "name", None)
        if name is None:
            continue
        name_lower = name.lower()
        if target_lower is not None and name_lower == target_lower:
            continue
        if obstacle_name_set is not None and name_lower not in obstacle_name_set:
            continue
        if not object_is_active_in_scene(sim, obj):
            continue
        center, radius = object_center_and_radius(sim, obj)
        if center is None:
            continue
        centers.append(center)
        radii.append(radius)
        names.append(name)
    if not centers:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), names
    return np.stack(centers, axis=0).astype(np.float32), np.asarray(radii, dtype=np.float32), names


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
            centers, radii, _ = get_oracle_obstacle_geometry(
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
