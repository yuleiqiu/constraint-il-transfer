"""Matched baseline versus paper-gradient LAN-O3DP rollout evaluation."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_DIR = SCRIPT_DIR.parent / "guided_denoising"
if str(GUIDED_DIR) not in sys.path:
    sys.path.append(str(GUIDED_DIR))

import common as rollout_common  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
from robomimic.utils.guided_denoising_utils import (  # noqa: E402
    flatten_action_normalization_stats,
)
from robomimic.utils.paper_lan_guidance_utils import (  # noqa: E402
    paper_guided_policy_from_checkpoint,
    paper_lan_guidance_context_from_rollout_policy,
)
from robomimic.utils.target_pointcloud_utils import (  # noqa: E402
    render_target_pointcloud,
)


DEFAULT_ENVS = [
    "PickPlaceCan",
    "PickPlaceBreadCan",
    "PickPlaceBreadCerealCan",
    "PickPlaceBreadCerealMilkCan",
]


def state_hash(state):
    digest = hashlib.sha256()
    digest.update(np.asarray(state["states"]).tobytes())
    digest.update((state.get("model") or "").encode("utf-8"))
    digest.update((state.get("ep_meta") or "").encode("utf-8"))
    return digest.hexdigest()


def obstacle_geom_ids(raw_env, records, target_object_name="Can"):
    target = target_object_name.lower()
    result = []
    for name, obj, _ in records:
        if name.lower() == target:
            continue
        for attribute in ("visual_geoms", "contact_geoms"):
            for geom_name in getattr(obj, attribute, []):
                try:
                    result.append(raw_env.sim.model.geom_name2id(geom_name))
                except Exception:
                    pass
    return tuple(sorted(set(result)))


def visible_obstacle_pointcloud(raw_env, geom_ids):
    if not geom_ids:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        return render_target_pointcloud(
            raw_env,
            config={
                "camera_name": "agentview",
                "height": 120,
                "width": 120,
                "num_points": 128,
                "padding_mode": "repeat",
                "max_geom_distance": None,
                "target_object": "unused",
                "include_visual_goal": False,
            },
            target_geom_ids=geom_ids,
        )
    except RuntimeError as error:
        if "No target pixels rendered" not in str(error):
            raise
        return np.zeros((0, 3), dtype=np.float32)


def sample_raw_chunk(policy, observation, context):
    if context is None:
        policy.policy.clear_guidance_context()
    else:
        policy.policy.set_guidance_context(context)
    prepared = policy._prepare_observation(observation)
    with torch.no_grad():
        normalized = policy.policy._get_action_trajectory(prepared)[0]
    normalized = normalized.detach().cpu().numpy()
    scale, offset = flatten_action_normalization_stats(
        policy.action_normalization_stats,
        policy.policy.global_config.train.action_keys,
    )
    raw = normalized * scale.numpy().reshape(1, -1) + offset.numpy().reshape(1, -1)
    return raw.astype(np.float32), policy.policy.get_guidance_diagnostics()


def rollout_condition(policy, env, initial_state, condition, policy_seed, args):
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    policy.start_episode()
    observation = env.reset_to(initial_state)
    raw_env = rollout_common.raw_env_from_wrapper(env)
    records = rollout_common.active_object_records(raw_env)
    contact_maps = rollout_common.contact_maps(
        raw_env,
        records,
        args.target_object_name,
    )
    obstacle_ids = obstacle_geom_ids(raw_env, records, args.target_object_name)
    target_start = rollout_common.target_object_position(
        raw_env,
        records,
        args.target_object_name,
    )
    low, high = rollout_common.action_limits(env, policy.policy.ac_dim)

    step = 0
    chunks = 0
    success = rollout_common.task_success(env)
    collision_any = False
    collision_names = set()
    clip_count = 0
    clip_max = 0.0
    reverse_steps = 0
    active_reverse_steps = 0
    update_norms = []
    obstacle_cloud_missing_chunks = 0
    minimum_clearance = float("inf")

    while step < args.horizon and not success:
        current = np.asarray(
            rollout_common.latest_obs_value(observation, "agent_pos")[:3]
        )
        obstacle_cloud = visible_obstacle_pointcloud(raw_env, obstacle_ids)
        if obstacle_ids and len(obstacle_cloud) == 0:
            obstacle_cloud_missing_chunks += 1
        context = None
        if condition == "guided" and len(obstacle_cloud):
            context = paper_lan_guidance_context_from_rollout_policy(
                policy,
                current_eef_pos=current,
                obstacle_points=obstacle_cloud,
                guidance_scale=args.guidance_scale,
                safety_distance=args.safety_distance,
                xy_only=args.xy_only,
            )
        raw_chunk, diagnostics = sample_raw_chunk(policy, observation, context)
        reverse_steps += len(diagnostics)
        active_reverse_steps += sum(
            item["active_waypoint_count"] > 0 for item in diagnostics
        )
        update_norms.extend(item["applied_update_norm"] for item in diagnostics)
        clip_delta = np.abs(np.clip(raw_chunk, low, high) - raw_chunk)
        clip_count += int(np.count_nonzero(clip_delta > 1e-5))
        clip_max = max(clip_max, float(np.max(clip_delta)))

        done = False
        for action in raw_chunk:
            if step >= args.horizon:
                break
            observation, _, done, _ = env.step(action)
            success = success or rollout_common.task_success(env)
            centers, radii, _ = rollout_common.oracle_obstacle_geometry(
                raw_env,
                records,
                args.target_object_name,
            )
            eef = np.asarray(
                rollout_common.latest_obs_value(observation, "agent_pos")[:3]
            )
            minimum_clearance = min(
                minimum_clearance,
                rollout_common.minimum_clearance(eef, centers, radii),
            )
            _, obstacle_contacts, names = rollout_common.contact_snapshot(
                raw_env,
                contact_maps,
            )
            collision_any = collision_any or obstacle_contacts > 0
            collision_names.update(names)
            step += 1
            if done or (args.terminate_on_success and success):
                break
        chunks += 1
        if done or (args.terminate_on_success and success):
            break

    target_end = rollout_common.target_object_position(
        raw_env,
        records,
        args.target_object_name,
    )
    return {
        "condition": condition,
        "success": success,
        "collision_any": collision_any,
        "safe_success": success and not collision_any,
        "success_with_collision": success and collision_any,
        "collision_failure": (not success) and collision_any,
        "ncr": (not success) and (not collision_any),
        "collision_names": sorted(collision_names),
        "horizon": step,
        "chunks": chunks,
        "target_displacement_m": float(np.linalg.norm(target_end - target_start)),
        "minimum_eef_obstacle_clearance_m": minimum_clearance,
        "guidance_trigger_rate": (
            active_reverse_steps / reverse_steps if reverse_steps else 0.0
        ),
        "guidance_update_norm_mean": (
            float(np.mean(update_norms)) if update_norms else 0.0
        ),
        "guidance_update_norm_max": (
            float(np.max(update_norms)) if update_norms else 0.0
        ),
        "obstacle_cloud_missing_chunks": obstacle_cloud_missing_chunks,
        "action_clip_count": clip_count,
        "action_clip_max_abs": clip_max,
    }


def aggregate(rows):
    summary = []
    for env_name in sorted({row["env"] for row in rows}):
        for condition in ("baseline", "guided"):
            selected = [
                row
                for row in rows
                if row["env"] == env_name and row["condition"] == condition
            ]
            summary.append(
                {
                    "env": env_name,
                    "condition": condition,
                    "episodes": len(selected),
                    "task_sr": float(np.mean([row["success"] for row in selected])),
                    "safe_sr": float(np.mean([row["safe_success"] for row in selected])),
                    "cr": float(np.mean([row["collision_any"] for row in selected])),
                    "ncr": float(np.mean([row["ncr"] for row in selected])),
                    "safe_success": int(sum(row["safe_success"] for row in selected)),
                    "success_with_collision": int(sum(row["success_with_collision"] for row in selected)),
                    "collision_failure": int(sum(row["collision_failure"] for row in selected)),
                    "collision_free_non_completion": int(sum(row["ncr"] for row in selected)),
                }
            )
    return summary


def paired_transitions(rows):
    result = []
    for env_name in sorted({row["env"] for row in rows}):
        pairs = {}
        for row in rows:
            if row["env"] == env_name:
                pairs.setdefault((row["seed"], row["episode"]), {})[
                    row["condition"]
                ] = row
        result.append(
            {
                "env": env_name,
                "pairs": len(pairs),
                "failure_to_success": int(
                    sum(not p["baseline"]["success"] and p["guided"]["success"] for p in pairs.values())
                ),
                "success_to_failure": int(
                    sum(p["baseline"]["success"] and not p["guided"]["success"] for p in pairs.values())
                ),
            }
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--safety-distance", type=float, required=True)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--xy-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminate-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Output already exists: {}".format(args.out_dir))
    args.out_dir.mkdir(parents=True)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = paper_guided_policy_from_checkpoint(
        ckpt_path=str(args.agent.resolve()),
        device=device,
    )
    rows = []
    with (args.out_dir / "episode_metrics.jsonl").open("w") as stream:
        for env_name in args.envs:
            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=checkpoint,
                env_name=env_name,
                render=False,
                render_offscreen=True,
            )
            try:
                for seed in args.seeds:
                    for episode in range(args.n_rollouts):
                        layout_seed = seed * 1000 + episode
                        policy_seed = seed * 1_000_000 + episode
                        random.seed(layout_seed)
                        np.random.seed(layout_seed)
                        env.reset()
                        initial_state = env.get_state()
                        initial_hash = state_hash(initial_state)
                        for condition in ("baseline", "guided"):
                            metrics = rollout_condition(
                                policy,
                                env,
                                initial_state,
                                condition,
                                policy_seed,
                                args,
                            )
                            row = {
                                "env": env_name,
                                "seed": seed,
                                "episode": episode,
                                "layout_seed": layout_seed,
                                "policy_seed": policy_seed,
                                "initial_state_hash": initial_hash,
                                **metrics,
                            }
                            rows.append(row)
                            stream.write(json.dumps(rollout_common.json_safe(row)) + "\n")
                            stream.flush()
                            print(
                                "{} seed={} episode={} {} success={} collision={}".format(
                                    env_name,
                                    seed,
                                    episode,
                                    condition,
                                    row["success"],
                                    row["collision_any"],
                                ),
                                flush=True,
                            )
            finally:
                rollout_common.raw_env_from_wrapper(env).close()

    (args.out_dir / "summary.json").write_text(
        json.dumps(aggregate(rows), indent=2) + "\n"
    )
    (args.out_dir / "paired_transitions.json").write_text(
        json.dumps(paired_transitions(rows), indent=2) + "\n"
    )
    manifest = {
        "agent": str(args.agent.resolve()),
        "envs": args.envs,
        "seeds": args.seeds,
        "n_rollouts": args.n_rollouts,
        "horizon": args.horizon,
        "guidance_scale": args.guidance_scale,
        "safety_distance": args.safety_distance,
        "xy_only": args.xy_only,
        "obstacle_protocol": "agentview 120x120 metric-depth oracle mask, visible FPS128 world points",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote {}".format(args.out_dir))


if __name__ == "__main__":
    main()
