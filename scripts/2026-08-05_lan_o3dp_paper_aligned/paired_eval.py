"""Matched baseline versus paper-gradient LAN-O3DP rollout evaluation."""

import argparse
import hashlib
import itertools
import json
import pickle
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


def load_state_bank(path):
    with Path(path).open("rb") as stream:
        bank = pickle.load(stream)
    if bank.get("format_version") != 1 or not isinstance(bank.get("states"), dict):
        raise ValueError("Unsupported initial-state bank format: {}".format(path))
    return bank


def save_state_bank(path, bank):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(bank, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


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


def target_obstacle_contact_snapshot(raw_env, maps):
    """Count carried-target contacts omitted by the robot-obstacle metric."""

    _, target_ids, obstacle_by_geom = maps
    count = 0
    names = set()
    for index in range(raw_env.sim.data.ncon):
        contact = raw_env.sim.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        for target_geom, other_geom in ((geom1, geom2), (geom2, geom1)):
            if target_geom in target_ids and other_geom in obstacle_by_geom:
                count += 1
                names.add(obstacle_by_geom[other_geom])
                break
    return count, sorted(names)


def categorized_robot_geom_ids(raw_env):
    """Return separate contact-geom maps for arm links and grippers."""

    arm = {}
    gripper = {}

    def register(destination, names):
        for name in names:
            try:
                destination[raw_env.sim.model.geom_name2id(name)] = name
            except Exception:
                pass

    for robot in getattr(raw_env, "robots", []):
        model = getattr(robot, "robot_model", None)
        if model is not None:
            register(arm, getattr(model, "contact_geoms", []))
        robot_gripper = getattr(robot, "gripper", None)
        grippers = (
            robot_gripper.values()
            if isinstance(robot_gripper, dict)
            else [robot_gripper]
        )
        for item in grippers:
            if item is not None:
                register(gripper, getattr(item, "contact_geoms", []))
    return arm, gripper


def categorized_robot_obstacle_contact_snapshot(raw_env, maps, categories):
    """Return arm/gripper obstacle contacts and the exact robot geom names."""

    _, _, obstacle_by_geom = maps
    arm_ids, gripper_ids = categories
    arm_count = 0
    gripper_count = 0
    obstacle_names = set()
    robot_geom_names = set()
    for index in range(raw_env.sim.data.ncon):
        contact = raw_env.sim.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        for robot_geom, other_geom in ((geom1, geom2), (geom2, geom1)):
            if other_geom not in obstacle_by_geom:
                continue
            matched = False
            if robot_geom in arm_ids:
                arm_count += 1
                robot_geom_names.add(arm_ids[robot_geom])
                matched = True
            if robot_geom in gripper_ids:
                gripper_count += 1
                robot_geom_names.add(gripper_ids[robot_geom])
                matched = True
            if matched:
                obstacle_names.add(obstacle_by_geom[other_geom])
                break
    return arm_count, gripper_count, sorted(obstacle_names), sorted(robot_geom_names)


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


def rollout_condition(
    policy,
    env,
    initial_state,
    condition,
    policy_seed,
    args,
    *,
    guidance_scale=0.0,
    safety_distance=None,
    frame_callback=None,
):
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
    robot_geom_categories = categorized_robot_geom_ids(raw_env)
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
    robot_obstacle_collision_any = False
    arm_obstacle_collision_any = False
    gripper_obstacle_collision_any = False
    target_obstacle_collision_any = False
    collision_names = set()
    collision_robot_geom_names = set()
    clip_count = 0
    clip_max = 0.0
    reverse_steps = 0
    active_reverse_steps = 0
    update_norms = []
    obstacle_cloud_missing_chunks = 0
    minimum_clearance = float("inf")

    if frame_callback is not None:
        frame_callback(env, observation, step, success, collision_any)

    while (
        step < args.horizon
        and not success
        and not (args.terminate_on_collision and collision_any)
    ):
        current = np.asarray(
            rollout_common.latest_obs_value(observation, "agent_pos")[:3]
        )
        obstacle_cloud = visible_obstacle_pointcloud(raw_env, obstacle_ids)
        if obstacle_ids and len(obstacle_cloud) == 0:
            obstacle_cloud_missing_chunks += 1
        context = None
        if condition == "guided" and len(obstacle_cloud):
            if safety_distance is None:
                raise ValueError("guided rollouts require a safety distance")
            context = paper_lan_guidance_context_from_rollout_policy(
                policy,
                current_eef_pos=current,
                obstacle_points=obstacle_cloud,
                guidance_scale=guidance_scale,
                safety_distance=safety_distance,
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
            (
                arm_obstacle_contacts,
                gripper_obstacle_contacts,
                robot_names,
                robot_geom_names,
            ) = categorized_robot_obstacle_contact_snapshot(
                raw_env,
                contact_maps,
                robot_geom_categories,
            )
            target_obstacle_contacts, target_names = target_obstacle_contact_snapshot(
                raw_env,
                contact_maps,
            )
            arm_obstacle_collision_any = (
                arm_obstacle_collision_any or arm_obstacle_contacts > 0
            )
            gripper_obstacle_collision_any = (
                gripper_obstacle_collision_any or gripper_obstacle_contacts > 0
            )
            robot_obstacle_collision_any = (
                arm_obstacle_collision_any or gripper_obstacle_collision_any
            )
            target_obstacle_collision_any = (
                target_obstacle_collision_any or target_obstacle_contacts > 0
            )
            collision_any = (
                robot_obstacle_collision_any or target_obstacle_collision_any
            )
            collision_names.update(robot_names)
            collision_names.update(target_names)
            collision_robot_geom_names.update(robot_geom_names)
            step += 1
            if frame_callback is not None:
                frame_callback(env, observation, step, success, collision_any)
            if (
                done
                or (args.terminate_on_success and success)
                or (args.terminate_on_collision and collision_any)
            ):
                break
        chunks += 1
        if (
            done
            or (args.terminate_on_success and success)
            or (args.terminate_on_collision and collision_any)
        ):
            break

    target_end = rollout_common.target_object_position(
        raw_env,
        records,
        args.target_object_name,
    )
    return {
        "condition": condition,
        "guidance_scale": float(guidance_scale) if condition == "guided" else None,
        "safety_distance": float(safety_distance) if condition == "guided" else None,
        "success": success,
        "collision_any": collision_any,
        "robot_obstacle_collision_any": robot_obstacle_collision_any,
        "arm_obstacle_collision_any": arm_obstacle_collision_any,
        "gripper_obstacle_collision_any": gripper_obstacle_collision_any,
        "target_obstacle_collision_any": target_obstacle_collision_any,
        "safe_success": success and not collision_any,
        "success_with_collision": success and collision_any,
        "collision_failure": (not success) and collision_any,
        "ncr": (not success) and (not collision_any),
        "horizon_truncated": (not success) and step >= args.horizon,
        "collision_names": sorted(collision_names),
        "collision_robot_geom_names": sorted(collision_robot_geom_names),
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
        candidates = [("baseline", None, None)] + sorted(
            {
                (
                    "guided",
                    row["guidance_scale"],
                    row["safety_distance"],
                )
                for row in rows
                if row["env"] == env_name and row["condition"] == "guided"
            }
        )
        for condition, guidance_scale, safety_distance in candidates:
            selected = [
                row
                for row in rows
                if row["env"] == env_name and row["condition"] == condition
                and (
                    condition == "baseline"
                    or (
                        row["guidance_scale"] == guidance_scale
                        and row["safety_distance"] == safety_distance
                    )
                )
            ]
            summary.append(
                {
                    "env": env_name,
                    "condition": condition,
                    "guidance_scale": guidance_scale,
                    "safety_distance": safety_distance,
                    "episodes": len(selected),
                    "task_sr": float(np.mean([row["success"] for row in selected])),
                    "safe_sr": float(np.mean([row["safe_success"] for row in selected])),
                    "cr": float(np.mean([row["collision_any"] for row in selected])),
                    "robot_obstacle_cr": float(
                        np.mean(
                            [
                                row.get("robot_obstacle_collision_any", row["collision_any"])
                                for row in selected
                            ]
                        )
                    ),
                    "arm_obstacle_cr": float(
                        np.mean(
                            [
                                row.get("arm_obstacle_collision_any", False)
                                for row in selected
                            ]
                        )
                    ),
                    "gripper_obstacle_cr": float(
                        np.mean(
                            [
                                row.get("gripper_obstacle_collision_any", False)
                                for row in selected
                            ]
                        )
                    ),
                    "target_obstacle_cr": float(
                        np.mean(
                            [
                                row.get("target_obstacle_collision_any", False)
                                for row in selected
                            ]
                        )
                    ),
                    "ncr": float(np.mean([row["ncr"] for row in selected])),
                    "horizon_truncation_rate": float(
                        np.mean([row["horizon_truncated"] for row in selected])
                    ),
                    "safe_success": int(sum(row["safe_success"] for row in selected)),
                    "success_with_collision": int(sum(row["success_with_collision"] for row in selected)),
                    "collision_failure": int(sum(row["collision_failure"] for row in selected)),
                    "collision_free_non_completion": int(sum(row["ncr"] for row in selected)),
                    "horizon_truncated": int(
                        sum(row["horizon_truncated"] for row in selected)
                    ),
                }
            )
    return summary


def paired_transitions(rows):
    result = []
    for env_name in sorted({row["env"] for row in rows}):
        baselines = {
            (row["seed"], row["episode"]): row
            for row in rows
            if row["env"] == env_name and row["condition"] == "baseline"
        }
        candidates = sorted(
            {
                (row["guidance_scale"], row["safety_distance"])
                for row in rows
                if row["env"] == env_name and row["condition"] == "guided"
            }
        )
        for guidance_scale, safety_distance in candidates:
            guided = {
                (row["seed"], row["episode"]): row
                for row in rows
                if row["env"] == env_name
                and row["condition"] == "guided"
                and row["guidance_scale"] == guidance_scale
                and row["safety_distance"] == safety_distance
            }
            pair_keys = sorted(set(baselines) & set(guided))
            result.append(
                {
                    "env": env_name,
                    "guidance_scale": guidance_scale,
                    "safety_distance": safety_distance,
                    "pairs": len(pair_keys),
                    "failure_to_success": int(
                        sum(
                            not baselines[key]["success"] and guided[key]["success"]
                            for key in pair_keys
                        )
                    ),
                    "success_to_failure": int(
                        sum(
                            baselines[key]["success"] and not guided[key]["success"]
                            for key in pair_keys
                        )
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
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--baseline-only", action="store_true")
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--guidance-scale", type=float)
    scale_group.add_argument("--guidance-scales", nargs="+", type=float)
    distance_group = parser.add_mutually_exclusive_group()
    distance_group.add_argument("--safety-distance", type=float)
    distance_group.add_argument("--safety-distances", nargs="+", type=float)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--xy-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminate-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminate-on-collision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--state-bank-in", type=Path)
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
    if args.baseline_only:
        if any(
            value is not None
            for value in (
                args.guidance_scale,
                args.guidance_scales,
                args.safety_distance,
                args.safety_distances,
            )
        ):
            raise ValueError("baseline-only cannot be combined with guidance parameters")
        guidance_scales = []
        safety_distances = []
        candidates = []
    else:
        if args.guidance_scale is None and args.guidance_scales is None:
            raise ValueError("A guidance scale is required unless --baseline-only is set")
        if args.safety_distance is None and args.safety_distances is None:
            raise ValueError("A safety distance is required unless --baseline-only is set")
        guidance_scales = (
            args.guidance_scales
            if args.guidance_scales is not None
            else [args.guidance_scale]
        )
        safety_distances = (
            args.safety_distances
            if args.safety_distances is not None
            else [args.safety_distance]
        )
        candidates = list(itertools.product(guidance_scales, safety_distances))
    state_bank_path = args.out_dir / "initial_state_bank.pkl"
    if args.state_bank_in is None:
        state_bank = {"format_version": 1, "states": {}}
        state_bank_source = "generated"
    else:
        state_bank = load_state_bank(args.state_bank_in)
        state_bank_source = str(args.state_bank_in.resolve())
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
                    episodes = (
                        args.episodes
                        if args.episodes is not None
                        else range(args.n_rollouts)
                    )
                    for episode in episodes:
                        layout_seed = seed * 1000 + episode
                        policy_seed = seed * 1_000_000 + episode
                        state_key = "{}:{}:{}".format(env_name, seed, episode)
                        if args.state_bank_in is None:
                            random.seed(layout_seed)
                            np.random.seed(layout_seed)
                            env.reset()
                            initial_state = env.get_state()
                            state_bank["states"][state_key] = {
                                "layout_seed": layout_seed,
                                "state": initial_state,
                                "state_hash": state_hash(initial_state),
                            }
                            save_state_bank(state_bank_path, state_bank)
                        else:
                            if state_key not in state_bank["states"]:
                                raise KeyError(
                                    "State bank does not contain {}".format(state_key)
                                )
                            record = state_bank["states"][state_key]
                            if record["layout_seed"] != layout_seed:
                                raise ValueError(
                                    "State bank layout seed mismatch for {}".format(
                                        state_key
                                    )
                                )
                            initial_state = record["state"]
                        initial_hash = state_hash(initial_state)
                        if args.state_bank_in is not None:
                            if initial_hash != record["state_hash"]:
                                raise ValueError(
                                    "State bank hash mismatch for {}".format(state_key)
                                )
                        conditions = [("baseline", 0.0, None)] + [
                            ("guided", guidance_scale, safety_distance)
                            for guidance_scale, safety_distance in candidates
                        ]
                        for condition, guidance_scale, safety_distance in conditions:
                            metrics = rollout_condition(
                                policy,
                                env,
                                initial_state,
                                condition,
                                policy_seed,
                                args,
                                guidance_scale=guidance_scale,
                                safety_distance=safety_distance,
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
                                "{} seed={} episode={} {} rho={} q={} success={} collision={}".format(
                                    env_name,
                                    seed,
                                    episode,
                                    condition,
                                    metrics["guidance_scale"],
                                    metrics["safety_distance"],
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
        "episodes": args.episodes,
        "horizon": args.horizon,
        "baseline_only": args.baseline_only,
        "guidance_scales": guidance_scales,
        "safety_distances": safety_distances,
        "candidates": [
            {"guidance_scale": scale, "safety_distance": distance}
            for scale, distance in candidates
        ],
        "xy_only": args.xy_only,
        "terminate_on_collision": args.terminate_on_collision,
        "state_bank_source": state_bank_source,
        "state_bank_output": (
            str(state_bank_path.resolve())
            if args.state_bank_in is None
            else None
        ),
        "obstacle_protocol": "agentview 120x120 metric-depth oracle mask, visible FPS128 world points",
        "collision_protocol": (
            "union of arm-link, gripper, and target-Can contacts with obstacles; "
            "all components reported separately"
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote {}".format(args.out_dir))


if __name__ == "__main__":
    main()
