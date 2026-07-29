"""Evaluate a simple overhead waypoint teacher on saved pre-collision states.

This is a mechanism prototype, not a data collector. It restores exact MuJoCo
states, executes a deterministic oracle EEF-position route, then hands control
back to the frozen delta-EEF Diffusion Policy.
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_DIR = SCRIPT_DIR.parent / "guided_denoising"
for directory in (SCRIPT_DIR, GUIDED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common as guided_common  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


DEFAULT_STATE_BANKS = [
    Path("outputs/task_constrained_guidance/one_chunk_lead_states/states.hdf5"),
    Path(
        "outputs/task_constrained_guidance/"
        "one_chunk_lead_states_seed601_602/states.hdf5"
    ),
    Path(
        "outputs/task_constrained_guidance/"
        "one_chunk_lead_states_seed603_607/states.hdf5"
    ),
]


def _text_value(item, name):
    value = item[name][()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_saved_state(item, bank_path):
    ep_meta = _text_value(item, "ep_meta")
    return {
        "id": "{}:{}".format(bank_path.parent.name, item.name.rsplit("/", 1)[-1]),
        "env": str(item.attrs["env"]),
        "seed": int(item.attrs["seed"]),
        "episode": int(item.attrs["episode"]),
        "start_step": int(item.attrs["start_step"]),
        "noise_seed": int(item.attrs["noise_seed"]),
        "state": {
            "states": item["sim_state"][()],
            "model": _text_value(item, "model_xml"),
            **({"ep_meta": ep_meta} if ep_meta else {}),
        },
    }


def interpolate_position_route(start_position, waypoints, max_delta_m):
    """Convert Cartesian waypoints to desired-goal delta position commands."""

    if max_delta_m <= 0:
        raise ValueError("max_delta_m must be positive")
    desired = np.asarray(start_position, dtype=np.float64).copy()
    deltas = []
    segment_ends = []
    for waypoint in waypoints:
        waypoint = np.asarray(waypoint, dtype=np.float64)
        distance = float(np.linalg.norm(waypoint - desired))
        steps = max(1, int(np.ceil(distance / max_delta_m)))
        segment_delta = (waypoint - desired) / steps
        deltas.extend([segment_delta.copy() for _ in range(steps)])
        desired = waypoint.copy()
        segment_ends.append(len(deltas))
    return np.asarray(deltas, dtype=np.float32), segment_ends


def overhead_waypoints(
    start_position,
    target_position,
    obstacle_centers,
    minimum_lift_m,
    object_overhead_m,
    handoff_height_m,
):
    """Construct lift, traverse, and target-handoff Cartesian waypoints."""

    start = np.asarray(start_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    object_z = [target[2]]
    if len(obstacle_centers):
        object_z.extend(np.asarray(obstacle_centers, dtype=np.float64)[:, 2].tolist())
    safe_z = max(
        float(start[2] + minimum_lift_m),
        float(max(object_z) + object_overhead_m),
    )
    handoff_z = float(target[2] + handoff_height_m)
    return np.asarray(
        [
            [start[0], start[1], safe_z],
            [target[0], target[1], safe_z],
            [target[0], target[1], handoff_z],
        ],
        dtype=np.float64,
    )


def action_from_position_delta(
    delta_position,
    gripper_command,
    delta_axis_angle=None,
):
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.asarray(delta_position, dtype=np.float32)
    if delta_axis_angle is not None:
        action[3:6] = np.asarray(delta_axis_angle, dtype=np.float32)
    action[6] = float(gripper_command)
    return action


def contact_state(raw_env, maps):
    target_contacts, obstacle_contacts, obstacle_names = (
        guided_common.contact_snapshot(raw_env, maps)
    )
    return target_contacts > 0, obstacle_contacts > 0, obstacle_names


def execute_action(
    env,
    raw_env,
    maps,
    obs,
    action,
    metrics,
    phase,
    label=None,
    transition_callback=None,
):
    action = np.asarray(action, dtype=np.float32)
    state_before = env.get_state() if transition_callback is not None else None
    obs_before = (
        {key: np.asarray(value).copy() for key, value in obs.items()}
        if transition_callback is not None
        else None
    )
    next_obs, reward, done, info = env.step(action)
    target_contact, obstacle_contact, obstacle_names = contact_state(raw_env, maps)
    metrics["target_contact_any"] |= target_contact
    metrics["collision_any"] |= obstacle_contact
    metrics["collision_names"].update(obstacle_names)
    metrics["{}_target_contact_any".format(phase)] |= target_contact
    metrics["{}_collision_any".format(phase)] |= obstacle_contact
    if target_contact:
        metrics["target_contact_events"].append(
            {"step": metrics["executed_steps"], "phase": phase, "label": label}
        )
    if obstacle_contact:
        metrics["collision_events"].append(
            {
                "step": metrics["executed_steps"],
                "phase": phase,
                "label": label,
                "obstacle_names": obstacle_names,
            }
        )
    metrics["success"] |= guided_common.task_success(env)
    metrics["executed_steps"] += 1
    if transition_callback is not None:
        transition_callback(
            state_before=state_before,
            obs=obs_before,
            next_obs={
                key: np.asarray(value).copy() for key, value in next_obs.items()
            },
            action=action.copy(),
            reward=float(reward),
            done=bool(done),
            info=info,
            phase=phase,
            label=label,
            target_contact=target_contact,
            obstacle_contact=obstacle_contact,
            obstacle_names=obstacle_names,
            task_success=metrics["success"],
        )
    return next_obs, bool(done or metrics["success"])


def execute_frozen_policy(
    policy,
    env,
    obs,
    raw_env,
    maps,
    metrics,
    target_object_name,
    maximum_steps,
    noise_seed_base,
    transition_callback=None,
):
    policy.start_episode()
    chunk_index = 0
    terminal = False
    while metrics["executed_steps"] < maximum_steps and not terminal:
        records = guided_common.active_object_records(raw_env)
        centers, radii, _ = guided_common.oracle_obstacle_geometry(
            raw_env, records, target_object_name
        )
        policy_obs = guided_common.policy_observation(obs, policy)
        context = guided_common.make_guidance_context(
            policy,
            policy_obs,
            centers,
            radii,
            scale=0.0,
            clearance_margin=0.0,
        )
        _, actions, _ = guided_common.sample_action_chunk(
            policy,
            policy_obs,
            context,
            noise_seed=noise_seed_base + chunk_index,
        )
        for action in actions:
            if metrics["executed_steps"] >= maximum_steps:
                break
            obs, terminal = execute_action(
                env,
                raw_env,
                maps,
                obs,
                action,
                metrics,
                "policy",
                "policy_chunk",
                transition_callback=transition_callback,
            )
            if terminal:
                break
        chunk_index += 1
    metrics["policy_chunks"] = chunk_index
    return obs


def new_metrics():
    return {
        "executed_steps": 0,
        "teacher_steps": 0,
        "policy_chunks": 0,
        "target_contact_any": False,
        "collision_any": False,
        "collision_names": set(),
        "teacher_target_contact_any": False,
        "teacher_collision_any": False,
        "policy_target_contact_any": False,
        "policy_collision_any": False,
        "target_contact_events": [],
        "collision_events": [],
        "success": False,
    }


def finalize_metrics(metrics, obs, raw_env, target_object_name):
    result = dict(metrics)
    result["collision_names"] = sorted(metrics["collision_names"])
    result["final_eef_position"] = guided_common.current_eef_position(obs).tolist()
    records = guided_common.active_object_records(raw_env)
    result["final_target_position"] = guided_common.target_object_position(
        raw_env, records, target_object_name
    ).tolist()
    result["final_staged_rewards"] = guided_common.staged_task_progress(raw_env)
    return result


def restore_stacked_observation(env, state, obs_history=None, timestep=None):
    """Reset a frame-stacked env and optionally restore its exact history."""

    obs = env.reset_to(state)
    if obs_history is None:
        return obs
    if not hasattr(env, "obs_history") or not hasattr(env, "num_frames"):
        raise TypeError("Expected a FrameStackWrapper when restoring obs history")
    restored = {}
    for key, values in obs_history.items():
        values = np.asarray(values)
        if values.shape[0] != env.num_frames:
            raise ValueError(
                "Observation history for {} has {} frames, expected {}".format(
                    key, values.shape[0], env.num_frames
                )
            )
        restored[key] = deque(
            [values[index : index + 1].copy() for index in range(env.num_frames)],
            maxlen=env.num_frames,
        )
    missing = set(obs) - set(restored)
    extra = set(restored) - set(obs)
    if missing or extra:
        raise ValueError(
            "Restored observation keys differ from environment keys; "
            "missing={} extra={}".format(sorted(missing), sorted(extra))
        )
    env.obs_history = restored
    if timestep is not None:
        env.timestep = int(timestep)
    return env._get_stacked_obs_from_history()


def evaluate_baseline(
    policy,
    env,
    saved,
    target_object_name,
    maximum_steps,
):
    obs = env.reset_to(saved["state"])
    raw_env = guided_common.raw_env_from_wrapper(env)
    records = guided_common.active_object_records(raw_env)
    maps = guided_common.contact_maps(raw_env, records, target_object_name)
    metrics = new_metrics()
    obs = execute_frozen_policy(
        policy,
        env,
        obs,
        raw_env,
        maps,
        metrics,
        target_object_name,
        maximum_steps,
        saved["noise_seed"],
    )
    return finalize_metrics(metrics, obs, raw_env, target_object_name)


def evaluate_teacher(
    policy,
    env,
    saved,
    target_object_name,
    maximum_steps,
    minimum_lift_m,
    object_overhead_m,
    handoff_height_m,
    max_delta_m,
    settle_steps,
    gripper_command,
    route,
    grasp_offset,
    close_steps,
    grasp_lift_m,
    relative_yaw_deg,
    max_rotation_delta_rad,
    restore_yaw_before_handoff,
    continue_with_policy=True,
    transition_callback=None,
):
    obs = restore_stacked_observation(
        env,
        saved["state"],
        obs_history=saved.get("obs_history"),
        timestep=saved.get("start_step"),
    )
    raw_env = guided_common.raw_env_from_wrapper(env)
    records = guided_common.active_object_records(raw_env)
    maps = guided_common.contact_maps(raw_env, records, target_object_name)
    target_position = guided_common.target_object_position(
        raw_env, records, target_object_name
    )
    centers, _, obstacle_names = guided_common.oracle_obstacle_geometry(
        raw_env, records, target_object_name
    )
    start_position = guided_common.current_eef_position(obs)
    overhead_route = overhead_waypoints(
        start_position,
        target_position,
        centers,
        minimum_lift_m=minimum_lift_m,
        object_overhead_m=object_overhead_m,
        handoff_height_m=handoff_height_m,
    )
    if route == "overhead_handoff":
        waypoints = overhead_route
    elif route == "grasp_handoff":
        grasp_position = target_position + np.asarray(grasp_offset, dtype=np.float32)
        pregrasp_position = grasp_position.copy()
        pregrasp_position[2] = max(
            float(grasp_position[2] + 0.06),
            float(target_position[2] + handoff_height_m),
        )
        waypoints = np.asarray(
            [
                overhead_route[0],
                overhead_route[1],
                pregrasp_position,
                grasp_position,
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError("Unknown route '{}'".format(route))
    deltas, segment_ends = interpolate_position_route(
        start_position, waypoints, max_delta_m=max_delta_m
    )

    metrics = new_metrics()
    terminal = False
    segment_end_set = set(segment_ends)
    segment_index = 0
    for action_index, delta in enumerate(deltas, start=1):
        action = action_from_position_delta(delta, gripper_command)
        obs, terminal = execute_action(
            env,
            raw_env,
            maps,
            obs,
            action,
            metrics,
            "teacher",
            "route_segment_{}".format(segment_index),
            transition_callback=transition_callback,
        )
        metrics["teacher_steps"] += 1
        if terminal or metrics["executed_steps"] >= maximum_steps:
            break
        if action_index in segment_end_set:
            hold = action_from_position_delta(np.zeros(3), gripper_command)
            for _ in range(settle_steps):
                obs, terminal = execute_action(
                    env,
                    raw_env,
                    maps,
                    obs,
                    hold,
                    metrics,
                    "teacher",
                    "settle_segment_{}".format(segment_index),
                    transition_callback=transition_callback,
                )
                metrics["teacher_steps"] += 1
                if terminal or metrics["executed_steps"] >= maximum_steps:
                    break
            segment_index += 1
            if segment_index == 2 and abs(relative_yaw_deg) > 1e-9:
                yaw_rad = float(np.deg2rad(relative_yaw_deg))
                rotation_steps = max(
                    1, int(np.ceil(abs(yaw_rad) / max_rotation_delta_rad))
                )
                delta_yaw = yaw_rad / rotation_steps
                for _ in range(rotation_steps):
                    rotate = action_from_position_delta(
                        np.zeros(3),
                        gripper_command,
                        delta_axis_angle=[0.0, 0.0, delta_yaw],
                    )
                    obs, terminal = execute_action(
                        env,
                        raw_env,
                        maps,
                        obs,
                        rotate,
                        metrics,
                        "teacher",
                        "yaw_rotation",
                        transition_callback=transition_callback,
                    )
                    metrics["teacher_steps"] += 1
                    if terminal or metrics["executed_steps"] >= maximum_steps:
                        break
        if terminal or metrics["executed_steps"] >= maximum_steps:
            break

    planned_teacher_steps = int(len(deltas) + settle_steps * len(segment_ends))
    if abs(relative_yaw_deg) > 1e-9:
        planned_teacher_steps += max(
            1,
            int(
                np.ceil(
                    abs(float(np.deg2rad(relative_yaw_deg)))
                    / max_rotation_delta_rad
                )
            ),
        )
    final_teacher_goal = waypoints[-1].copy()
    if (
        route == "grasp_handoff"
        and not terminal
        and metrics["executed_steps"] < maximum_steps
    ):
        close_action = action_from_position_delta(np.zeros(3), 1.0)
        for _ in range(close_steps):
            obs, terminal = execute_action(
                env,
                raw_env,
                maps,
                obs,
                close_action,
                metrics,
                "teacher",
                "close",
                transition_callback=transition_callback,
            )
            metrics["teacher_steps"] += 1
            if terminal or metrics["executed_steps"] >= maximum_steps:
                break
        lift_goal = waypoints[-1].copy()
        lift_goal[2] += grasp_lift_m
        lift_deltas, _ = interpolate_position_route(
            waypoints[-1], [lift_goal], max_delta_m=max_delta_m
        )
        for delta in lift_deltas:
            if terminal or metrics["executed_steps"] >= maximum_steps:
                break
            action = action_from_position_delta(delta, 1.0)
            obs, terminal = execute_action(
                env,
                raw_env,
                maps,
                obs,
                action,
                metrics,
                "teacher",
                "grasp_lift",
                transition_callback=transition_callback,
            )
            metrics["teacher_steps"] += 1
        final_teacher_goal = lift_goal
        planned_teacher_steps += close_steps + len(lift_deltas)
        if (
            restore_yaw_before_handoff
            and abs(relative_yaw_deg) > 1e-9
            and not terminal
            and metrics["executed_steps"] < maximum_steps
        ):
            restore_rad = -float(np.deg2rad(relative_yaw_deg))
            restore_steps = max(
                1, int(np.ceil(abs(restore_rad) / max_rotation_delta_rad))
            )
            restore_delta = restore_rad / restore_steps
            for _ in range(restore_steps):
                rotate = action_from_position_delta(
                    np.zeros(3),
                    1.0,
                    delta_axis_angle=[0.0, 0.0, restore_delta],
                )
                obs, terminal = execute_action(
                    env,
                    raw_env,
                    maps,
                    obs,
                    rotate,
                    metrics,
                    "teacher",
                    "restore_yaw",
                    transition_callback=transition_callback,
                )
                metrics["teacher_steps"] += 1
                if terminal or metrics["executed_steps"] >= maximum_steps:
                    break
            planned_teacher_steps += restore_steps

    handoff_position = guided_common.current_eef_position(obs)
    handoff_records = guided_common.active_object_records(raw_env)
    handoff_target_position = guided_common.target_object_position(
        raw_env, handoff_records, target_object_name
    )
    if (
        continue_with_policy
        and not terminal
        and metrics["executed_steps"] < maximum_steps
    ):
        obs = execute_frozen_policy(
            policy,
            env,
            obs,
            raw_env,
            maps,
            metrics,
            target_object_name,
            maximum_steps,
            saved["noise_seed"] + 100_000,
            transition_callback=transition_callback,
        )
    result = finalize_metrics(metrics, obs, raw_env, target_object_name)
    result.update(
        {
            "route": route,
            "relative_yaw_deg": relative_yaw_deg,
            "restore_yaw_before_handoff": restore_yaw_before_handoff,
            "start_position": start_position.tolist(),
            "start_target_position": target_position.tolist(),
            "target_position": target_position.tolist(),
            "obstacle_names_at_start": obstacle_names,
            "waypoints": waypoints.tolist(),
            "planned_teacher_steps": planned_teacher_steps,
            "handoff_position": handoff_position.tolist(),
            "handoff_target_position": handoff_target_position.tolist(),
            "target_lift_at_handoff_m": float(
                handoff_target_position[2] - target_position[2]
            ),
            "target_displacement_at_handoff_m": float(
                np.linalg.norm(handoff_target_position - target_position)
            ),
            "handoff_error_m": float(
                np.linalg.norm(
                    handoff_position
                    - np.asarray(final_teacher_goal, dtype=np.float32)
                )
            ),
        }
    )
    return result


def candidate_is_safe_pick(candidate, minimum_target_lift_m):
    return bool(
        candidate["teacher_target_contact_any"]
        and not candidate["teacher_collision_any"]
        and candidate["target_lift_at_handoff_m"] >= minimum_target_lift_m
    )


def select_yaw_candidate(candidates, minimum_target_lift_m):
    """Prefer the smallest orientation change that achieves a safe lifted pick."""

    if not candidates:
        raise ValueError("At least one yaw candidate is required")
    safe = [
        candidate
        for candidate in candidates
        if candidate_is_safe_pick(candidate, minimum_target_lift_m)
    ]
    if safe:
        return min(
            safe,
            key=lambda candidate: (
                abs(candidate["relative_yaw_deg"]),
                -candidate["target_lift_at_handoff_m"],
                candidate["handoff_error_m"],
            ),
        )

    # A collision-free contact or a lifted-but-colliding pick is still a more
    # useful fallback than a route that never reaches the target. This fallback
    # is diagnostic and is explicitly marked as not passing the safe-pick gate.
    return max(
        candidates,
        key=lambda candidate: (
            candidate["target_lift_at_handoff_m"] >= minimum_target_lift_m,
            candidate["teacher_target_contact_any"],
            not candidate["teacher_collision_any"],
            candidate["target_lift_at_handoff_m"],
            -abs(candidate["relative_yaw_deg"]),
            -candidate["handoff_error_m"],
        ),
    )


def compact_candidate(candidate, minimum_target_lift_m):
    return {
        "relative_yaw_deg": candidate["relative_yaw_deg"],
        "safe_pick": candidate_is_safe_pick(
            candidate, minimum_target_lift_m
        ),
        "teacher_target_contact_any": candidate["teacher_target_contact_any"],
        "teacher_collision_any": candidate["teacher_collision_any"],
        "collision_names": candidate["collision_names"],
        "target_lift_at_handoff_m": candidate["target_lift_at_handoff_m"],
        "target_displacement_at_handoff_m": (
            candidate["target_displacement_at_handoff_m"]
        ),
        "handoff_error_m": candidate["handoff_error_m"],
        "teacher_steps": candidate["teacher_steps"],
        "first_collision_event": (
            candidate["collision_events"][0]
            if candidate["collision_events"]
            else None
        ),
    }


def yaw_candidate_groups(yaw_candidates):
    """Group candidates by increasing absolute orientation change."""

    groups = {}
    for yaw in yaw_candidates:
        groups.setdefault(abs(float(yaw)), []).append(float(yaw))
    return [groups[magnitude] for magnitude in sorted(groups)]


def summarize(rows):
    summary = {
        "states": len(rows),
        "baseline": {
            "successes": sum(row["baseline"]["success"] for row in rows),
            "target_contacts": sum(
                row["baseline"]["target_contact_any"] for row in rows
            ),
            "collisions": sum(row["baseline"]["collision_any"] for row in rows),
        },
        "teacher_then_policy": {
            "successes": sum(row["teacher_then_policy"]["success"] for row in rows),
            "target_contacts": sum(
                row["teacher_then_policy"]["target_contact_any"] for row in rows
            ),
            "collisions": sum(
                row["teacher_then_policy"]["collision_any"] for row in rows
            ),
        },
        "paired": {
            "failure_to_success": sum(
                not row["baseline"]["success"]
                and row["teacher_then_policy"]["success"]
                for row in rows
            ),
            "success_to_failure": sum(
                row["baseline"]["success"]
                and not row["teacher_then_policy"]["success"]
                for row in rows
            ),
            "collision_to_safe": sum(
                row["baseline"]["collision_any"]
                and not row["teacher_then_policy"]["collision_any"]
                for row in rows
            ),
            "safe_to_collision": sum(
                not row["baseline"]["collision_any"]
                and row["teacher_then_policy"]["collision_any"]
                for row in rows
            ),
        },
    }
    searched = [row["yaw_search"] for row in rows if row["yaw_search"] is not None]
    if searched:
        yaw_counts = {}
        for result in searched:
            key = str(float(result["selected_yaw_deg"]))
            yaw_counts[key] = yaw_counts.get(key, 0) + 1
        summary["yaw_search"] = {
            "safe_picks": sum(result["selected_safe_pick"] for result in searched),
            "selected_yaw_counts": yaw_counts,
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=guided_common.DEFAULT_AGENT)
    parser.add_argument(
        "--states",
        nargs="+",
        type=Path,
        default=DEFAULT_STATE_BANKS,
    )
    parser.add_argument("--state-limit", type=int, default=None)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--rollout-horizon", type=int, default=400)
    parser.add_argument("--minimum-lift", type=float, default=0.06)
    parser.add_argument("--object-overhead", type=float, default=0.18)
    parser.add_argument("--handoff-height", type=float, default=0.12)
    parser.add_argument("--max-delta", type=float, default=0.006)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--gripper-command", type=float, default=-1.0)
    parser.add_argument(
        "--route",
        choices=["overhead_handoff", "grasp_handoff"],
        default="overhead_handoff",
    )
    parser.add_argument(
        "--grasp-offset",
        nargs=3,
        type=float,
        default=[0.002, -0.0025, 0.018],
        metavar=("DX", "DY", "DZ"),
        help="EEF position relative to the Can center while closing the gripper.",
    )
    parser.add_argument("--close-steps", type=int, default=10)
    parser.add_argument("--grasp-lift", type=float, default=0.15)
    parser.add_argument(
        "--relative-yaw-deg",
        type=float,
        default=0.0,
        help="Rotate the EEF about world Z above the Can before descending.",
    )
    parser.add_argument("--max-rotation-delta", type=float, default=0.04)
    parser.add_argument(
        "--restore-yaw-before-handoff",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--search-yaw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate grasp-and-lift candidates from the same saved state and "
            "execute the smallest safe yaw before policy handoff."
        ),
    )
    parser.add_argument(
        "--yaw-candidates",
        nargs="+",
        type=float,
        default=[0.0, -20.0, 20.0, -30.0, 30.0, -45.0, 45.0, 90.0],
    )
    parser.add_argument("--minimum-target-lift", type=float, default=0.05)
    parser.add_argument(
        "--exhaustive-yaw-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate larger yaw magnitudes even after a safe pick is found at "
            "the smallest tested magnitude."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/semi_manual_waypoint_teacher.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.state_limit is not None and args.state_limit <= 0:
        raise ValueError("--state-limit must be positive")
    if args.rollout_horizon <= 0:
        raise ValueError("--rollout-horizon must be positive")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be nonnegative")
    if args.close_steps < 0:
        raise ValueError("--close-steps must be nonnegative")
    if args.max_rotation_delta <= 0:
        raise ValueError("--max-rotation-delta must be positive")
    if args.minimum_target_lift <= 0:
        raise ValueError("--minimum-target-lift must be positive")
    if args.search_yaw and args.route != "grasp_handoff":
        raise ValueError("--search-yaw requires --route grasp_handoff")
    if args.search_yaw and not args.yaw_candidates:
        raise ValueError("--yaw-candidates must not be empty")

    saved_states = []
    for bank_path in args.states:
        with h5py.File(bank_path, "r") as input_file:
            for state_id in sorted(input_file["states"]):
                saved_states.append(
                    read_saved_state(input_file["states"][state_id], bank_path)
                )
                if (
                    args.state_limit is not None
                    and len(saved_states) >= args.state_limit
                ):
                    break
        if args.state_limit is not None and len(saved_states) >= args.state_limit:
            break

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = guided_common.load_guided_policy_and_checkpoint(
        args.agent, device
    )
    envs = {}
    rows = []
    try:
        for saved in saved_states:
            if saved["env"] not in envs:
                envs[saved["env"]], _ = FileUtils.env_from_checkpoint(
                    ckpt_dict=checkpoint,
                    env_name=saved["env"],
                    render=False,
                    render_offscreen=False,
                    verbose=False,
                )
            env = envs[saved["env"]]
            maximum_steps = max(1, args.rollout_horizon - saved["start_step"])
            baseline = evaluate_baseline(
                policy,
                env,
                saved,
                args.target_object_name,
                maximum_steps,
            )
            yaw_search = None
            selected_yaw = args.relative_yaw_deg
            if args.search_yaw:
                candidate_results = []
                for yaw_group in yaw_candidate_groups(args.yaw_candidates):
                    group_results = []
                    for candidate_yaw in yaw_group:
                        group_results.append(
                            evaluate_teacher(
                                policy,
                                env,
                                saved,
                                args.target_object_name,
                                maximum_steps,
                                args.minimum_lift,
                                args.object_overhead,
                                args.handoff_height,
                                args.max_delta,
                                args.settle_steps,
                                args.gripper_command,
                                args.route,
                                args.grasp_offset,
                                args.close_steps,
                                args.grasp_lift,
                                candidate_yaw,
                                args.max_rotation_delta,
                                False,
                                continue_with_policy=False,
                            )
                        )
                    candidate_results.extend(group_results)
                    if (
                        not args.exhaustive_yaw_search
                        and any(
                            candidate_is_safe_pick(
                                candidate, args.minimum_target_lift
                            )
                            for candidate in group_results
                        )
                    ):
                        break
                selected_candidate = select_yaw_candidate(
                    candidate_results, args.minimum_target_lift
                )
                selected_yaw = selected_candidate["relative_yaw_deg"]
                yaw_search = {
                    "minimum_target_lift_m": args.minimum_target_lift,
                    "selected_yaw_deg": selected_yaw,
                    "selected_safe_pick": candidate_is_safe_pick(
                        selected_candidate, args.minimum_target_lift
                    ),
                    "candidates": [
                        compact_candidate(candidate, args.minimum_target_lift)
                        for candidate in candidate_results
                    ],
                }
            teacher = evaluate_teacher(
                policy,
                env,
                saved,
                args.target_object_name,
                maximum_steps,
                args.minimum_lift,
                args.object_overhead,
                args.handoff_height,
                args.max_delta,
                args.settle_steps,
                args.gripper_command,
                args.route,
                args.grasp_offset,
                args.close_steps,
                args.grasp_lift,
                selected_yaw,
                args.max_rotation_delta,
                args.restore_yaw_before_handoff,
            )
            row = {
                "state_id": saved["id"],
                "env": saved["env"],
                "seed": saved["seed"],
                "episode": saved["episode"],
                "start_step": saved["start_step"],
                "baseline": baseline,
                "teacher_then_policy": teacher,
                "yaw_search": yaw_search,
            }
            rows.append(row)
            print(
                "{} yaw={:.1f} safe_pick={} success={}->{} target={}->{} collision={}->{} "
                "teacher_steps={} handoff_error_cm={:.2f}".format(
                    saved["id"],
                    selected_yaw,
                    (
                        yaw_search["selected_safe_pick"]
                        if yaw_search is not None
                        else None
                    ),
                    baseline["success"],
                    teacher["success"],
                    baseline["target_contact_any"],
                    teacher["target_contact_any"],
                    baseline["collision_any"],
                    teacher["collision_any"],
                    teacher["teacher_steps"],
                    100.0 * teacher["handoff_error_m"],
                ),
                flush=True,
            )
    finally:
        envs.clear()

    payload = {
        "config": {
            "agent": str(args.agent),
            "states": [str(path) for path in args.states],
            "state_limit": args.state_limit,
            "target_object_name": args.target_object_name,
            "rollout_horizon": args.rollout_horizon,
            "minimum_lift_m": args.minimum_lift,
            "object_overhead_m": args.object_overhead,
            "handoff_height_m": args.handoff_height,
            "max_delta_m": args.max_delta,
            "settle_steps": args.settle_steps,
            "gripper_command": args.gripper_command,
            "route": args.route,
            "grasp_offset_m": args.grasp_offset,
            "close_steps": args.close_steps,
            "grasp_lift_m": args.grasp_lift,
            "relative_yaw_deg": args.relative_yaw_deg,
            "max_rotation_delta_rad": args.max_rotation_delta,
            "restore_yaw_before_handoff": args.restore_yaw_before_handoff,
            "search_yaw": args.search_yaw,
            "yaw_candidates_deg": args.yaw_candidates,
            "minimum_target_lift_m": args.minimum_target_lift,
            "exhaustive_yaw_search": args.exhaustive_yaw_search,
        },
        "summary": summarize(rows),
        "states": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(guided_common.json_safe(payload), indent=2) + "\n"
    )
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print("wrote {}".format(args.out), flush=True)


if __name__ == "__main__":
    main()
