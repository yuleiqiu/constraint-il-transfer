"""Evaluate the eight-step feasibility upper bound on saved collision states."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_DIR = SCRIPT_DIR.parent / "guided_denoising"
for directory in (SCRIPT_DIR, GUIDED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common as guided_common  # noqa: E402
from feasibility import (  # noqa: E402
    FeasibilityConfig,
    closest_target_progress,
    minimum_segment_clearance_xy,
    reconstruct_positions,
    solve_eight_step_feasibility,
)
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


def _text_value(item, name):
    value = item[name][()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_state(item):
    ep_meta = _text_value(item, "ep_meta")
    return {
        "id": item.name.rsplit("/", 1)[-1],
        "env": str(item.attrs["env"]),
        "seed": int(item.attrs["seed"]),
        "episode": int(item.attrs["episode"]),
        "chunk": int(item.attrs["chunk"]),
        "start_step": int(item.attrs["start_step"]),
        "noise_seed": int(item.attrs["noise_seed"]),
        "state": {
            "states": item["sim_state"][()],
            "model": _text_value(item, "model_xml"),
            **({"ep_meta": ep_meta} if ep_meta else {}),
        },
        "obs": {key: dataset[()] for key, dataset in item["obs"].items()},
    }


def robot_geom_categories(raw_env):
    categories = {}
    for robot in raw_env.robots:
        model = getattr(robot, "robot_model", None)
        for name in getattr(model, "contact_geoms", []):
            try:
                categories[raw_env.sim.model.geom_name2id(name)] = "arm"
            except Exception:
                pass
        grippers = robot.gripper.values() if isinstance(robot.gripper, dict) else [robot.gripper]
        for gripper in grippers:
            for name in getattr(gripper, "contact_geoms", []):
                try:
                    categories[raw_env.sim.model.geom_name2id(name)] = "gripper"
                except Exception:
                    pass
    return categories


def obstacle_contact_details(raw_env, maps, categories, step):
    robot_ids, _, obstacle_by_geom = maps
    details = []
    for index in range(raw_env.sim.data.ncon):
        contact = raw_env.sim.data.contact[index]
        pairs = (
            (int(contact.geom1), int(contact.geom2)),
            (int(contact.geom2), int(contact.geom1)),
        )
        for robot_geom, obstacle_geom in pairs:
            if robot_geom in robot_ids and obstacle_geom in obstacle_by_geom:
                details.append(
                    {
                        "step": step,
                        "category": categories.get(robot_geom, "unknown"),
                        "robot_geom": raw_env.sim.model.geom_id2name(robot_geom),
                        "obstacle": obstacle_by_geom[obstacle_geom],
                        "obstacle_geom": raw_env.sim.model.geom_id2name(obstacle_geom),
                    }
                )
                break
    return details


def evaluate_condition(
    policy,
    env,
    saved_state,
    first_actions,
    target_position,
    centers,
    radii,
    maps,
    categories,
    target_object_name,
    continuation_chunks,
    continuation_noise_seed,
    repeat_feasibility_config=None,
    eligibility_clearance_threshold=0.0,
    maximum_executed_steps=None,
):
    """Execute one supplied chunk, then resume the frozen policy normally."""

    policy.start_episode()
    obs = env.reset_to(saved_state)
    raw_env = guided_common.raw_env_from_wrapper(env)
    start_position = guided_common.current_eef_position(obs)
    actual_positions = []
    collision_names = set()
    contact_details = []
    target_contact_any = False
    success = guided_common.task_success(env)
    done = False
    chunk_summaries = []
    repeat_correction_attempts = 0
    repeat_correction_accepts = 0
    action_index = 0
    actions = np.asarray(first_actions, dtype=np.float32)
    for chunk_offset in range(continuation_chunks + 1):
        if maximum_executed_steps is not None and action_index >= maximum_executed_steps:
            break
        chunk_start_position = guided_common.current_eef_position(obs)
        correction_eligible = None
        repeated_solver = None
        predicted_progress = None
        predicted_clearance = None
        if chunk_offset == 0:
            noise_seed = None
            source = "supplied_first_chunk"
        else:
            records = guided_common.active_object_records(raw_env)
            chunk_centers, chunk_radii, _ = guided_common.oracle_obstacle_geometry(
                raw_env, records, target_object_name
            )
            policy_obs = guided_common.policy_observation(obs, policy)
            context = guided_common.make_guidance_context(
                policy,
                policy_obs,
                chunk_centers,
                chunk_radii,
                scale=0.0,
                clearance_margin=0.0,
            )
            noise_seed = continuation_noise_seed + chunk_offset - 1
            _, actions, _ = guided_common.sample_action_chunk(
                policy,
                policy_obs,
                context,
                noise_seed=noise_seed,
            )
            source = "frozen_policy_resample"

            predicted_positions = reconstruct_positions(chunk_start_position, actions)
            chunk_target_position = guided_common.target_object_position(
                raw_env, records, target_object_name
            )
            predicted_progress = closest_target_progress(
                chunk_start_position, predicted_positions, chunk_target_position
            )
            predicted_clearance = minimum_segment_clearance_xy(
                chunk_start_position,
                predicted_positions,
                chunk_centers,
                chunk_radii,
            )
            correction_eligible = (
                not target_contact_any
                and predicted_progress > 0.0
                and predicted_clearance <= eligibility_clearance_threshold
            )
            if repeat_feasibility_config is not None and correction_eligible:
                repeat_correction_attempts += 1
                low, high = guided_common.action_limits(env, actions.shape[1])
                repeated_solver = solve_eight_step_feasibility(
                    start_position=chunk_start_position,
                    baseline_actions=actions,
                    target_position=chunk_target_position,
                    obstacle_centers=chunk_centers,
                    obstacle_radii=chunk_radii,
                    action_low=low,
                    action_high=high,
                    config=repeat_feasibility_config,
                )
                if repeated_solver.feasible:
                    actions = repeated_solver.corrected_actions
                    source = "repeated_feasibility_correction"
                    repeat_correction_accepts += 1

        chunk_positions = []
        chunk_collision_names = set()
        for action in actions:
            if (
                maximum_executed_steps is not None
                and action_index >= maximum_executed_steps
            ):
                break
            obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32))
            position = guided_common.current_eef_position(obs)
            actual_positions.append(position)
            chunk_positions.append(position)
            target_contacts, obstacle_contacts, names = guided_common.contact_snapshot(
                raw_env, maps
            )
            target_contact_any = target_contact_any or target_contacts > 0
            if obstacle_contacts > 0:
                collision_names.update(names)
                chunk_collision_names.update(names)
                contact_details.extend(
                    obstacle_contact_details(raw_env, maps, categories, action_index)
                )
            success = success or guided_common.task_success(env)
            action_index += 1
            if done:
                break

        chunk_positions = np.asarray(chunk_positions, dtype=np.float64)
        chunk_summaries.append(
            {
                "chunk_offset": chunk_offset,
                "source": source,
                "noise_seed": noise_seed,
                "actions": actions,
                "executed_steps": len(chunk_positions),
                "actual_positions": chunk_positions,
                "collision_any": bool(chunk_collision_names),
                "collision_names": sorted(chunk_collision_names),
                "correction_eligible": correction_eligible,
                "predicted_target_progress_m": predicted_progress,
                "predicted_min_clearance_m": predicted_clearance,
                "repeated_solver": repeated_solver.to_dict()
                if repeated_solver is not None
                else None,
            }
        )
        if done:
            break

    actual_positions = np.asarray(actual_positions, dtype=np.float64)
    if len(actual_positions):
        clearance = minimum_segment_clearance_xy(
            start_position, actual_positions, centers, radii
        )
        progress = closest_target_progress(
            start_position, actual_positions, target_position
        )
    else:
        clearance = float("inf")
        progress = 0.0
    return {
        "executed_steps": len(actual_positions),
        "executed_chunks": len(chunk_summaries),
        "chunk_summaries": chunk_summaries,
        "repeat_correction_attempts": repeat_correction_attempts,
        "repeat_correction_accepts": repeat_correction_accepts,
        "actual_positions": actual_positions,
        "actual_min_clearance_m": clearance,
        "actual_target_progress_m": progress,
        "collision_any": bool(collision_names),
        "collision_names": sorted(collision_names),
        "contact_details": contact_details,
        "first_contact_step": min(
            (item["step"] for item in contact_details), default=None
        ),
        "contact_categories": sorted({item["category"] for item in contact_details}),
        "target_contact_any": target_contact_any,
        "success": success,
        "done": bool(done),
    }


def summarize(rows):
    eligible = [row for row in rows if row["eligible"]]
    solved = [row for row in eligible if row["solver"]["feasible"]]
    executed = [row for row in solved if row.get("corrected_execution") is not None]
    collision_conversions = sum(
        row["baseline_execution"]["collision_any"]
        and not row["corrected_execution"]["collision_any"]
        for row in executed
    )
    collision_regressions = sum(
        not row["baseline_execution"]["collision_any"]
        and row["corrected_execution"]["collision_any"]
        for row in executed
    )
    target_contact_gains = sum(
        not row["baseline_execution"]["target_contact_any"]
        and row["corrected_execution"]["target_contact_any"]
        for row in executed
    )
    return {
        "states": len(rows),
        "eligible_target_directed_obstructed": len(eligible),
        "solver_feasible": len(solved),
        "solver_feasible_rate": len(solved) / len(eligible) if eligible else None,
        "executed_pairs": len(executed),
        "baseline_collision_count": sum(
            row["baseline_execution"]["collision_any"] for row in executed
        ),
        "corrected_collision_count": sum(
            row["corrected_execution"]["collision_any"] for row in executed
        ),
        "collision_to_safe_conversions": collision_conversions,
        "safe_to_collision_regressions": collision_regressions,
        "target_contact_gains": target_contact_gains,
        "mean_predicted_clearance_change_m": float(
            np.mean(
                [
                    row["solver"]["corrected_min_clearance_m"]
                    - row["solver"]["baseline_min_clearance_m"]
                    for row in solved
                ]
            )
        )
        if solved
        else None,
        "mean_actual_clearance_change_m": float(
            np.mean(
                [
                    row["corrected_execution"]["actual_min_clearance_m"]
                    - row["baseline_execution"]["actual_min_clearance_m"]
                    for row in executed
                ]
            )
        )
        if executed
        else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=guided_common.DEFAULT_AGENT)
    parser.add_argument(
        "--states",
        type=Path,
        default=Path("outputs/guided_denoising/same_state/states.hdf5"),
    )
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument(
        "--state-ids",
        nargs="+",
        default=None,
        help="Optional exact saved-state ids to evaluate.",
    )
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--clearance-margin", type=float, default=0.0)
    parser.add_argument("--clearance-recovery-steps", type=int, default=0)
    parser.add_argument("--eligibility-clearance-threshold", type=float, default=0.0)
    parser.add_argument("--progress-tolerance", type=float, default=5e-4)
    parser.add_argument("--trust-radius", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=1.0)
    parser.add_argument(
        "--continuation-chunks",
        type=int,
        default=0,
        help=(
            "After the supplied eight-action chunk, resample and execute this many "
            "ordinary eight-action chunks from the frozen policy."
        ),
    )
    parser.add_argument(
        "--repeat-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "At continuation replans before target contact, reapply the same "
            "feasibility solver to eligible chunks in the corrected branch."
        ),
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=None,
        help=(
            "Optional original-episode step limit. Each saved state executes at "
            "most rollout_horizon - start_step actions."
        ),
    )
    parser.add_argument(
        "--require-target-directed-obstructed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/task_constrained_guidance/fixed_state_feasibility"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.continuation_chunks < 0:
        raise ValueError("--continuation-chunks must be nonnegative")
    if args.rollout_horizon is not None and args.rollout_horizon <= 0:
        raise ValueError("--rollout-horizon must be positive")
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError("{} exists; pass --overwrite".format(args.out_dir))
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    config = FeasibilityConfig(
        clearance_margin_m=args.clearance_margin,
        clearance_recovery_steps=args.clearance_recovery_steps,
        progress_tolerance_m=args.progress_tolerance,
        trust_radius_m=args.trust_radius,
        smoothness_weight=args.smoothness_weight,
    )
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt = guided_common.load_guided_policy_and_checkpoint(args.agent, device)
    envs = {}
    rows = []
    try:
        with h5py.File(args.states, "r") as input_file:
            for state_id in sorted(input_file["states"]):
                if args.state_ids is not None and state_id not in args.state_ids:
                    continue
                saved = read_state(input_file["states"][state_id])
                if args.envs is not None and saved["env"] not in args.envs:
                    continue
                if saved["env"] not in envs:
                    envs[saved["env"]], _ = FileUtils.env_from_checkpoint(
                        ckpt_dict=ckpt,
                        env_name=saved["env"],
                        render=False,
                        render_offscreen=False,
                        verbose=False,
                    )
                env = envs[saved["env"]]
                policy.start_episode()
                env.reset_to(saved["state"])
                raw_env = guided_common.raw_env_from_wrapper(env)
                records = guided_common.active_object_records(raw_env)
                maps = guided_common.contact_maps(
                    raw_env, records, args.target_object_name
                )
                categories = robot_geom_categories(raw_env)
                centers, radii, obstacle_names = guided_common.oracle_obstacle_geometry(
                    raw_env, records, args.target_object_name
                )
                target_position = guided_common.target_object_position(
                    raw_env, records, args.target_object_name
                )
                start_position = guided_common.current_eef_position(saved["obs"])
                context = guided_common.make_guidance_context(
                    policy,
                    saved["obs"],
                    centers,
                    radii,
                    scale=0.0,
                    clearance_margin=0.0,
                )
                _, baseline_actions, _ = guided_common.sample_action_chunk(
                    policy,
                    saved["obs"],
                    context,
                    noise_seed=saved["noise_seed"],
                )
                baseline_positions = reconstruct_positions(
                    start_position, baseline_actions
                )
                baseline_progress = closest_target_progress(
                    start_position, baseline_positions, target_position
                )
                baseline_clearance = minimum_segment_clearance_xy(
                    start_position, baseline_positions, centers, radii
                )
                eligible = (
                    baseline_progress > 0.0
                    and baseline_clearance <= args.eligibility_clearance_threshold
                )
                low, high = guided_common.action_limits(env, baseline_actions.shape[1])
                solver_result = solve_eight_step_feasibility(
                    start_position=start_position,
                    baseline_actions=baseline_actions,
                    target_position=target_position,
                    obstacle_centers=centers,
                    obstacle_radii=radii,
                    action_low=low,
                    action_high=high,
                    config=config,
                )

                baseline_execution = evaluate_condition(
                    policy,
                    env,
                    saved["state"],
                    baseline_actions,
                    target_position,
                    centers,
                    radii,
                    maps,
                    categories,
                    args.target_object_name,
                    args.continuation_chunks,
                    saved["noise_seed"] + 1,
                    maximum_executed_steps=(
                        max(0, args.rollout_horizon - saved["start_step"])
                        if args.rollout_horizon is not None
                        else None
                    ),
                )
                corrected_execution = None
                should_execute = solver_result.feasible and (
                    eligible or not args.require_target_directed_obstructed
                )
                if should_execute:
                    corrected_execution = evaluate_condition(
                        policy,
                        env,
                        saved["state"],
                        solver_result.corrected_actions,
                        target_position,
                        centers,
                        radii,
                        maps,
                        categories,
                        args.target_object_name,
                        args.continuation_chunks,
                        saved["noise_seed"] + 1,
                        config if args.repeat_correction else None,
                        args.eligibility_clearance_threshold,
                        (
                            max(0, args.rollout_horizon - saved["start_step"])
                            if args.rollout_horizon is not None
                            else None
                        ),
                    )
                row = {
                    "state_id": state_id,
                    "env": saved["env"],
                    "seed": saved["seed"],
                    "episode": saved["episode"],
                    "chunk": saved["chunk"],
                    "noise_seed": saved["noise_seed"],
                    "obstacle_names": obstacle_names,
                    "target_position": target_position,
                    "start_position": start_position,
                    "baseline_actions": baseline_actions,
                    "baseline_predicted_positions": baseline_positions,
                    "baseline_target_progress_m": baseline_progress,
                    "baseline_min_clearance_m": baseline_clearance,
                    "eligible": eligible,
                    "solver": solver_result.to_dict(),
                    "baseline_execution": baseline_execution,
                    "corrected_execution": corrected_execution,
                }
                rows.append(row)
                print(
                    "{} eligible={} feasible={} predicted_clearance={:.4f}->{:.4f} actual_collision={}->{}".format(
                        state_id,
                        eligible,
                        solver_result.feasible,
                        baseline_clearance,
                        solver_result.corrected_min_clearance_m,
                        baseline_execution["collision_any"],
                        corrected_execution["collision_any"]
                        if corrected_execution is not None
                        else None,
                    ),
                    flush=True,
                )
    finally:
        envs.clear()

    jsonl_path = args.out_dir / "state_metrics.jsonl"
    with jsonl_path.open("w") as output:
        for row in rows:
            output.write(json.dumps(guided_common.json_safe(row)) + "\n")
    summary = summarize(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(guided_common.json_safe(summary), indent=2) + "\n"
    )
    manifest = {
        "agent": str(args.agent),
        "states": str(args.states),
        "envs": args.envs,
        "state_ids": args.state_ids,
        "target_object_name": args.target_object_name,
        "require_target_directed_obstructed": args.require_target_directed_obstructed,
        "eligibility_clearance_threshold_m": args.eligibility_clearance_threshold,
        "continuation_chunks": args.continuation_chunks,
        "repeat_correction": args.repeat_correction,
        "rollout_horizon": args.rollout_horizon,
        "frozen_horizons": {"observation": 2, "prediction": 16, "action": 8},
        "config": as_json(config),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(guided_common.json_safe(summary), indent=2), flush=True)


def as_json(config):
    return {
        key: value
        for key, value in config.__dict__.items()
    }


if __name__ == "__main__":
    main()
