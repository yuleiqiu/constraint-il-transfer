"""Evaluate source and SDP action chunks by cloned-simulator continuation.

For each held-out pre-collision state, sample one source chunk and K corrective
chunks from the SDP policy. Restore the exact simulator state and observation
history for every branch, execute one candidate chunk, then hand control to the
same frozen source policy until the original 400-step episode horizon.
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_DIR = SCRIPT_DIR.parent / "guided_denoising"
MANIFOLD_DIR = SCRIPT_DIR.parent / "2026-07-27_policy_manifold_diagnostic"
TEACHER_DIR = SCRIPT_DIR.parent / "2026-07-27_semi_manual_waypoint_teacher"
SELECTOR_DIR = SCRIPT_DIR.parent / "2026-07-29_gated_support_selector"
for directory in (
    SCRIPT_DIR,
    GUIDED_DIR,
    MANIFOLD_DIR,
    TEACHER_DIR,
    SELECTOR_DIR,
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common as guided_common  # noqa: E402
import evaluate_waypoint_teacher as teacher  # noqa: E402
from branch_value import (  # noqa: E402
    outcome_label,
    select_oracle_branch,
    summarize_states,
)
from run_fixed_state import DEFAULT_STATE_BANKS, read_state  # noqa: E402
from selector import (  # noqa: E402
    SelectorConfig,
    chunk_geometry,
    normalized_chunk_deviation,
)
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


DEFAULT_CORRECTIVE_AGENT = Path(
    "outputs/robomimic/train/dp_can_delta_pose_osc_targeted_sdp/"
    "20260729173254/models/model_epoch_12.pth"
)


def json_dump(path, payload):
    path.write_text(
        json.dumps(guided_common.json_safe(payload), indent=2) + "\n"
    )


def load_saved_states(paths, state_limit):
    states = []
    for path in paths:
        with h5py.File(path, "r") as input_file:
            for key in sorted(input_file["states"]):
                item = input_file["states"][key]
                saved = read_state(item, path.parent.name)
                saved["baseline_actions"] = item["baseline_actions"][()]
                states.append(saved)
                if state_limit is not None and len(states) >= state_limit:
                    return states
    return states


def restore_policy_observation_history(env, saved):
    """Restore exact saved policy keys while retaining unused wrapper keys."""

    obs = env.reset_to(saved["state"])
    if not hasattr(env, "obs_history") or not hasattr(env, "num_frames"):
        raise TypeError("Expected a FrameStackWrapper")
    missing = set(saved["obs"]) - set(env.obs_history)
    if missing:
        raise KeyError(
            "Saved policy observations are absent from wrapper history: {}".format(
                sorted(missing)
            )
        )
    for key, values in saved["obs"].items():
        values = np.asarray(values)
        if values.shape[0] != env.num_frames:
            raise ValueError(
                "Observation history for {} has {} frames, expected {}".format(
                    key, values.shape[0], env.num_frames
                )
            )
        env.obs_history[key] = deque(
            [
                values[index : index + 1].copy()
                for index in range(env.num_frames)
            ],
            maxlen=env.num_frames,
        )
    env.timestep = int(saved["start_step"])
    restored = env._get_stacked_obs_from_history()
    for key, expected in saved["obs"].items():
        if not np.array_equal(np.asarray(restored[key]), np.asarray(expected)):
            raise RuntimeError(
                "Failed to restore exact policy observation history for {}".format(
                    key
                )
            )
    return restored


def sample_candidates(
    *,
    source_policy,
    corrective_policy,
    saved,
    raw_env,
    records,
    target_object_name,
    candidate_count,
    selector_config,
):
    centers, radii, obstacle_names = guided_common.oracle_obstacle_geometry(
        raw_env, records, target_object_name
    )
    target_position = guided_common.target_object_position(
        raw_env, records, target_object_name
    )
    source_obs = guided_common.policy_observation(saved["obs"], source_policy)
    start_position = guided_common.current_eef_position(source_obs)
    source_context = guided_common.make_guidance_context(
        source_policy,
        source_obs,
        centers,
        radii,
        scale=0.0,
        clearance_margin=0.0,
    )
    source_normalized, source_raw, _ = guided_common.sample_action_chunk(
        source_policy,
        source_obs,
        source_context,
        noise_seed=saved["noise_seed"],
    )
    maximum_difference = float(
        np.max(np.abs(source_raw - saved["baseline_actions"]))
    )
    if maximum_difference > 1e-3:
        raise RuntimeError(
            "Source sample does not reproduce captured baseline actions for {} "
            "(max abs difference {:.3e})".format(
                saved["state_id"], maximum_difference
            )
        )
    source_geometry = chunk_geometry(
        start_position, source_raw, target_position, centers, radii
    )
    required_progress = (
        source_geometry["progress_m"] - selector_config.progress_tolerance_m
        if selector_config.progress_rule == "source_relative"
        else -selector_config.progress_tolerance_m
    )

    candidates = [
        {
            "candidate_type": "source",
            "candidate_index": -1,
            "noise_seed": int(saved["noise_seed"]),
            "raw_action_chunk": source_raw,
            "predicted_progress_m": source_geometry["progress_m"],
            "predicted_clearance_m": source_geometry["clearance_m"],
            "required_progress_m": required_progress,
            "geometry_eligible": bool(
                source_geometry["progress_m"] >= required_progress
                and source_geometry["clearance_m"]
                >= selector_config.candidate_clearance_m
            ),
            "source_normalized_rms_deviation": 0.0,
        }
    ]

    corrective_obs = guided_common.policy_observation(
        saved["obs"], corrective_policy
    )
    corrective_context = guided_common.make_guidance_context(
        corrective_policy,
        corrective_obs,
        centers,
        radii,
        scale=0.0,
        clearance_margin=0.0,
    )
    source_scale = np.asarray(
        source_context.action_scale, dtype=np.float64
    ).reshape(-1)
    source_offset = np.asarray(
        source_context.action_offset, dtype=np.float64
    ).reshape(-1)
    seeds = [saved["noise_seed"]]
    seeds.extend(
        saved["noise_seed"] + 100_000 + index
        for index in range(candidate_count - 1)
    )
    for index, seed in enumerate(seeds):
        _, raw, _ = guided_common.sample_action_chunk(
            corrective_policy,
            corrective_obs,
            corrective_context,
            noise_seed=seed,
        )
        geometry = chunk_geometry(
            start_position, raw, target_position, centers, radii
        )
        source_normalized_candidate = (
            np.asarray(raw, dtype=np.float64)
            - source_offset.reshape(1, -1)
        ) / source_scale.reshape(1, -1)
        candidates.append(
            {
                "candidate_type": "corrective",
                "candidate_index": index,
                "noise_seed": int(seed),
                "raw_action_chunk": raw,
                "predicted_progress_m": geometry["progress_m"],
                "predicted_clearance_m": geometry["clearance_m"],
                "required_progress_m": required_progress,
                "geometry_eligible": bool(
                    geometry["progress_m"] >= required_progress
                    and geometry["clearance_m"]
                    >= selector_config.candidate_clearance_m
                ),
                "source_normalized_rms_deviation": (
                    normalized_chunk_deviation(
                        source_normalized_candidate, source_normalized
                    )
                ),
            }
        )
    return candidates, {
        "start_position": start_position,
        "target_position": target_position,
        "obstacle_names": obstacle_names,
        "obstacle_centers": centers,
        "obstacle_radii": radii,
        "source_predicted_progress_m": source_geometry["progress_m"],
        "source_predicted_clearance_m": source_geometry["clearance_m"],
        "source_capture_reproduction_max_abs": maximum_difference,
        "required_candidate_progress_m": required_progress,
    }


def branch_rollout(
    *,
    policy,
    env,
    saved,
    candidate,
    target_object_name,
    maximum_steps,
    continuation_seed,
    action_low,
    action_high,
    checkpoints,
):
    start_time = time.monotonic()
    obs = restore_policy_observation_history(env, saved)
    raw_env = guided_common.raw_env_from_wrapper(env)
    records = guided_common.active_object_records(raw_env)
    maps = guided_common.contact_maps(
        raw_env, records, target_object_name
    )
    target_start = guided_common.target_object_position(
        raw_env, records, target_object_name
    )
    metrics = teacher.new_metrics()
    trace = {
        "maximum_target_lift_m": 0.0,
        "maximum_target_displacement_m": 0.0,
        "checkpoints": {},
    }

    def snapshot(step):
        current_records = guided_common.active_object_records(raw_env)
        target = guided_common.target_object_position(
            raw_env, current_records, target_object_name
        )
        lift = float(target[2] - target_start[2])
        displacement = float(np.linalg.norm(target - target_start))
        trace["maximum_target_lift_m"] = max(
            trace["maximum_target_lift_m"], lift
        )
        trace["maximum_target_displacement_m"] = max(
            trace["maximum_target_displacement_m"], displacement
        )
        if step in checkpoints:
            trace["checkpoints"][str(step)] = {
                "step": step,
                "success": bool(metrics["success"]),
                "collision_any": bool(metrics["collision_any"]),
                "target_contact_any": bool(metrics["target_contact_any"]),
                "target_lift_m": lift,
                "target_displacement_m": displacement,
                "staged_rewards": guided_common.staged_task_progress(raw_env),
            }

    def transition_callback(**_):
        snapshot(metrics["executed_steps"])

    clip_delta = np.abs(
        np.clip(candidate["raw_action_chunk"], action_low, action_high)
        - candidate["raw_action_chunk"]
    )
    candidate_actions_executed = 0
    terminal = False
    for action in candidate["raw_action_chunk"]:
        if metrics["executed_steps"] >= maximum_steps:
            break
        obs, terminal = teacher.execute_action(
            env,
            raw_env,
            maps,
            obs,
            action,
            metrics,
            "teacher",
            "candidate_chunk",
            transition_callback=transition_callback,
        )
        candidate_actions_executed += 1
        if terminal:
            break

    candidate_target = guided_common.target_object_position(
        raw_env,
        guided_common.active_object_records(raw_env),
        target_object_name,
    )
    after_candidate = {
        "executed_actions": candidate_actions_executed,
        "success": bool(metrics["success"]),
        "collision_any": bool(metrics["collision_any"]),
        "target_contact_any": bool(metrics["target_contact_any"]),
        "target_lift_m": float(candidate_target[2] - target_start[2]),
        "target_displacement_m": float(
            np.linalg.norm(candidate_target - target_start)
        ),
        "staged_rewards": guided_common.staged_task_progress(raw_env),
    }

    if not terminal and metrics["executed_steps"] < maximum_steps:
        obs = teacher.execute_frozen_policy(
            policy,
            env,
            obs,
            raw_env,
            maps,
            metrics,
            target_object_name,
            maximum_steps,
            continuation_seed,
            transition_callback=transition_callback,
        )
    snapshot(metrics["executed_steps"])
    final = teacher.finalize_metrics(
        metrics, obs, raw_env, target_object_name
    )
    result = {
        key: (
            value.tolist() if isinstance(value, np.ndarray) else value
        )
        for key, value in candidate.items()
        if key != "raw_action_chunk"
    }
    result.update(
        {
            "raw_action_chunk": np.asarray(
                candidate["raw_action_chunk"], dtype=np.float32
            ),
            "success": bool(final["success"]),
            "collision_any": bool(final["collision_any"]),
            "target_contact_any": bool(final["target_contact_any"]),
            "outcome": outcome_label(
                final["success"], final["collision_any"]
            ),
            "executed_steps": int(final["executed_steps"]),
            "candidate_actions_executed": candidate_actions_executed,
            "continuation_steps": int(
                final["executed_steps"] - candidate_actions_executed
            ),
            "candidate_collision_any": bool(
                final["teacher_collision_any"]
            ),
            "continuation_collision_any": bool(
                final["policy_collision_any"]
            ),
            "collision_names": final["collision_names"],
            "first_collision_event": (
                final["collision_events"][0]
                if final["collision_events"]
                else None
            ),
            "after_candidate": after_candidate,
            "maximum_target_lift_m": trace["maximum_target_lift_m"],
            "maximum_target_displacement_m": trace[
                "maximum_target_displacement_m"
            ],
            "final_staged_rewards": final["final_staged_rewards"],
            "final_target_position": final["final_target_position"],
            "checkpoints": trace["checkpoints"],
            "action_clip_count": int(
                np.count_nonzero(clip_delta > 1e-5)
            ),
            "action_clip_max_abs": float(np.max(clip_delta)),
            "wall_time_s": time.monotonic() - start_time,
        }
    )
    return result


def local_selector_branch(branches):
    source = next(
        branch for branch in branches if branch["candidate_type"] == "source"
    )
    eligible = [
        branch
        for branch in branches
        if branch["candidate_type"] == "corrective"
        and branch["geometry_eligible"]
    ]
    if not eligible:
        return source
    return min(
        eligible,
        key=lambda branch: (
            branch["source_normalized_rms_deviation"],
            -branch["predicted_clearance_m"],
            -branch["predicted_progress_m"],
            branch["candidate_index"],
        ),
    )


def state_compact_summary(state):
    branches = state["branches"]
    source = next(
        branch for branch in branches if branch["candidate_type"] == "source"
    )
    local = local_selector_branch(branches)
    oracle_any = select_oracle_branch(branches)
    oracle_eligible = select_oracle_branch(branches, eligible_only=True)
    corrective = [
        branch for branch in branches if branch["candidate_type"] == "corrective"
    ]
    return {
        "state_id": state["state_id"],
        "source_outcome": source["outcome"],
        "local_selector_outcome": local["outcome"],
        "local_selector_index": local["candidate_index"],
        "oracle_any_outcome": oracle_any["outcome"],
        "oracle_any_index": oracle_any["candidate_index"],
        "oracle_eligible_outcome": oracle_eligible["outcome"],
        "oracle_eligible_index": oracle_eligible["candidate_index"],
        "corrective_success_branches": int(
            sum(branch["success"] for branch in corrective)
        ),
        "corrective_safe_success_branches": int(
            sum(
                branch["success"] and not branch["collision_any"]
                for branch in corrective
            )
        ),
        "eligible_branches": int(
            sum(branch["geometry_eligible"] for branch in corrective)
        ),
    }


def write_report(path, states, summary):
    lines = [
        "# Oracle Simulator-Branch Value Upper Bound",
        "",
        "Each branch executes one source or SDP action chunk from the same saved "
        "state, then hands control to the frozen source policy for the remaining "
        "episode horizon.",
        "",
        "## Aggregate",
        "",
        "- states: {}".format(summary["states"]),
        "- source outcomes: `{}`".format(summary["source_outcomes"]),
        "- oracle over all candidates: `{}`".format(
            summary["oracle_any_outcomes"]
        ),
        "- oracle over source + geometry-eligible candidates: `{}`".format(
            summary["oracle_geometry_eligible_outcomes"]
        ),
        "- states with any corrective success / safe success: {} / {}".format(
            summary["states_with_any_corrective_success"],
            summary["states_with_any_corrective_safe_success"],
        ),
        "- states with eligible corrective success / safe success: {} / {}".format(
            summary["states_with_eligible_corrective_success"],
            summary["states_with_eligible_corrective_safe_success"],
        ),
        "- oracle-any failure→success / success→failure: {} / {}".format(
            summary["oracle_any_failure_to_success"],
            summary["oracle_any_success_to_failure"],
        ),
        "- oracle-eligible failure→success / success→failure: {} / {}".format(
            summary["oracle_eligible_failure_to_success"],
            summary["oracle_eligible_success_to_failure"],
        ),
        "",
        "## Per state",
        "",
        "| state | source | local selector | oracle any | oracle eligible | "
        "corrective success / safe | eligible |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for state in states:
        row = state_compact_summary(state)
        lines.append(
            "| {state_id} | {source_outcome} | {local_selector_outcome} "
            "(#{local_selector_index}) | {oracle_any_outcome} "
            "(#{oracle_any_index}) | {oracle_eligible_outcome} "
            "(#{oracle_eligible_index}) | {corrective_success_branches} / "
            "{corrective_safe_success_branches} | {eligible_branches} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-agent", type=Path, default=guided_common.DEFAULT_AGENT
    )
    parser.add_argument(
        "--corrective-agent", type=Path, default=DEFAULT_CORRECTIVE_AGENT
    )
    parser.add_argument(
        "--states", nargs="+", type=Path, default=DEFAULT_STATE_BANKS
    )
    parser.add_argument("--state-limit", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=400)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--candidate-clearance", type=float, default=0.01)
    parser.add_argument("--progress-tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--progress-rule",
        choices=("nonretreat", "source_relative"),
        default="nonretreat",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[8, 32, 64, 128],
    )
    parser.add_argument(
        "--deterministic-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/oracle_branch_value/full_20"),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true")
    group.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.state_limit is not None and args.state_limit <= 0:
        raise ValueError("--state-limit must be positive")
    if args.candidate_count <= 0 or args.rollout_horizon <= 0:
        raise ValueError("Candidate count and rollout horizon must be positive")
    if any(step <= 0 for step in args.checkpoints):
        raise ValueError("Checkpoints must be positive")
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    if args.out_dir.exists() and not args.resume:
        if any(args.out_dir.iterdir()):
            raise FileExistsError(
                "{} exists; pass --overwrite or --resume".format(args.out_dir)
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    selector_config = SelectorConfig(
        candidate_clearance_m=args.candidate_clearance,
        progress_tolerance_m=args.progress_tolerance,
        progress_rule=args.progress_rule,
    )
    saved_states = load_saved_states(args.states, args.state_limit)
    manifest = {
        "source_agent": str(args.source_agent),
        "corrective_agent": str(args.corrective_agent),
        "states": [str(path) for path in args.states],
        "state_limit": args.state_limit,
        "candidate_count": args.candidate_count,
        "rollout_horizon": args.rollout_horizon,
        "selector": selector_config.to_dict(),
        "checkpoints": args.checkpoints,
        "continuation_seed_formula": "saved_noise_seed + 1 + continuation_chunk",
        "deterministic_torch": args.deterministic_torch,
    }
    manifest_path = args.out_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest != manifest:
            raise ValueError("Resume manifest does not match current arguments")
    else:
        json_dump(manifest_path, manifest)

    output_path = args.out_dir / "state_results.jsonl"
    completed = []
    completed_ids = set()
    if args.resume and output_path.exists():
        completed = [
            json.loads(line)
            for line in output_path.read_text().splitlines()
            if line.strip()
        ]
        completed_ids = {row["state_id"] for row in completed}

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if args.deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    source_policy, source_ckpt = guided_common.load_guided_policy_and_checkpoint(
        args.source_agent, device
    )
    corrective_policy, _ = guided_common.load_guided_policy_and_checkpoint(
        args.corrective_agent, device
    )
    envs = {}
    mode = "a" if args.resume else "w"
    try:
        with output_path.open(mode) as stream:
            for state_number, saved in enumerate(saved_states, start=1):
                if saved["state_id"] in completed_ids:
                    print(
                        "skip completed {}".format(saved["state_id"]), flush=True
                    )
                    continue
                if saved["env"] not in envs:
                    envs[saved["env"]], _ = FileUtils.env_from_checkpoint(
                        ckpt_dict=source_ckpt,
                        env_name=saved["env"],
                        render=False,
                        render_offscreen=False,
                        verbose=False,
                    )
                env = envs[saved["env"]]
                restore_policy_observation_history(env, saved)
                raw_env = guided_common.raw_env_from_wrapper(env)
                records = guided_common.active_object_records(raw_env)
                source_policy.start_episode()
                corrective_policy.start_episode()
                candidates, state_geometry = sample_candidates(
                    source_policy=source_policy,
                    corrective_policy=corrective_policy,
                    saved=saved,
                    raw_env=raw_env,
                    records=records,
                    target_object_name=args.target_object_name,
                    candidate_count=args.candidate_count,
                    selector_config=selector_config,
                )
                action_low, action_high = guided_common.action_limits(
                    env, source_policy.policy.ac_dim
                )
                maximum_steps = max(
                    1, args.rollout_horizon - saved["start_step"]
                )
                continuation_seed = saved["noise_seed"] + 1
                branches = []
                for branch_number, candidate in enumerate(candidates, start=1):
                    print(
                        "state {}/{} {} branch {}/{} type={} index={}".format(
                            state_number,
                            len(saved_states),
                            saved["state_id"],
                            branch_number,
                            len(candidates),
                            candidate["candidate_type"],
                            candidate["candidate_index"],
                        ),
                        flush=True,
                    )
                    branch = branch_rollout(
                        policy=source_policy,
                        env=env,
                        saved=saved,
                        candidate=candidate,
                        target_object_name=args.target_object_name,
                        maximum_steps=maximum_steps,
                        continuation_seed=continuation_seed,
                        action_low=action_low,
                        action_high=action_high,
                        checkpoints=set(args.checkpoints),
                    )
                    branches.append(branch)
                    print(
                        "  -> {} collision={} contact={} steps={} {:.2f}s".format(
                            branch["outcome"],
                            branch["collision_any"],
                            branch["target_contact_any"],
                            branch["executed_steps"],
                            branch["wall_time_s"],
                        ),
                        flush=True,
                    )
                state_result = {
                    "state_id": saved["state_id"],
                    "env": saved["env"],
                    "seed": saved["seed"],
                    "episode": saved["episode"],
                    "chunk": saved["chunk"],
                    "start_step": saved["start_step"],
                    "noise_seed": saved["noise_seed"],
                    "maximum_steps": maximum_steps,
                    "state_geometry": state_geometry,
                    "branches": branches,
                }
                stream.write(
                    json.dumps(guided_common.json_safe(state_result)) + "\n"
                )
                stream.flush()
                completed.append(guided_common.json_safe(state_result))
                summary = summarize_states(completed)
                json_dump(args.out_dir / "summary.json", summary)
                write_report(args.out_dir / "report.md", completed, summary)
                compact = state_compact_summary(state_result)
                print(
                    "completed {}: source={} local={} oracle_any={} "
                    "oracle_eligible={} successes={}".format(
                        saved["state_id"],
                        compact["source_outcome"],
                        compact["local_selector_outcome"],
                        compact["oracle_any_outcome"],
                        compact["oracle_eligible_outcome"],
                        compact["corrective_success_branches"],
                    ),
                    flush=True,
                )
    finally:
        for env in envs.values():
            close = getattr(
                guided_common.raw_env_from_wrapper(env), "close", None
            )
            if close is not None:
                close()

    summary = summarize_states(completed)
    json_dump(args.out_dir / "summary.json", summary)
    write_report(args.out_dir / "report.md", completed, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
