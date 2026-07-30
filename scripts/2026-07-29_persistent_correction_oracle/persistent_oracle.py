"""Oracle temporal-handoff test for persistent SDP correction.

The corrective policy is rolled closed loop for multiple consecutive chunks.
At every collision-free chunk boundary, an exact simulator clone is handed to
the frozen source policy. A hindsight oracle then asks whether any temporal
handoff completes safely.
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_DIR = SCRIPT_DIR.parent / "guided_denoising"
MANIFOLD_DIR = SCRIPT_DIR.parent / "2026-07-27_policy_manifold_diagnostic"
TEACHER_DIR = SCRIPT_DIR.parent / "2026-07-27_semi_manual_waypoint_teacher"
BRANCH_DIR = SCRIPT_DIR.parent / "2026-07-29_oracle_branch_value"
for directory in (
    SCRIPT_DIR,
    GUIDED_DIR,
    MANIFOLD_DIR,
    TEACHER_DIR,
    BRANCH_DIR,
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common as guided_common  # noqa: E402
import evaluate_waypoint_teacher as teacher  # noqa: E402
from branch_value import outcome_label  # noqa: E402
from run_fixed_state import DEFAULT_STATE_BANKS  # noqa: E402
from simulator_branch_eval import (  # noqa: E402
    DEFAULT_CORRECTIVE_AGENT,
    load_saved_states,
    restore_policy_observation_history,
)
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


def correction_seed(saved_noise_seed, trajectory_index, chunk_index):
    return (
        int(saved_noise_seed)
        + 10_000_000
        + int(trajectory_index) * 100_000
        + int(chunk_index)
    )


def continuation_seed(saved_noise_seed):
    return int(saved_noise_seed) + 1


def select_saved_states(paths, state_offset, state_limit):
    requested = state_offset + state_limit
    states = load_saved_states(paths, requested)
    selected = states[state_offset:requested]
    if len(selected) != state_limit:
        raise ValueError(
            "Requested states [{}:{}) but only {} states are available".format(
                state_offset,
                requested,
                len(states),
            )
        )
    return selected


def earliest_safe_handoff(trajectory):
    eligible = [
        boundary
        for boundary in trajectory["boundaries"]
        if boundary["safe_recoverable_handoff"]
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item["correction_actions"],
            item["absolute_handoff_step"],
        ),
    )


def summarize_states(states):
    handoff_outcomes = Counter()
    states_with_safe_handoff = 0
    trajectories_with_safe_handoff = 0
    safe_handoffs = 0
    total_boundaries = 0
    direct_safe_successes = 0
    for state in states:
        state_has_safe_handoff = False
        for trajectory in state["trajectories"]:
            selected = earliest_safe_handoff(trajectory)
            if selected is not None:
                trajectories_with_safe_handoff += 1
                state_has_safe_handoff = True
            direct_safe_successes += int(
                trajectory["direct_success"]
                and not trajectory["correction_collision_any"]
            )
            for boundary in trajectory["boundaries"]:
                total_boundaries += 1
                safe_handoffs += int(
                    boundary["safe_recoverable_handoff"]
                )
                handoff_outcomes[boundary["continuation"]["outcome"]] += 1
        states_with_safe_handoff += int(state_has_safe_handoff)
    return {
        "states": len(states),
        "trajectories": sum(
            len(state["trajectories"]) for state in states
        ),
        "total_safe_prefix_boundaries": total_boundaries,
        "safe_recoverable_handoffs": safe_handoffs,
        "states_with_safe_recoverable_handoff": states_with_safe_handoff,
        "trajectories_with_safe_recoverable_handoff": (
            trajectories_with_safe_handoff
        ),
        "direct_safe_correction_successes": direct_safe_successes,
        "continuation_outcomes": dict(sorted(handoff_outcomes.items())),
        "pilot_pass": bool(states_with_safe_handoff >= 1),
    }


def json_dump(path, payload):
    path.write_text(
        json.dumps(guided_common.json_safe(payload), indent=2) + "\n"
    )


def clone_boundary(env, obs, source_policy, absolute_step):
    return {
        "state": env.get_state(),
        "obs": guided_common.policy_observation(obs, source_policy),
        "start_step": int(absolute_step),
    }


def rollout_correction_trajectory(
    *,
    env,
    saved,
    source_policy,
    corrective_policy,
    target_object_name,
    trajectory_index,
    maximum_chunks,
):
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
    corrective_policy.start_episode()
    action_low, action_high = guided_common.action_limits(
        env, corrective_policy.policy.ac_dim
    )
    clip_count = 0
    clip_max = 0.0
    boundaries = []
    terminal = False

    for chunk_index in range(maximum_chunks):
        records = guided_common.active_object_records(raw_env)
        centers, radii, _ = guided_common.oracle_obstacle_geometry(
            raw_env, records, target_object_name
        )
        policy_obs = guided_common.policy_observation(
            obs, corrective_policy
        )
        context = guided_common.make_guidance_context(
            corrective_policy,
            policy_obs,
            centers,
            radii,
            scale=0.0,
            clearance_margin=0.0,
        )
        seed = correction_seed(
            saved["noise_seed"], trajectory_index, chunk_index
        )
        _, actions, _ = guided_common.sample_action_chunk(
            corrective_policy,
            policy_obs,
            context,
            noise_seed=seed,
        )
        clipped = np.clip(actions, action_low, action_high)
        clip_delta = np.abs(clipped - actions)
        clip_count += int(np.count_nonzero(clip_delta > 1e-5))
        clip_max = max(clip_max, float(np.max(clip_delta)))

        for action in clipped:
            obs, terminal = teacher.execute_action(
                env,
                raw_env,
                maps,
                obs,
                action,
                metrics,
                "teacher",
                "persistent_correction",
            )
            if terminal or metrics["collision_any"]:
                break

        absolute_step = saved["start_step"] + metrics["executed_steps"]
        target = guided_common.target_object_position(
            raw_env,
            guided_common.active_object_records(raw_env),
            target_object_name,
        )
        compact = {
            "chunk": chunk_index + 1,
            "seed": seed,
            "correction_actions": int(metrics["executed_steps"]),
            "absolute_handoff_step": int(absolute_step),
            "prefix_target_contact_any": bool(
                metrics["target_contact_any"]
            ),
            "prefix_collision_any": bool(metrics["collision_any"]),
            "prefix_success": bool(metrics["success"]),
            "target_lift_m": float(target[2] - target_start[2]),
            "target_displacement_m": float(
                np.linalg.norm(target - target_start)
            ),
            "staged_rewards": guided_common.staged_task_progress(raw_env),
        }
        if not metrics["collision_any"]:
            compact["_snapshot"] = clone_boundary(
                env, obs, source_policy, absolute_step
            )
            boundaries.append(compact)
        if terminal or metrics["collision_any"]:
            break

    return {
        "trajectory_index": int(trajectory_index),
        "correction_actions": int(metrics["executed_steps"]),
        "correction_chunks": len(boundaries),
        "correction_collision_any": bool(metrics["collision_any"]),
        "correction_target_contact_any": bool(
            metrics["target_contact_any"]
        ),
        "direct_success": bool(metrics["success"]),
        "collision_names": sorted(metrics["collision_names"]),
        "action_clip_count": clip_count,
        "action_clip_max_abs": clip_max,
        "boundaries": boundaries,
    }


def evaluate_source_continuation(
    *,
    env,
    snapshot,
    source_policy,
    target_object_name,
    rollout_horizon,
    noise_seed,
):
    start_time = time.monotonic()
    obs = restore_policy_observation_history(env, snapshot)
    raw_env = guided_common.raw_env_from_wrapper(env)
    records = guided_common.active_object_records(raw_env)
    maps = guided_common.contact_maps(
        raw_env, records, target_object_name
    )
    metrics = teacher.new_metrics()
    remaining_steps = max(
        0, int(rollout_horizon) - int(snapshot["start_step"])
    )
    if remaining_steps > 0:
        obs = teacher.execute_frozen_policy(
            source_policy,
            env,
            obs,
            raw_env,
            maps,
            metrics,
            target_object_name,
            remaining_steps,
            noise_seed,
        )
    final = teacher.finalize_metrics(
        metrics, obs, raw_env, target_object_name
    )
    absolute_end_step = snapshot["start_step"] + final["executed_steps"]
    return {
        "success": bool(final["success"]),
        "collision_any": bool(final["collision_any"]),
        "target_contact_any": bool(final["target_contact_any"]),
        "outcome": outcome_label(
            final["success"], final["collision_any"]
        ),
        "continuation_actions": int(final["executed_steps"]),
        "absolute_end_step": int(absolute_end_step),
        "collision_names": final["collision_names"],
        "final_staged_rewards": final["final_staged_rewards"],
        "wall_time_s": time.monotonic() - start_time,
    }


def evaluate_state(
    *,
    env,
    saved,
    source_policy,
    corrective_policy,
    target_object_name,
    trajectory_count,
    maximum_correction_chunks,
    rollout_horizon,
):
    trajectories = []
    for trajectory_index in range(trajectory_count):
        print(
            "  correction trajectory {}/{}".format(
                trajectory_index + 1, trajectory_count
            ),
            flush=True,
        )
        trajectory = rollout_correction_trajectory(
            env=env,
            saved=saved,
            source_policy=source_policy,
            corrective_policy=corrective_policy,
            target_object_name=target_object_name,
            trajectory_index=trajectory_index,
            maximum_chunks=maximum_correction_chunks,
        )
        for boundary in trajectory["boundaries"]:
            snapshot = boundary.pop("_snapshot")
            continuation = evaluate_source_continuation(
                env=env,
                snapshot=snapshot,
                source_policy=source_policy,
                target_object_name=target_object_name,
                rollout_horizon=rollout_horizon,
                noise_seed=continuation_seed(saved["noise_seed"]),
            )
            boundary["continuation"] = continuation
            boundary["safe_recoverable_handoff"] = bool(
                not boundary["prefix_collision_any"]
                and continuation["success"]
                and not continuation["collision_any"]
            )
            print(
                "    chunk {:02d} step={} -> {} safe_handoff={} {:.2f}s".format(
                    boundary["chunk"],
                    boundary["absolute_handoff_step"],
                    continuation["outcome"],
                    boundary["safe_recoverable_handoff"],
                    continuation["wall_time_s"],
                ),
                flush=True,
            )
        trajectories.append(trajectory)
    return {
        "state_id": saved["state_id"],
        "env": saved["env"],
        "seed": saved["seed"],
        "episode": saved["episode"],
        "chunk": saved["chunk"],
        "start_step": saved["start_step"],
        "noise_seed": saved["noise_seed"],
        "trajectories": trajectories,
    }


def state_report_row(state):
    selected = []
    for trajectory in state["trajectories"]:
        handoff = earliest_safe_handoff(trajectory)
        if handoff is not None:
            selected.append(
                (
                    handoff["correction_actions"],
                    trajectory["trajectory_index"],
                    handoff,
                )
            )
    earliest = min(selected) if selected else None
    return {
        "state_id": state["state_id"],
        "trajectories": len(state["trajectories"]),
        "boundaries": sum(
            len(item["boundaries"]) for item in state["trajectories"]
        ),
        "trajectories_with_safe_handoff": sum(
            earliest_safe_handoff(item) is not None
            for item in state["trajectories"]
        ),
        "earliest_trajectory": earliest[1] if earliest else None,
        "earliest_chunk": earliest[2]["chunk"] if earliest else None,
        "earliest_actions": earliest[0] if earliest else None,
        "earliest_absolute_step": (
            earliest[2]["absolute_handoff_step"] if earliest else None
        ),
    }


def compact_chunk_ranges(chunks):
    if not chunks:
        return "--"
    ranges = []
    start = previous = chunks[0]
    for chunk in chunks[1:]:
        if chunk == previous + 1:
            previous = chunk
            continue
        ranges.append(
            str(start) if start == previous else "{}-{}".format(start, previous)
        )
        start = previous = chunk
    ranges.append(
        str(start) if start == previous else "{}-{}".format(start, previous)
    )
    return ", ".join(ranges)


def write_report(path, states, summary):
    title = (
        "# Persistent-Correction Oracle: Full 20-State Evaluation"
        if len(states) == 20
        else "# Persistent-Correction Oracle Pilot"
    )
    decision_label = (
        "existence-gate decision"
        if len(states) == 20
        else "pilot decision"
    )
    lines = [
        title,
        "",
        "The existing SDP-6k policy is executed for multiple consecutive "
        "closed-loop chunks. Every collision-free chunk boundary is cloned and "
        "handed to the frozen source policy.",
        "",
        "## Aggregate",
        "",
        "- states / trajectories: {} / {}".format(
            summary["states"], summary["trajectories"]
        ),
        "- safe-prefix temporal boundaries: {}".format(
            summary["total_safe_prefix_boundaries"]
        ),
        "- safe recoverable handoffs: {}".format(
            summary["safe_recoverable_handoffs"]
        ),
        "- states with a safe recoverable handoff: {}".format(
            summary["states_with_safe_recoverable_handoff"]
        ),
        "- trajectories with a safe recoverable handoff: {}".format(
            summary["trajectories_with_safe_recoverable_handoff"]
        ),
        "- direct safe corrective successes: {}".format(
            summary["direct_safe_correction_successes"]
        ),
        "- source-continuation outcomes: `{}`".format(
            summary["continuation_outcomes"]
        ),
        "- {}: **{}**".format(
            decision_label,
            "PASS" if summary["pilot_pass"] else "FAIL",
        ),
        "",
        "## Per state",
        "",
        "| state | trajectories | boundaries | trajectories with safe handoff | "
        "earliest trajectory / chunk / actions / absolute step |",
        "|---|---:|---:|---:|---|",
    ]
    for state in states:
        row = state_report_row(state)
        earliest = (
            "{} / {} / {} / {}".format(
                row["earliest_trajectory"],
                row["earliest_chunk"],
                row["earliest_actions"],
                row["earliest_absolute_step"],
            )
            if row["earliest_chunk"] is not None
            else "--"
        )
        lines.append(
            "| {state_id} | {trajectories} | {boundaries} | "
            "{trajectories_with_safe_handoff} | {earliest} |".format(
                earliest=earliest, **row
            )
        )
    lines.extend(
        [
            "",
            "Trajectory indices below are zero-based. Counts over temporal "
            "boundaries are descriptive, not independent trials.",
            "",
            "## Recoverability windows",
            "",
            "| state | trajectory | safe chunk windows | correction ending |",
            "|---|---:|---|---|",
        ]
    )
    for state_index, state in enumerate(states):
        for trajectory in state["trajectories"]:
            safe_chunks = [
                boundary["chunk"]
                for boundary in trajectory["boundaries"]
                if boundary["safe_recoverable_handoff"]
            ]
            ending = (
                "collision after {} safe boundaries".format(
                    len(trajectory["boundaries"])
                )
                if trajectory["correction_collision_any"]
                else "maximum duration"
            )
            lines.append(
                "| {} | {} | {} | {} |".format(
                    state_index,
                    trajectory["trajectory_index"],
                    compact_chunk_ranges(safe_chunks),
                    ending,
                )
            )
    lines.extend(
        [
            "",
            "No corrective trajectory directly completed the task. At the "
            "earliest safe handoff of each successful trajectory, the "
            "correction prefix had not contacted the target; persistence "
            "changed the robot-scene configuration before the frozen source "
            "completed the task.",
            "",
            "The windows are non-monotonic: continuing correction can leave "
            "and later re-enter source recoverability. A fixed correction "
            "duration therefore cannot implement the oracle.",
            "",
            "A pass is an existence result for temporal persistence and an "
            "oracle handoff. It is not evidence that a learned recoverability "
            "critic is ready.",
        ]
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
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--state-limit", type=int, default=3)
    parser.add_argument("--trajectory-count", type=int, default=4)
    parser.add_argument("--maximum-correction-chunks", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=600)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument(
        "--deterministic-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/persistent_correction_oracle/pilot_3state"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.state_offset < 0:
        raise ValueError("state_offset must be non-negative")
    for name in (
        "state_limit",
        "trajectory_count",
        "maximum_correction_chunks",
        "rollout_horizon",
    ):
        if getattr(args, name) <= 0:
            raise ValueError("{} must be positive".format(name))
    if args.out_dir.exists():
        if not args.overwrite and any(args.out_dir.iterdir()):
            raise FileExistsError(
                "{} exists; pass --overwrite".format(args.out_dir)
            )
        if args.overwrite:
            shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    states = select_saved_states(
        args.states, args.state_offset, args.state_limit
    )
    manifest = {
        "source_agent": str(args.source_agent),
        "corrective_agent": str(args.corrective_agent),
        "states": [str(path) for path in args.states],
        "state_offset": args.state_offset,
        "state_limit": args.state_limit,
        "trajectory_count": args.trajectory_count,
        "maximum_correction_chunks": args.maximum_correction_chunks,
        "maximum_correction_actions": args.maximum_correction_chunks * 8,
        "rollout_horizon": args.rollout_horizon,
        "target_object_name": args.target_object_name,
        "correction_seed_formula": (
            "saved_noise_seed + 10000000 + trajectory_index * 100000 "
            "+ chunk_index"
        ),
        "continuation_seed_formula": "saved_noise_seed + 1",
        "deterministic_torch": args.deterministic_torch,
        "gate": (
            "pass iff at least one state has a collision-free correction "
            "prefix and collision-free successful source continuation"
        ),
    }
    json_dump(args.out_dir / "manifest.json", manifest)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if args.deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    source_policy, source_ckpt = (
        guided_common.load_guided_policy_and_checkpoint(
            args.source_agent, device
        )
    )
    corrective_policy, _ = (
        guided_common.load_guided_policy_and_checkpoint(
            args.corrective_agent, device
        )
    )
    envs = {}
    results = []
    output_path = args.out_dir / "state_results.jsonl"
    try:
        with output_path.open("w") as stream:
            for index, saved in enumerate(states, start=1):
                print(
                    "state {}/{} {}".format(
                        index, len(states), saved["state_id"]
                    ),
                    flush=True,
                )
                if saved["env"] not in envs:
                    envs[saved["env"]], _ = FileUtils.env_from_checkpoint(
                        ckpt_dict=source_ckpt,
                        env_name=saved["env"],
                        render=False,
                        render_offscreen=False,
                        verbose=False,
                    )
                result = evaluate_state(
                    env=envs[saved["env"]],
                    saved=saved,
                    source_policy=source_policy,
                    corrective_policy=corrective_policy,
                    target_object_name=args.target_object_name,
                    trajectory_count=args.trajectory_count,
                    maximum_correction_chunks=(
                        args.maximum_correction_chunks
                    ),
                    rollout_horizon=args.rollout_horizon,
                )
                safe_result = guided_common.json_safe(result)
                stream.write(json.dumps(safe_result) + "\n")
                stream.flush()
                results.append(safe_result)
                summary = summarize_states(results)
                json_dump(args.out_dir / "summary.json", summary)
                write_report(args.out_dir / "report.md", results, summary)
                print(
                    "  state safe-handoff trajectories: {}".format(
                        state_report_row(result)[
                            "trajectories_with_safe_handoff"
                        ]
                    ),
                    flush=True,
                )
    finally:
        for env in envs.values():
            raw_env = guided_common.raw_env_from_wrapper(env)
            close_fn = getattr(raw_env, "close", None)
            if close_fn is not None:
                close_fn()

    summary = summarize_states(results)
    json_dump(args.out_dir / "summary.json", summary)
    write_report(args.out_dir / "report.md", results, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
