"""Paired source-only versus gated corrective-policy rollout evaluation."""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_DIR = SCRIPT_DIR.parent / "guided_denoising"
for directory in (SCRIPT_DIR, GUIDED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common as guided_common  # noqa: E402
from runtime import choose_action_chunk  # noqa: E402
from selector import (  # noqa: E402
    SelectorConfig,
    chunk_geometry,
    chunk_noise_seeds,
    point_clearance_xy,
)
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


DEFAULT_CORRECTIVE_AGENT = Path(
    "outputs/robomimic/train/dp_can_delta_pose_osc_targeted_sdp/"
    "20260729173254/models/model_epoch_12.pth"
)
DEFAULT_ENVS = ["PickPlaceCan", "PickPlaceBreadCerealCan"]


def state_hash(state):
    digest = hashlib.sha256()
    digest.update(np.asarray(state["states"]).tobytes())
    digest.update(state["model"].encode("utf-8"))
    digest.update(state.get("ep_meta", "").encode("utf-8"))
    return digest.hexdigest()


def compact_chunk_diagnostic(diagnostic):
    return {
        key: value
        for key, value in diagnostic.items()
        if key not in ("candidate_metrics", "candidate_seeds")
    }


def source_action_chunk(
    policy,
    obs,
    raw_env,
    records,
    target_object_name,
    nominal_seed,
):
    centers, radii, obstacle_names = guided_common.oracle_obstacle_geometry(
        raw_env, records, target_object_name
    )
    target_position = guided_common.target_object_position(
        raw_env, records, target_object_name
    )
    policy_obs = guided_common.policy_observation(obs, policy)
    start_position = guided_common.current_eef_position(policy_obs)
    context = guided_common.make_guidance_context(
        policy,
        policy_obs,
        centers,
        radii,
        scale=0.0,
        clearance_margin=0.0,
    )
    queue_length = len(policy.policy.action_queue)
    _, raw, _ = guided_common.sample_action_chunk(
        policy, policy_obs, context, noise_seed=nominal_seed
    )
    if len(policy.policy.action_queue) != queue_length:
        raise RuntimeError("Source sampling mutated its rollout action queue")
    geometry = chunk_geometry(
        start_position, raw, target_position, centers, radii
    )
    return raw, {
        "mode": "source",
        "nominal_seed": int(nominal_seed),
        "obstacle_names": obstacle_names,
        "start_clearance_m": point_clearance_xy(
            start_position, centers, radii
        ),
        "nominal_progress_m": geometry["progress_m"],
        "nominal_clearance_m": geometry["clearance_m"],
        "triggered": False,
        "eligible_candidate_count": 0,
        "selected_candidate_index": None,
        "selected_candidate_seed": None,
        "selected_progress_m": geometry["progress_m"],
        "selected_clearance_m": geometry["clearance_m"],
        "selected_normalized_rms_deviation": 0.0,
    }


def rollout_condition(
    *,
    source_policy,
    corrective_policy,
    env,
    initial_state,
    condition,
    policy_seed,
    args,
    config,
):
    source_policy.start_episode()
    corrective_policy.start_episode()
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    obs = env.reset_to(initial_state)
    raw_env = guided_common.raw_env_from_wrapper(env)
    records = guided_common.active_object_records(raw_env)
    maps = guided_common.contact_maps(
        raw_env, records, args.target_object_name
    )
    target_start = guided_common.target_object_position(
        raw_env, records, args.target_object_name
    )
    low, high = guided_common.action_limits(env, source_policy.policy.ac_dim)

    step = 0
    chunk_index = 0
    success = guided_common.task_success(env)
    collision_any = False
    collision_names = set()
    target_contact_any = False
    clip_count = 0
    clip_max = 0.0
    min_clearance = float("inf")
    reconstruction_errors = []
    chunk_diagnostics = []
    observation_failure = False
    observation_failure_reason = None
    done = False

    while step < args.horizon and not success:
        valid, error = guided_common.validate_policy_observation(obs)
        if not valid:
            observation_failure = True
            observation_failure_reason = error
            break
        nominal_seed, candidate_seeds = chunk_noise_seeds(
            policy_seed, chunk_index, args.candidate_count
        )
        start_position = guided_common.current_eef_position(obs)
        if condition == "source":
            raw_chunk, diagnostic = source_action_chunk(
                source_policy,
                obs,
                raw_env,
                records,
                args.target_object_name,
                nominal_seed,
            )
        else:
            raw_chunk, diagnostic = choose_action_chunk(
                common=guided_common,
                source_policy=source_policy,
                corrective_policy=corrective_policy,
                obs=obs,
                raw_env=raw_env,
                object_records=records,
                target_object_name=args.target_object_name,
                nominal_seed=nominal_seed,
                candidate_seeds=candidate_seeds,
                target_contact_seen=target_contact_any,
                obstacle_contact_seen=collision_any,
                config=config,
            )
        predicted_positions = guided_common.predict_eef_positions(
            start_position, raw_chunk
        )
        clip_delta = np.abs(np.clip(raw_chunk, low, high) - raw_chunk)
        clip_count += int(np.count_nonzero(clip_delta > 1e-5))
        clip_max = max(clip_max, float(np.max(clip_delta)))
        diagnostic = {
            "chunk": chunk_index,
            "start_step": step,
            **compact_chunk_diagnostic(diagnostic),
        }

        actual_positions = []
        chunk_observation_failure = False
        for action in raw_chunk:
            if step >= args.horizon:
                break
            try:
                obs, _, done, _ = env.step(action)
                position = guided_common.current_eef_position(obs)
            except (RuntimeError, AssertionError, ValueError) as exc:
                obs = None
                done = False
                position = guided_common.current_raw_eef_position(raw_env)
                chunk_observation_failure = True
                observation_failure = True
                observation_failure_reason = str(exc) or type(exc).__name__
            actual_positions.append(position)
            centers, radii, _ = guided_common.oracle_obstacle_geometry(
                raw_env, records, args.target_object_name
            )
            min_clearance = min(
                min_clearance,
                guided_common.minimum_clearance(position, centers, radii),
            )
            target_contacts, obstacle_contacts, names = (
                guided_common.contact_snapshot(raw_env, maps)
            )
            target_contact_any = target_contact_any or target_contacts > 0
            collision_any = collision_any or obstacle_contacts > 0
            collision_names.update(names)
            success = success or guided_common.task_success(env)
            step += 1
            if done or (args.terminate_on_success and success):
                break

        actual_positions = np.asarray(actual_positions, dtype=np.float32)
        if len(actual_positions):
            reconstruction_errors.extend(
                np.linalg.norm(
                    predicted_positions[: len(actual_positions)] - actual_positions,
                    axis=1,
                ).tolist()
            )
        diagnostic["executed_actions"] = len(actual_positions)
        diagnostic["collision_after_chunk"] = collision_any
        diagnostic["target_contact_after_chunk"] = target_contact_any
        chunk_diagnostics.append(diagnostic)
        chunk_index += 1
        if (
            len(actual_positions) == 0
            or done
            or chunk_observation_failure
            or (args.terminate_on_success and success)
        ):
            break

    target_end = guided_common.target_object_position(
        raw_env, records, args.target_object_name
    )
    triggered = sum(row["triggered"] for row in chunk_diagnostics)
    selected = sum(row["mode"] == "corrective" for row in chunk_diagnostics)
    fallback = sum(
        row["triggered"] and row["mode"] == "source"
        for row in chunk_diagnostics
    )
    safe_success = success and not collision_any
    success_with_collision = success and collision_any
    collision_failure = (not success) and collision_any
    ncr = (not success) and (not collision_any)
    return {
        "condition": condition,
        "success": success,
        "horizon": step,
        "chunks": chunk_index,
        "collision_any": collision_any,
        "collision_names": sorted(collision_names),
        "target_contact_any": target_contact_any,
        "safe_success": safe_success,
        "success_with_collision": success_with_collision,
        "collision_failure": collision_failure,
        "ncr": ncr,
        "min_eef_obstacle_clearance_m": min_clearance,
        "target_displacement_m": float(np.linalg.norm(target_end - target_start)),
        "action_clip_count": clip_count,
        "action_clip_max_abs": clip_max,
        "triggered_chunks": triggered,
        "corrective_chunks": selected,
        "fallback_chunks": fallback,
        "trigger_rate": triggered / chunk_index if chunk_index else 0.0,
        "selection_rate_given_trigger": selected / triggered if triggered else 0.0,
        "observation_failure": observation_failure,
        "observation_failure_reason": observation_failure_reason,
        "trajectory_reconstruction_error_mean_cm": (
            float(np.mean(reconstruction_errors) * 100.0)
            if reconstruction_errors
            else None
        ),
        "trajectory_reconstruction_error_max_cm": (
            float(np.max(reconstruction_errors) * 100.0)
            if reconstruction_errors
            else None
        ),
        "chunk_diagnostics": chunk_diagnostics,
    }


def outcome(row):
    for name in (
        "safe_success",
        "success_with_collision",
        "collision_failure",
        "ncr",
    ):
        if row[name]:
            return name
    raise ValueError("Episode has no four-way outcome")


def aggregate(rows):
    summaries = []
    transitions = []
    for env_name in sorted({row["env"] for row in rows}):
        for condition in ("source", "gated"):
            selected = [
                row
                for row in rows
                if row["env"] == env_name and row["condition"] == condition
            ]
            summaries.append(
                {
                    "env": env_name,
                    "condition": condition,
                    "episodes": len(selected),
                    "task_sr": float(np.mean([row["success"] for row in selected])),
                    "safe_sr": float(
                        np.mean([row["safe_success"] for row in selected])
                    ),
                    "cr": float(
                        np.mean([row["collision_any"] for row in selected])
                    ),
                    "ncr": float(np.mean([row["ncr"] for row in selected])),
                    "safe_success": int(sum(row["safe_success"] for row in selected)),
                    "success_with_collision": int(
                        sum(row["success_with_collision"] for row in selected)
                    ),
                    "collision_failure": int(
                        sum(row["collision_failure"] for row in selected)
                    ),
                    "collision_free_non_completion": int(
                        sum(row["ncr"] for row in selected)
                    ),
                    "triggered_chunks": int(
                        sum(row["triggered_chunks"] for row in selected)
                    ),
                    "corrective_chunks": int(
                        sum(row["corrective_chunks"] for row in selected)
                    ),
                    "fallback_chunks": int(
                        sum(row["fallback_chunks"] for row in selected)
                    ),
                    "action_clip_count": int(
                        sum(row["action_clip_count"] for row in selected)
                    ),
                    "observation_failure_count": int(
                        sum(row["observation_failure"] for row in selected)
                    ),
                }
            )
        source_by_key = {
            (row["seed"], row["episode"]): row
            for row in rows
            if row["env"] == env_name and row["condition"] == "source"
        }
        gated_by_key = {
            (row["seed"], row["episode"]): row
            for row in rows
            if row["env"] == env_name and row["condition"] == "gated"
        }
        if source_by_key.keys() != gated_by_key.keys():
            raise ValueError("Paired episode keys differ")
        counts = Counter(
            (outcome(source_by_key[key]), outcome(gated_by_key[key]))
            for key in source_by_key
        )
        transitions.extend(
            {
                "env": env_name,
                "source_outcome": before,
                "gated_outcome": after,
                "count": count,
            }
            for (before, after), count in sorted(counts.items())
        )
    return {"conditions": summaries, "paired_transitions": transitions}


def write_report(path, result):
    lines = [
        "# Gated Support Selector Paired Rollout",
        "",
        "The four-way partition is safe success / success with collision / "
        "collision failure / collision-free non-completion.",
        "",
        "| env | condition | n | Task SR | Safe SR | CR | NCR | "
        "triggered / corrective / fallback chunks |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["conditions"]:
        lines.append(
            "| {env} | {condition} | {episodes} | {task_sr:.3f} | "
            "{safe_sr:.3f} | {cr:.3f} | {ncr:.3f} | "
            "{triggered_chunks} / {corrective_chunks} / {fallback_chunks} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Paired transitions",
            "",
            "| env | source | gated | count |",
            "|---|---|---|---:|",
        ]
    )
    for row in result["paired_transitions"]:
        lines.append(
            "| {env} | {source_outcome} | {gated_outcome} | {count} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-agent", type=Path, default=guided_common.DEFAULT_AGENT)
    parser.add_argument(
        "--corrective-agent", type=Path, default=DEFAULT_CORRECTIVE_AGENT
    )
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--n-rollouts", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--trigger-clearance", type=float, default=0.01)
    parser.add_argument("--minimum-start-clearance", type=float, default=0.01)
    parser.add_argument("--candidate-clearance", type=float, default=0.01)
    parser.add_argument("--progress-tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--progress-rule",
        choices=("nonretreat", "source_relative"),
        default="nonretreat",
    )
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument(
        "--terminate-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/gated_support_selector/paired_pilot"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.candidate_count <= 0 or args.n_rollouts <= 0:
        raise ValueError("Candidate count and rollout count must be positive")
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError("{} exists; pass --overwrite".format(args.out_dir))
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    if args.deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    config = SelectorConfig(
        trigger_clearance_m=args.trigger_clearance,
        minimum_start_clearance_m=args.minimum_start_clearance,
        candidate_clearance_m=args.candidate_clearance,
        progress_tolerance_m=args.progress_tolerance,
        progress_rule=args.progress_rule,
    )
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if args.deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # GuidedDiffusionPolicy evaluates a CUDA cumsum even at guidance scale
        # zero. PyTorch does not label that kernel deterministic, although the
        # sampling path is explicitly seeded, so preserve the other deterministic
        # settings and report unsupported kernels instead of aborting.
        torch.use_deterministic_algorithms(True, warn_only=True)
    source_policy, source_ckpt = guided_common.load_guided_policy_and_checkpoint(
        args.source_agent, device
    )
    corrective_policy, _ = guided_common.load_guided_policy_and_checkpoint(
        args.corrective_agent, device
    )
    if (
        source_policy.policy.global_config.all_obs_keys
        != corrective_policy.policy.global_config.all_obs_keys
    ):
        raise ValueError("Source and corrective policies require different observations")

    rows = []
    with (args.out_dir / "episode_metrics.jsonl").open("w") as stream:
        for env_name in args.envs:
            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=source_ckpt,
                env_name=env_name,
                render=False,
                render_offscreen=False,
                verbose=False,
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
                        for condition in ("source", "gated"):
                            metrics = rollout_condition(
                                source_policy=source_policy,
                                corrective_policy=corrective_policy,
                                env=env,
                                initial_state=initial_state,
                                condition=condition,
                                policy_seed=policy_seed,
                                args=args,
                                config=config,
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
                            stream.write(
                                json.dumps(guided_common.json_safe(row)) + "\n"
                            )
                            stream.flush()
                            print(
                                "{} seed={} ep={} {} success={} collision={} "
                                "triggered={} selected={}".format(
                                    env_name,
                                    seed,
                                    episode,
                                    condition,
                                    row["success"],
                                    row["collision_any"],
                                    row["triggered_chunks"],
                                    row["corrective_chunks"],
                                ),
                                flush=True,
                            )
            finally:
                close = getattr(
                    guided_common.raw_env_from_wrapper(env), "close", None
                )
                if close is not None:
                    close()

    result = aggregate(rows)
    manifest = {
        "source_agent": str(args.source_agent),
        "corrective_agent": str(args.corrective_agent),
        "envs": args.envs,
        "seeds": args.seeds,
        "n_rollouts": args.n_rollouts,
        "horizon": args.horizon,
        "candidate_count": args.candidate_count,
        "selector": config.to_dict(),
        "paired_initial_states": True,
        "policy_seed_formula": "seed * 1_000_000 + episode",
        "nominal_chunk_seed_formula": "policy_seed * 10_000 + chunk_index",
        "candidate_seed_formula": (
            "[nominal_seed] + [nominal_seed + 100_000 + i for i in range(K-1)]"
        ),
        "deterministic_torch": args.deterministic_torch,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(guided_common.json_safe(result), indent=2) + "\n"
    )
    write_report(args.out_dir / "report.md", result)
    print("wrote {}".format(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
