"""Collect successful paired corrections on fresh two-distractor layouts."""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
GUIDED_DIR = SCRIPTS_DIR / "guided_denoising"
FEASIBILITY_DIR = SCRIPTS_DIR / "2026-07-22_task_constrained_guidance"
TEACHER_PATH = (
    SCRIPTS_DIR
    / "2026-07-27_semi_manual_waypoint_teacher"
    / "evaluate_waypoint_teacher.py"
)
for directory in (SCRIPT_DIR, GUIDED_DIR, FEASIBILITY_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common as guided_common  # noqa: E402
from feasibility import (  # noqa: E402
    closest_target_progress,
    minimum_segment_clearance_xy,
    reconstruct_positions,
)
from paired_correction import (  # noqa: E402
    POLICY_CONTROLLER,
    TEACHER_CONTROLLER,
    append_raw_episode,
    copy_current_observation,
    initialize_raw_file,
    realized_delta_eef_pose_action,
    sha256_file,
)
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


def _load_teacher_module():
    spec = importlib.util.spec_from_file_location(
        "targeted_correction_waypoint_teacher", TEACHER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Teacher = _load_teacher_module()


def json_safe(value):
    return guided_common.json_safe(value)


def stable_split(layout_id, valid_fraction, split_seed):
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("--valid-fraction must be in [0, 1)")
    token = "{}:{}".format(split_seed, layout_id).encode("utf-8")
    score = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    return "valid" if score < valid_fraction else "train"


def reference_observation_keys(dataset_path):
    with h5py.File(dataset_path, "r") as dataset:
        demo_name = sorted(
            dataset["data"], key=lambda name: int(name.split("_")[-1])
        )[0]
        return sorted(dataset["data/{}/obs".format(demo_name)])


class EpisodeRecorder:
    """In-memory recorder for the selected continuous branch only."""

    def __init__(
        self,
        *,
        policy,
        raw_env,
        maps,
        raw_obs_keys,
        query_seed_base,
    ):
        self.policy = policy
        self.raw_env = raw_env
        self.maps = maps
        self.raw_obs_keys = tuple(raw_obs_keys)
        self.query_seed_base = int(query_seed_base)
        self.model_file = None
        self.ep_meta = ""
        self.trace = {
            "states": [],
            "obs": {key: [] for key in self.raw_obs_keys},
            "next_obs": {key: [] for key in self.raw_obs_keys},
            "actions": [],
            "delta_eef_pose_action": [],
            "rewards": [],
            "dones": [],
            "controller": [],
            "intervention_id": [],
            "robot_query_action": [],
            "robot_query_valid": [],
            "robot_query_seed": [],
            "target_contact": [],
            "distractor_contact": [],
        }

    def _query_robot_action(self, obs, query_seed):
        policy_obs = guided_common.policy_observation(obs, self.policy)
        context = guided_common.make_guidance_context(
            self.policy,
            policy_obs,
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            scale=0.0,
            clearance_margin=0.0,
        )
        queue_length = len(self.policy.policy.action_queue)
        _, actions, _ = guided_common.sample_action_chunk(
            self.policy,
            policy_obs,
            context,
            noise_seed=query_seed,
        )
        if len(self.policy.policy.action_queue) != queue_length:
            raise RuntimeError("A robot query mutated the rollout action queue")
        if actions.shape != (8, 7):
            raise ValueError(
                "Expected queried executable action shape (8, 7), got {}".format(
                    actions.shape
                )
            )
        return actions[0].copy()

    def __call__(
        self,
        *,
        state_before,
        obs,
        next_obs,
        action,
        reward,
        done,
        info,
        phase,
        label,
        target_contact,
        obstacle_contact,
        obstacle_names,
        task_success,
    ):
        del info, label, obstacle_names
        missing = [
            key
            for key in self.raw_obs_keys
            if key not in obs or key not in next_obs
        ]
        if missing:
            raise KeyError(
                "Environment is missing reference observation keys: {}".format(
                    missing
                )
            )
        step = len(self.trace["actions"])
        if self.model_file is None:
            self.model_file = state_before["model"]
            self.ep_meta = state_before.get("ep_meta", "")
        self.trace["states"].append(np.asarray(state_before["states"]).copy())
        for key in self.raw_obs_keys:
            self.trace["obs"][key].append(
                copy_current_observation({key: obs[key]})[key]
            )
            self.trace["next_obs"][key].append(
                copy_current_observation({key: next_obs[key]})[key]
            )
        self.trace["actions"].append(np.asarray(action, dtype=np.float32).copy())
        self.trace["delta_eef_pose_action"].append(
            realized_delta_eef_pose_action(obs, next_obs, action)
        )
        self.trace["rewards"].append(float(reward))
        self.trace["dones"].append(bool(done or task_success))
        is_teacher = phase == "teacher"
        self.trace["controller"].append(
            TEACHER_CONTROLLER if is_teacher else POLICY_CONTROLLER
        )
        self.trace["intervention_id"].append(0 if is_teacher else -1)
        if is_teacher:
            query_seed = self.query_seed_base + step
            query_action = self._query_robot_action(obs, query_seed)
            query_valid = True
        else:
            query_seed = -1
            query_action = np.full(7, np.nan, dtype=np.float32)
            query_valid = False
        self.trace["robot_query_action"].append(query_action)
        self.trace["robot_query_valid"].append(query_valid)
        self.trace["robot_query_seed"].append(query_seed)
        self.trace["target_contact"].append(bool(target_contact))
        self.trace["distractor_contact"].append(bool(obstacle_contact))

    def arrays(self):
        if not self.trace["actions"]:
            raise ValueError("Recorder contains no transitions")
        result = {}
        for key, value in self.trace.items():
            if key in ("obs", "next_obs"):
                result[key] = {
                    obs_key: np.asarray(obs_values)
                    for obs_key, obs_values in value.items()
                }
            else:
                result[key] = np.asarray(value)
        return result


def point_clearance_xy(position, centers, radii):
    if len(radii) == 0:
        return float("inf")
    distances = np.linalg.norm(
        np.asarray(centers)[:, :2] - np.asarray(position)[None, :2], axis=1
    )
    return float(np.min(distances - np.asarray(radii)))


def trigger_diagnostic(policy, obs, raw_env, records, args, noise_seed):
    centers, radii, obstacle_names = guided_common.oracle_obstacle_geometry(
        raw_env, records, args.target_object_name
    )
    target_position = guided_common.target_object_position(
        raw_env, records, args.target_object_name
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
    _, actions, _ = guided_common.sample_action_chunk(
        policy, policy_obs, context, noise_seed=noise_seed
    )
    predicted_positions = reconstruct_positions(start_position, actions)
    progress = closest_target_progress(
        start_position, predicted_positions, target_position
    )
    clearance = minimum_segment_clearance_xy(
        start_position, predicted_positions, centers, radii
    )
    start_clearance = point_clearance_xy(start_position, centers, radii)
    return {
        "actions": actions,
        "centers": centers,
        "radii": radii,
        "obstacle_names": obstacle_names,
        "target_position": target_position,
        "predicted_positions": predicted_positions,
        "predicted_progress_m": progress,
        "predicted_min_clearance_m": clearance,
        "start_min_clearance_m": start_clearance,
        "trigger": bool(
            progress > 0.0
            and clearance <= args.predicted_clearance_threshold
            and start_clearance >= args.minimum_start_clearance
        ),
    }


def teacher_kwargs(args):
    return {
        "target_object_name": args.target_object_name,
        "minimum_lift_m": args.minimum_lift,
        "object_overhead_m": args.object_overhead,
        "handoff_height_m": args.handoff_height,
        "max_delta_m": args.max_delta,
        "settle_steps": args.settle_steps,
        "gripper_command": args.gripper_command,
        "route": "grasp_handoff",
        "grasp_offset": args.grasp_offset,
        "close_steps": args.close_steps,
        "grasp_lift_m": args.grasp_lift,
        "max_rotation_delta_rad": args.max_rotation_delta,
        "restore_yaw_before_handoff": False,
    }


def run_teacher(
    policy,
    env,
    saved,
    args,
    yaw,
    *,
    continue_with_policy,
    transition_callback=None,
):
    config = teacher_kwargs(args)
    return Teacher.evaluate_teacher(
        policy=policy,
        env=env,
        saved=saved,
        maximum_steps=max(1, args.horizon - saved["start_step"]),
        relative_yaw_deg=yaw,
        continue_with_policy=continue_with_policy,
        transition_callback=transition_callback,
        **config,
    )


def search_yaw(policy, env, saved, args):
    candidates = []
    for group in Teacher.yaw_candidate_groups(args.yaw_candidates):
        group_results = [
            run_teacher(
                policy,
                env,
                saved,
                args,
                yaw,
                continue_with_policy=False,
            )
            for yaw in group
        ]
        candidates.extend(group_results)
        if any(
            Teacher.candidate_is_safe_pick(
                candidate, args.minimum_target_lift
            )
            for candidate in group_results
        ):
            break
    selected = Teacher.select_yaw_candidate(
        candidates, args.minimum_target_lift
    )
    return selected, {
        "minimum_target_lift_m": args.minimum_target_lift,
        "selected_yaw_deg": selected["relative_yaw_deg"],
        "selected_safe_pick": Teacher.candidate_is_safe_pick(
            selected, args.minimum_target_lift
        ),
        "candidates": [
            Teacher.compact_candidate(candidate, args.minimum_target_lift)
            for candidate in candidates
        ],
    }


def append_attempt(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def compact_trigger(diagnostic):
    return {
        key: value
        for key, value in diagnostic.items()
        if key not in (
            "actions",
            "centers",
            "radii",
            "predicted_positions",
        )
    }


def collect_attempt(policy, env, args, seed, episode, raw_obs_keys):
    layout_seed = seed * 1000 + episode
    layout_id = "{}_seed{}_episode{}".format(args.env, seed, episode)
    guided_common.seed_everything(layout_seed)
    policy.start_episode()
    obs = env.reset()
    raw_env = guided_common.raw_env_from_wrapper(env)
    records = guided_common.active_object_records(raw_env)
    maps = guided_common.contact_maps(
        raw_env, records, args.target_object_name
    )
    recorder = EpisodeRecorder(
        policy=policy,
        raw_env=raw_env,
        maps=maps,
        raw_obs_keys=raw_obs_keys,
        query_seed_base=layout_seed * 1_000_000 + 500_000,
    )
    metrics = Teacher.new_metrics()
    step = 0
    chunk_index = 0
    target_seen = False

    while step < args.horizon and not target_seen:
        state_before = env.get_state()
        obs_history = {
            key: np.asarray(value).copy() for key, value in obs.items()
        }
        noise_seed = layout_seed * 1000 + chunk_index
        diagnostic = trigger_diagnostic(
            policy, obs, raw_env, records, args, noise_seed
        )
        target_contacts, obstacle_contacts, _ = (
            guided_common.contact_snapshot(raw_env, maps)
        )
        target_seen = target_seen or target_contacts > 0
        can_trigger = bool(
            diagnostic["trigger"]
            and not target_seen
            and obstacle_contacts == 0
        )
        if can_trigger:
            saved = {
                "state": state_before,
                "obs_history": obs_history,
                "start_step": step,
                "noise_seed": noise_seed,
            }
            selected, yaw_search = search_yaw(policy, env, saved, args)
            if not yaw_search["selected_safe_pick"]:
                return {
                    "accepted": False,
                    "reason": "no_safe_teacher_candidate",
                    "layout_id": layout_id,
                    "layout_seed": layout_seed,
                    "seed": seed,
                    "episode": episode,
                    "trigger_step": step,
                    "trigger": compact_trigger(diagnostic),
                    "yaw_search": yaw_search,
                }
            teacher_result = run_teacher(
                policy,
                env,
                saved,
                args,
                selected["relative_yaw_deg"],
                continue_with_policy=True,
                transition_callback=recorder,
            )
            episode_trace = recorder.arrays()
            complete_distractor_contact_any = bool(
                np.any(episode_trace["distractor_contact"])
            )
            train_eligible = bool(
                yaw_search["selected_safe_pick"]
                and teacher_result["teacher_target_contact_any"]
                and not teacher_result["teacher_collision_any"]
                and not complete_distractor_contact_any
                and teacher_result["target_lift_at_handoff_m"]
                >= args.minimum_target_lift
                and teacher_result["success"]
            )
            if train_eligible:
                reason = "accepted"
            elif complete_distractor_contact_any:
                reason = "distractor_contact_on_selected_branch"
            else:
                reason = "task_failed_after_teacher"
            return {
                "accepted": train_eligible,
                "reason": reason,
                "layout_id": layout_id,
                "layout_seed": layout_seed,
                "seed": seed,
                "episode": episode,
                "trigger_step": step,
                "trigger": compact_trigger(diagnostic),
                "yaw_search": yaw_search,
                "teacher_result": teacher_result,
                "complete_distractor_contact_any": (
                    complete_distractor_contact_any
                ),
                "episode_trace": episode_trace,
                "model_file": recorder.model_file,
                "ep_meta": recorder.ep_meta,
            }

        terminal = False
        for action in diagnostic["actions"]:
            if step >= args.horizon:
                break
            obs, terminal = Teacher.execute_action(
                env,
                raw_env,
                maps,
                obs,
                action,
                metrics,
                "policy",
                "pretrigger_policy",
                transition_callback=recorder,
            )
            step += 1
            target_contacts, _, _ = guided_common.contact_snapshot(
                raw_env, maps
            )
            target_seen = target_seen or target_contacts > 0
            if terminal or target_seen:
                break
        if terminal:
            break
        chunk_index += 1

    return {
        "accepted": False,
        "reason": (
            "target_contact_before_trigger"
            if target_seen
            else "no_trigger_before_horizon"
        ),
        "layout_id": layout_id,
        "layout_seed": layout_seed,
        "seed": seed,
        "episode": episode,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=guided_common.DEFAULT_AGENT)
    parser.add_argument("--env", default="PickPlaceBreadCerealCan")
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        default=Path(
            "third_party/robomimic/datasets/can/yq/"
            "image_v15_delta_eef_pose_osc.hdf5"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[700, 701, 702])
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--accepted-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--minimum-start-clearance", type=float, default=0.01)
    parser.add_argument(
        "--predicted-clearance-threshold", type=float, default=0.01
    )
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--minimum-lift", type=float, default=0.06)
    parser.add_argument("--object-overhead", type=float, default=0.18)
    parser.add_argument("--handoff-height", type=float, default=0.12)
    parser.add_argument("--max-delta", type=float, default=0.006)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--gripper-command", type=float, default=-1.0)
    parser.add_argument(
        "--grasp-offset",
        nargs=3,
        type=float,
        default=[0.002, -0.0025, 0.018],
    )
    parser.add_argument("--close-steps", type=int, default=10)
    parser.add_argument("--grasp-lift", type=float, default=0.15)
    parser.add_argument(
        "--max-rotation-delta",
        type=float,
        default=0.015,
        help=(
            "Teacher yaw increment. The 0.015 rad default stays within the "
            "epoch-260 rotation-z action range; the diagnostic teacher's "
            "historical 0.04 rad value is not valid for training data."
        ),
    )
    parser.add_argument(
        "--yaw-candidates",
        nargs="+",
        type=float,
        default=[0.0, -20.0, 20.0, -30.0, 30.0, -45.0, 45.0, 90.0],
    )
    parser.add_argument("--minimum-target-lift", type=float, default=0.05)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260728)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/targeted_finetuning/correction_raw.hdf5"),
    )
    parser.add_argument(
        "--attempt-log",
        type=Path,
        default=Path("outputs/targeted_finetuning/collection_attempts.jsonl"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.env != "PickPlaceBreadCerealCan":
        raise ValueError("C0 collection is fixed to PickPlaceBreadCerealCan")
    if any(600 <= seed <= 607 for seed in args.seeds):
        raise ValueError("Seeds 600--607 are held out and cannot be collected")
    if args.accepted_episodes <= 0:
        raise ValueError("--accepted-episodes must be positive")
    if args.episodes_per_seed <= 0:
        raise ValueError("--episodes-per-seed must be positive")
    if args.episode_start < 0:
        raise ValueError("--episode-start must be nonnegative")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if not args.reference_dataset.is_file():
        raise FileNotFoundError(args.reference_dataset)
    stable_split("preflight", args.valid_fraction, args.split_seed)


def main():
    args = parse_args()
    validate_args(args)
    args.out = args.out.resolve()
    args.attempt_log = args.attempt_log.resolve()
    args.agent = args.agent.resolve()
    args.reference_dataset = args.reference_dataset.resolve()
    if args.overwrite:
        for path in (args.out, args.attempt_log):
            if path.exists():
                path.unlink()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = guided_common.load_guided_policy_and_checkpoint(
        args.agent, device
    )
    checkpoint_sha256 = sha256_file(args.agent)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=args.env,
        render=False,
        render_offscreen=False,
        verbose=False,
    )
    policy_obs_keys = list(policy.policy.global_config.all_obs_keys)
    reference_keys = set(reference_observation_keys(args.reference_dataset))
    raw_obs_keys = sorted(set(policy_obs_keys + ["robot0_eef_quat_site"]))
    missing_reference_keys = set(raw_obs_keys) - reference_keys
    if missing_reference_keys:
        raise KeyError(
            "Reference dataset is missing required raw observation keys: {}".format(
                sorted(missing_reference_keys)
            )
        )
    action_key = policy.policy.global_config.train.action_keys[0]
    action_stats = policy.action_normalization_stats[action_key]
    env_metadata = copy.deepcopy(checkpoint["env_metadata"])
    env_metadata["env_name"] = args.env

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "a") as output:
        initialize_raw_file(
            output,
            source_checkpoint=args.agent,
            source_checkpoint_sha256=checkpoint_sha256,
            env_metadata=env_metadata,
            policy_obs_keys=policy_obs_keys,
            raw_obs_keys=raw_obs_keys,
            action_offset=action_stats["offset"],
            action_scale=action_stats["scale"],
        )
        accepted = sum(
            bool(output["data/{}".format(name)].attrs.get("train_eligible", False))
            for name in output["data"]
        )
        next_episode_id = len(output["data"])
        attempts = 0
        for seed in args.seeds:
            if accepted >= args.accepted_episodes:
                break
            for episode in range(
                args.episode_start,
                args.episode_start + args.episodes_per_seed,
            ):
                if accepted >= args.accepted_episodes:
                    break
                attempts += 1
                result = collect_attempt(
                    policy, env, args, seed, episode, raw_obs_keys
                )
                log_payload = {
                    key: value
                    for key, value in result.items()
                    if key not in ("episode_trace", "model_file", "ep_meta")
                }
                append_attempt(args.attempt_log, log_payload)
                if not result["accepted"]:
                    print(
                        "attempt {} {} rejected: {}".format(
                            attempts, result["layout_id"], result["reason"]
                        ),
                        flush=True,
                    )
                    continue

                yaw_search = result["yaw_search"]
                teacher_result = result["teacher_result"]
                attrs = {
                    "model_file": result["model_file"],
                    "ep_meta": result["ep_meta"],
                    "env_name": args.env,
                    "env_seed": seed,
                    "layout_seed": result["layout_seed"],
                    "layout_id": result["layout_id"],
                    "task_success": teacher_result["success"],
                    "selected_teacher_safe_pick": (
                        yaw_search["selected_safe_pick"]
                    ),
                    "train_eligible": True,
                    "selected_yaw_deg": yaw_search["selected_yaw_deg"],
                    "teacher_config_json": teacher_kwargs(args),
                    "trigger_step": result["trigger_step"],
                    "trigger_diagnostic_json": {
                        key: value for key, value in result["trigger"].items()
                    },
                    "yaw_search_json": yaw_search,
                    "split": stable_split(
                        result["layout_id"],
                        args.valid_fraction,
                        args.split_seed,
                    ),
                }
                name = append_raw_episode(
                    output,
                    next_episode_id,
                    result["episode_trace"],
                    attrs,
                )
                next_episode_id += 1
                accepted += 1
                print(
                    "accepted {} ({}/{}) layout={} yaw={} steps={}".format(
                        name,
                        accepted,
                        args.accepted_episodes,
                        result["layout_id"],
                        yaw_search["selected_yaw_deg"],
                        len(result["episode_trace"]["actions"]),
                    ),
                    flush=True,
                )

    summary = {
        "raw": str(args.out),
        "attempt_log": str(args.attempt_log),
        "accepted_episodes": accepted,
        "requested_accepted_episodes": args.accepted_episodes,
        "attempts_this_run": attempts,
        "held_out_seeds": list(range(600, 608)),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if accepted < args.accepted_episodes:
        raise RuntimeError(
            "Collected {} of {} requested accepted episodes".format(
                accepted, args.accepted_episodes
            )
        )


if __name__ == "__main__":
    main()
