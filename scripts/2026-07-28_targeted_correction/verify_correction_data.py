"""Verify paired-correction raw and derived HDF5 artifacts."""

import argparse
import json
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
from paired_correction import (  # noqa: E402
    PAIR_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    TEACHER_CONTROLLER,
    derive_episode_pairs,
    realized_delta_eef_pose_action,
    sha256_arrays,
    sha256_file,
)
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sorted_demo_names(data):
    return sorted(data, key=lambda name: int(name.split("_")[-1]))


def reference_observation_spec(path):
    with h5py.File(path, "r") as reference:
        demo_name = sorted_demo_names(reference["data"])[0]
        return {
            key: {
                "shape": reference["data/{}/obs/{}".format(demo_name, key)].shape[1:],
                "dtype": reference[
                    "data/{}/obs/{}".format(demo_name, key)
                ].dtype,
            }
            for key in reference["data/{}/obs".format(demo_name)]
        }


def longest_teacher_interval(controller, intervention_id):
    controller = np.asarray(controller)
    intervention_id = np.asarray(intervention_id)
    longest = 0
    current = 0
    current_id = -1
    for owner, identifier in zip(controller, intervention_id):
        if owner == TEACHER_CONTROLLER and identifier >= 0:
            if identifier == current_id:
                current += 1
            else:
                current = 1
                current_id = identifier
            longest = max(longest, current)
        else:
            current = 0
            current_id = -1
    return longest


def normalized_action_report(actions, offset, scale, tolerance):
    actions = np.asarray(actions, dtype=np.float32)
    normalized = (actions - offset.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
    outside = np.abs(normalized) > 1.0 + tolerance
    excess = np.maximum(np.abs(normalized) - 1.0, 0.0)
    return {
        "minimum": normalized.min(axis=(0, 1)).tolist(),
        "maximum": normalized.max(axis=(0, 1)).tolist(),
        "outside_count": outside.sum(axis=(0, 1)).astype(int).tolist(),
        "outside_fraction": outside.mean(axis=(0, 1)).tolist(),
        "maximum_excess": excess.max(axis=(0, 1)).tolist(),
        "total_outside": int(outside.sum()),
    }


def verify_raw(raw, reference_spec, action_tolerance):
    if raw.attrs.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError("Unexpected raw schema")
    policy_obs_keys = json.loads(text(raw.attrs["policy_obs_keys_json"]))
    raw_obs_keys = json.loads(text(raw.attrs["raw_obs_keys_json"]))
    if not set(policy_obs_keys).issubset(raw_obs_keys):
        raise ValueError("Raw observations do not contain every policy key")
    if "robot0_eef_quat_site" not in raw_obs_keys:
        raise ValueError("Raw observations lack the action-orientation key")
    offset = np.asarray(raw.attrs["source_action_offset"], dtype=np.float32)
    scale = np.asarray(raw.attrs["source_action_scale"], dtype=np.float32)
    if offset.shape != (7,) or scale.shape != (7,) or np.any(scale <= 0):
        raise ValueError("Invalid source action normalization statistics")
    if text(raw.attrs["source_action_stats_sha256"]) != sha256_arrays(
        offset, scale
    ):
        raise ValueError("Raw source action statistics fingerprint is invalid")

    episode_rows = []
    derived_parts = []
    total_pairs = 0
    for demo_name in sorted_demo_names(raw["data"]):
        group = raw["data/{}".format(demo_name)]
        length = int(group.attrs["num_samples"])
        train_eligible = bool(group.attrs.get("train_eligible", False))
        if length != len(group["actions"]):
            raise ValueError("{} num_samples mismatch".format(demo_name))
        if train_eligible and not bool(group.attrs.get("task_success", False)):
            raise ValueError("{} did not complete the task".format(demo_name))
        if train_eligible and not bool(
            group.attrs.get("selected_teacher_safe_pick", False)
        ):
            raise ValueError("{} did not pass the safe-pick gate".format(demo_name))
        if train_eligible and np.any(group["distractor_contact"][()]):
            raise ValueError(
                "{} is train-eligible but the selected episode branch "
                "contains distractor contact".format(demo_name)
            )
        if set(group["obs"]) != set(raw_obs_keys):
            raise ValueError(
                "{} raw observation keys differ from the contract".format(
                    demo_name
                )
            )
        for key in raw_obs_keys:
            spec = reference_spec[key]
            for prefix in ("obs", "next_obs"):
                dataset = group["{}/{}".format(prefix, key)]
                if dataset.shape != (length,) + spec["shape"]:
                    raise ValueError(
                        "{}/{} shape mismatch in {}".format(prefix, key, demo_name)
                    )
                if dataset.dtype != spec["dtype"]:
                    raise ValueError(
                        "{}/{} dtype mismatch in {}".format(prefix, key, demo_name)
                    )

        controller = group["controller"][()]
        intervention_id = group["intervention_id"][()]
        query_valid = group["robot_query_valid"][()].astype(bool)
        teacher = controller == TEACHER_CONTROLLER
        if not np.array_equal(query_valid, teacher):
            raise ValueError(
                "{} must contain one robot query per teacher step".format(demo_name)
            )
        if longest_teacher_interval(controller, intervention_id) < 16:
            raise ValueError(
                "{} has no contiguous 16-step intervention".format(demo_name)
            )

        actions = group["actions"][()]
        realized = np.asarray(
            [
                realized_delta_eef_pose_action(
                    {key: group["obs/{}".format(key)][index] for key in group["obs"]},
                    {
                        key: group["next_obs/{}".format(key)][index]
                        for key in group["next_obs"]
                    },
                    actions[index],
                )
                for index in range(length)
            ]
        )
        if not np.allclose(
            realized,
            group["delta_eef_pose_action"][()],
            atol=action_tolerance,
            rtol=0.0,
        ):
            maximum = float(
                np.max(
                    np.abs(
                        realized - group["delta_eef_pose_action"][()]
                    )
                )
            )
            raise ValueError(
                "{} realized-action mismatch, max {}".format(demo_name, maximum)
            )

        episode = {
            "obs": {
                key: group["obs/{}".format(key)][()]
                for key in policy_obs_keys
            },
            "delta_eef_pose_action": group["delta_eef_pose_action"][()],
            "controller": controller,
            "intervention_id": intervention_id,
            "robot_query_action": group["robot_query_action"][()],
            "robot_query_valid": query_valid,
        }
        pairs = derive_episode_pairs(episode, policy_obs_keys)
        pair_count = len(pairs["anchor_step"])
        if train_eligible and pair_count == 0:
            raise ValueError("{} yields no correction pairs".format(demo_name))
        if train_eligible:
            total_pairs += pair_count
            pairs["episode_id"] = np.asarray([demo_name] * pair_count)
            pairs["layout_id"] = np.asarray(
                [text(group.attrs["layout_id"])] * pair_count
            )
            pairs["selected_yaw_deg"] = np.full(
                pair_count,
                float(group.attrs["selected_yaw_deg"]),
                dtype=np.float32,
            )
            pairs["split"] = np.asarray(
                [text(group.attrs.get("split", "train"))] * pair_count
            )
            derived_parts.append(pairs)
        episode_rows.append(
            {
                "episode": demo_name,
                "train_eligible": train_eligible,
                "samples": length,
                "teacher_steps": int(teacher.sum()),
                "longest_teacher_interval": int(
                    longest_teacher_interval(controller, intervention_id)
                ),
                "pairs": pair_count if train_eligible else 0,
                "split": text(group.attrs.get("split", "train")),
            }
        )

    if int(raw["data"].attrs["total"]) != sum(
        row["samples"] for row in episode_rows
    ):
        raise ValueError("Raw data total attribute is inconsistent")
    return {
        "policy_obs_keys": policy_obs_keys,
        "action_offset": offset,
        "action_scale": scale,
        "episodes": episode_rows,
        "total_pairs": total_pairs,
        "derived_parts": derived_parts,
    }


def verify_pair_file(raw_path, raw, pairs, raw_summary, range_tolerance):
    if pairs.attrs.get("schema_version") != PAIR_SCHEMA_VERSION:
        raise ValueError("Unexpected pair schema")
    if text(pairs.attrs["raw_file_sha256"]) != sha256_file(raw_path):
        raise ValueError("Pair file was not derived from the current raw file")
    if text(pairs.attrs["source_checkpoint_sha256"]) != text(
        raw.attrs["source_checkpoint_sha256"]
    ):
        raise ValueError("Pair and raw checkpoint identities differ")
    if int(pairs.attrs["num_pairs"]) != raw_summary["total_pairs"]:
        raise ValueError("Pair count differs from deterministic raw extraction")

    positive = pairs["pairs/positive_actions"][()]
    negative = pairs["pairs/negative_actions"][()]
    if positive.shape != negative.shape or positive.shape[1:] != (16, 7):
        raise ValueError("Pair action arrays have invalid shapes")
    if not np.array_equal(positive[:, 0], negative[:, 0]):
        raise ValueError("Positive and negative history actions differ")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Pair actions contain non-finite values")

    expected_positive = np.concatenate(
        [part["positive_actions"] for part in raw_summary["derived_parts"]]
    )
    expected_negative = np.concatenate(
        [part["negative_actions"] for part in raw_summary["derived_parts"]]
    )
    if not np.array_equal(positive, expected_positive):
        raise ValueError("Stored positive pairs differ from raw extraction")
    if not np.array_equal(negative, expected_negative):
        raise ValueError("Stored negative pairs differ from raw extraction")

    for key in raw_summary["policy_obs_keys"]:
        value = pairs["pairs/obs/{}".format(key)]
        if value.shape[0] != len(positive) or value.shape[1] != 2:
            raise ValueError("Pair observation {} has invalid shape".format(key))
        expected = np.concatenate(
            [part["obs"][key] for part in raw_summary["derived_parts"]]
        )
        if not np.array_equal(value[()], expected):
            raise ValueError(
                "Stored pair observation {} differs from raw extraction".format(
                    key
                )
            )

    for key in (
        "anchor_step",
        "intervention_id",
        "episode_id",
        "layout_id",
        "selected_yaw_deg",
        "split",
    ):
        actual = pairs["pairs/{}".format(key)][()]
        if actual.dtype.kind in ("S", "O", "U"):
            actual = np.asarray([text(value) for value in actual])
        expected = np.concatenate(
            [part[key] for part in raw_summary["derived_parts"]]
        )
        if not np.array_equal(actual, expected):
            raise ValueError(
                "Stored pair metadata {} differs from raw extraction".format(key)
            )

    positive_report = normalized_action_report(
        positive,
        raw_summary["action_offset"],
        raw_summary["action_scale"],
        range_tolerance,
    )
    negative_report = normalized_action_report(
        negative,
        raw_summary["action_offset"],
        raw_summary["action_scale"],
        range_tolerance,
    )
    if positive_report["total_outside"]:
        raise ValueError("Positive correction actions exceed source normalization")
    if negative_report["total_outside"]:
        raise ValueError("Frozen-policy query actions exceed source normalization")
    return {
        "pairs": len(positive),
        "positive_normalized": positive_report,
        "negative_normalized": negative_report,
    }


def compare_observation(
    actual, expected, key, tolerance, image_integer_tolerance
):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.dtype.kind in ("u", "i", "b"):
        if key.endswith("_image"):
            matches = bool(
                np.max(
                    np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                )
                <= image_integer_tolerance
            )
        else:
            matches = np.array_equal(actual, expected)
    else:
        matches = np.allclose(actual, expected, atol=tolerance, rtol=0.0)
    if not matches:
        maximum = (
            float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
            if actual.shape == expected.shape
            else float("inf")
        )
        raise ValueError(
            "Replay observation {} mismatch, max {}".format(key, maximum)
        )


def quaternion_angle_error(q0, q1):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    cosine = np.clip(abs(float(np.dot(q0, q1))), 0.0, 1.0)
    return float(2.0 * np.arccos(cosine))


def replay_raw(
    raw,
    checkpoint_path,
    state_tolerance,
    observation_tolerance,
    image_integer_tolerance,
    maximum_eef_position_error,
    maximum_eef_orientation_error,
    maximum_gripper_error,
    episode_filter=None,
):
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=checkpoint_path)
    config, _ = FileUtils.config_from_checkpoint(
        algo_name=checkpoint["algo_name"],
        ckpt_dict=checkpoint,
        verbose=False,
    )
    ObsUtils.initialize_obs_utils_with_config(config)
    env = None
    current_env_name = None
    replay_rows = []
    for demo_name in sorted_demo_names(raw["data"]):
        if episode_filter is not None and demo_name not in episode_filter:
            continue
        if episode_filter is None and not bool(
            raw["data/{}".format(demo_name)].attrs.get(
                "train_eligible", False
            )
        ):
            continue
        group = raw["data/{}".format(demo_name)]
        env_name = text(group.attrs["env_name"])
        if env is None or current_env_name != env_name:
            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=checkpoint,
                env_name=env_name,
                render=False,
                render_offscreen=False,
                verbose=False,
            )
            current_env_name = env_name
        initial = {
            "states": group["states"][0],
            "model": text(group.attrs["model_file"]),
            "ep_meta": text(group.attrs.get("ep_meta", "")),
        }
        obs = env.reset_to(initial)
        raw_env = guided_common.raw_env_from_wrapper(env)
        records = guided_common.active_object_records(raw_env)
        maps = guided_common.contact_maps(raw_env, records, "Can")
        length = int(group.attrs["num_samples"])
        initial_state = env.get_state()["states"]
        if not np.allclose(
            initial_state,
            group["states"][0],
            atol=state_tolerance,
            rtol=0.0,
        ):
            raise ValueError("{} initial state did not restore".format(demo_name))
        for key in group["obs"]:
            if key.endswith("_image"):
                continue
            compare_observation(
                guided_common.latest_obs_value(obs, key),
                group["obs/{}".format(key)][0],
                key,
                observation_tolerance,
                image_integer_tolerance,
            )

        position_errors = []
        orientation_errors = []
        gripper_errors = []
        realized_action_errors = []
        maximum_state_drift = 0.0
        target_contact_any = False
        obstacle_contact_any = False
        for index in range(length):
            current_state = env.get_state()["states"]
            maximum_state_drift = max(
                maximum_state_drift,
                float(
                    np.max(
                        np.abs(current_state - group["states"][index])
                    )
                ),
            )
            current_position = guided_common.latest_obs_value(
                obs, "robot0_eef_pos"
            )
            expected_position = group["obs/robot0_eef_pos"][index]
            position_errors.append(
                float(np.linalg.norm(current_position - expected_position))
            )
            current_quaternion = guided_common.latest_obs_value(
                obs, "robot0_eef_quat_site"
            )
            expected_quaternion = group["obs/robot0_eef_quat_site"][index]
            orientation_errors.append(
                quaternion_angle_error(
                    current_quaternion, expected_quaternion
                )
            )
            current_gripper = guided_common.latest_obs_value(
                obs, "robot0_gripper_qpos"
            )
            expected_gripper = group["obs/robot0_gripper_qpos"][index]
            gripper_errors.append(
                float(np.max(np.abs(current_gripper - expected_gripper)))
            )
            action = group["actions"][index]
            next_obs, _, _, _ = env.step(action)
            target_contact, obstacle_contact, _ = (
                guided_common.contact_snapshot(raw_env, maps)
            )
            target_contact_any |= bool(target_contact)
            obstacle_contact_any |= bool(obstacle_contact)
            replay_action = realized_delta_eef_pose_action(
                obs, next_obs, action
            )
            realized_action_errors.append(
                float(
                    np.max(
                        np.abs(
                            replay_action
                            - group["delta_eef_pose_action"][index]
                        )
                    )
                )
            )
            obs = next_obs
        success = guided_common.task_success(env)
        failures = []
        if success != bool(group.attrs["task_success"]):
            failures.append("final_task_success_mismatch")
        expected_target_contact = bool(np.any(group["target_contact"][()]))
        expected_obstacle_contact = bool(
            np.any(group["distractor_contact"][()])
        )
        if target_contact_any != expected_target_contact:
            failures.append("target_contact_outcome_mismatch")
        if obstacle_contact_any != expected_obstacle_contact:
            failures.append("distractor_contact_outcome_mismatch")
        max_position = max(position_errors)
        max_orientation = max(orientation_errors)
        max_gripper = max(gripper_errors)
        if max_position > maximum_eef_position_error:
            failures.append("eef_position_error_exceeds_threshold")
        if max_orientation > maximum_eef_orientation_error:
            failures.append("eef_orientation_error_exceeds_threshold")
        if max_gripper > maximum_gripper_error:
            failures.append("gripper_error_exceeds_threshold")
        replay_rows.append(
            {
                "episode": demo_name,
                "pass": not failures,
                "failures": failures,
                "success": success,
                "expected_success": bool(group.attrs["task_success"]),
                "target_contact_any": target_contact_any,
                "expected_target_contact_any": expected_target_contact,
                "distractor_contact_any": obstacle_contact_any,
                "expected_distractor_contact_any": expected_obstacle_contact,
                "maximum_state_drift": maximum_state_drift,
                "maximum_eef_position_error_m": max_position,
                "maximum_eef_orientation_error_rad": max_orientation,
                "maximum_gripper_qpos_error": max_gripper,
                "maximum_realized_action_error": max(
                    realized_action_errors
                ),
            }
        )
    return replay_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("outputs/targeted_finetuning/correction_raw.hdf5"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("outputs/targeted_finetuning/correction_pairs.hdf5"),
    )
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        default=Path(
            "third_party/robomimic/datasets/can/yq/"
            "image_v15_delta_eef_pose_osc.hdf5"
        ),
    )
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-episodes", nargs="+", default=None)
    parser.add_argument("--allow-replay-failures", action="store_true")
    parser.add_argument(
        "--mark-replay-failures-ineligible",
        action="store_true",
        help=(
            "After replay, retain failed raw episodes but set train_eligible "
            "false and record the replay failure reasons."
        ),
    )
    parser.add_argument("--range-tolerance", type=float, default=1e-5)
    parser.add_argument("--action-tolerance", type=float, default=1e-6)
    parser.add_argument("--state-tolerance", type=float, default=1e-8)
    parser.add_argument("--observation-tolerance", type=float, default=1e-6)
    parser.add_argument("--image-integer-tolerance", type=int, default=1)
    parser.add_argument(
        "--maximum-eef-position-error", type=float, default=0.01
    )
    parser.add_argument(
        "--maximum-eef-orientation-error", type=float, default=0.05
    )
    parser.add_argument("--maximum-gripper-error", type=float, default=0.02)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.raw, args.pairs, args.reference_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference_spec = reference_observation_spec(args.reference_dataset)
    with h5py.File(args.raw, "r") as raw, h5py.File(args.pairs, "r") as pairs:
        checkpoint_path = Path(text(raw.attrs["source_checkpoint"]))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if text(raw.attrs["source_checkpoint_sha256"]) != sha256_file(
            checkpoint_path
        ):
            raise ValueError("Raw source checkpoint fingerprint is invalid")
        raw_summary = verify_raw(
            raw, reference_spec, action_tolerance=args.action_tolerance
        )
        pair_summary = verify_pair_file(
            args.raw,
            raw,
            pairs,
            raw_summary,
            range_tolerance=args.range_tolerance,
        )
        replay_summary = (
            replay_raw(
                raw,
                checkpoint_path=str(checkpoint_path),
                state_tolerance=args.state_tolerance,
                observation_tolerance=args.observation_tolerance,
                image_integer_tolerance=args.image_integer_tolerance,
                maximum_eef_position_error=args.maximum_eef_position_error,
                maximum_eef_orientation_error=(
                    args.maximum_eef_orientation_error
                ),
                maximum_gripper_error=args.maximum_gripper_error,
                episode_filter=(
                    set(args.replay_episodes)
                    if args.replay_episodes is not None
                    else None
                ),
            )
            if args.replay
            else None
        )
    replay_pass = bool(
        replay_summary is None
        or all(row["pass"] for row in replay_summary)
    )
    summary = {
        "status": "pass" if replay_pass else "fail",
        "raw": {
            "episodes": raw_summary["episodes"],
            "total_pairs": raw_summary["total_pairs"],
        },
        "pairs": pair_summary,
        "replay": replay_summary,
    }
    payload = json.dumps(summary, indent=2) + "\n"
    if args.mark_replay_failures_ineligible and replay_summary is not None:
        failed = [row for row in replay_summary if not row["pass"]]
        if failed:
            with h5py.File(args.raw, "r+") as raw:
                for row in failed:
                    group = raw["data/{}".format(row["episode"])]
                    group.attrs["train_eligible"] = False
                    group.attrs["replay_verified"] = False
                    group.attrs["rejection_reason"] = "c0_replay_failed"
                    group.attrs["replay_failures_json"] = json.dumps(
                        row["failures"]
                    )
                raw.flush()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")
    if not replay_pass and not args.allow_replay_failures:
        raise RuntimeError("One or more replay episodes failed")


if __name__ == "__main__":
    main()
