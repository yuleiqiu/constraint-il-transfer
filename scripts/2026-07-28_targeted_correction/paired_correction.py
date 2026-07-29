"""Pure data utilities for paired corrective Diffusion Policy supervision."""

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

import robosuite.utils.transform_utils as TransformUtils


RAW_SCHEMA_VERSION = "paired-correction-raw-v1"
PAIR_SCHEMA_VERSION = "paired-correction-pairs-v1"
POLICY_CONTROLLER = 0
TEACHER_CONTROLLER = 1
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def latest_frame(value):
    """Return the current frame from a raw or frame-stacked observation."""

    value = np.asarray(value)
    return value[-1] if value.ndim >= 2 else value


def copy_current_observation(obs):
    """Remove the frame-stack dimension while retaining every observation key."""

    return {
        key: np.asarray(latest_frame(value)).copy()
        for key, value in obs.items()
    }


def delta_axis_angle(q0_xyzw, q1_xyzw):
    """Return world-frame rotvec dR such that dR @ R0 = R1."""

    r0 = TransformUtils.quat2mat(np.asarray(q0_xyzw).copy())
    r1 = TransformUtils.quat2mat(np.asarray(q1_xyzw).copy())
    return Rotation.from_matrix(r1 @ r0.T).as_rotvec()


def realized_delta_eef_pose_action(
    obs,
    next_obs,
    executed_command,
    orientation_key="robot0_eef_quat_site",
):
    """Build the source-compatible action target from one realized transition."""

    current_position = np.asarray(latest_frame(obs["robot0_eef_pos"]))
    next_position = np.asarray(latest_frame(next_obs["robot0_eef_pos"]))
    current_quaternion = np.asarray(latest_frame(obs[orientation_key]))
    next_quaternion = np.asarray(latest_frame(next_obs[orientation_key]))
    command = np.asarray(executed_command, dtype=np.float32).reshape(-1)
    if command.shape != (7,):
        raise ValueError("Expected a 7-D executed command")

    action = np.zeros(7, dtype=np.float32)
    action[:3] = next_position - current_position
    action[3:6] = delta_axis_angle(current_quaternion, next_quaternion)
    action[6] = command[6]
    if not np.isfinite(action).all():
        raise ValueError("Realized delta-EEF action contains non-finite values")
    return action


def aligned_source_sample(obs, actions, anchor, prediction_horizon=16):
    """Construct the exact sample prefix consumed by the source DP trainer."""

    if anchor <= 0:
        raise ValueError("Pair anchors require a real previous transition")
    actions = np.asarray(actions)
    end = anchor + prediction_horizon - 1
    if end > len(actions):
        raise ValueError("Not enough actions after anchor")
    aligned_obs = {
        key: np.asarray(values)[anchor - 1 : anchor + 1]
        for key, values in obs.items()
    }
    aligned_actions = actions[anchor - 1 : end]
    if aligned_actions.shape[0] != prediction_horizon:
        raise AssertionError("Aligned action chunk has the wrong length")
    return aligned_obs, aligned_actions


def eligible_pair_anchors(
    controller,
    intervention_id,
    robot_query_valid,
    prediction_horizon=16,
):
    """Return anchors with one historical and 15 corrected future actions."""

    controller = np.asarray(controller).reshape(-1)
    intervention_id = np.asarray(intervention_id).reshape(-1)
    robot_query_valid = np.asarray(robot_query_valid, dtype=bool).reshape(-1)
    if not (
        len(controller) == len(intervention_id) == len(robot_query_valid)
    ):
        raise ValueError("Controller, intervention, and query arrays must align")

    corrected_length = prediction_horizon - 1
    anchors = []
    for anchor in range(1, len(controller) - corrected_length + 1):
        window = slice(anchor, anchor + corrected_length)
        ids = intervention_id[window]
        if not np.all(controller[window] == TEACHER_CONTROLLER):
            continue
        if ids[0] < 0 or not np.all(ids == ids[0]):
            continue
        if not np.all(robot_query_valid[window]):
            continue
        anchors.append(anchor)
    return anchors


def derive_episode_pairs(episode, obs_keys, prediction_horizon=16):
    """Derive all eligible pair windows from one in-memory raw episode."""

    required = (
        "delta_eef_pose_action",
        "controller",
        "intervention_id",
        "robot_query_action",
        "robot_query_valid",
        "obs",
    )
    missing = [key for key in required if key not in episode]
    if missing:
        raise KeyError("Missing raw episode fields: {}".format(missing))

    positive = np.asarray(episode["delta_eef_pose_action"], dtype=np.float32)
    negative_step = np.asarray(episode["robot_query_action"], dtype=np.float32)
    if positive.ndim != 2 or positive.shape[1] != 7:
        raise ValueError("delta_eef_pose_action must have shape [N, 7]")
    if negative_step.shape != positive.shape:
        raise ValueError("robot_query_action must match positive action shape")

    anchors = eligible_pair_anchors(
        episode["controller"],
        episode["intervention_id"],
        episode["robot_query_valid"],
        prediction_horizon=prediction_horizon,
    )
    result = {
        "obs": {key: [] for key in obs_keys},
        "positive_actions": [],
        "negative_actions": [],
        "anchor_step": [],
        "intervention_id": [],
    }
    for anchor in anchors:
        corrected_end = anchor + prediction_horizon - 1
        positive_chunk = np.concatenate(
            (
                positive[anchor - 1 : anchor],
                positive[anchor:corrected_end],
            ),
            axis=0,
        )
        negative_chunk = np.concatenate(
            (
                positive[anchor - 1 : anchor],
                negative_step[anchor:corrected_end],
            ),
            axis=0,
        )
        if positive_chunk.shape != (prediction_horizon, 7):
            raise AssertionError("Positive pair chunk has the wrong shape")
        if negative_chunk.shape != (prediction_horizon, 7):
            raise AssertionError("Negative pair chunk has the wrong shape")
        if not np.array_equal(positive_chunk[0], negative_chunk[0]):
            raise AssertionError("Pair history action must be shared")
        if not (
            np.isfinite(positive_chunk).all()
            and np.isfinite(negative_chunk).all()
        ):
            raise ValueError("Pair action chunk contains non-finite values")

        for key in obs_keys:
            if key not in episode["obs"]:
                raise KeyError("Missing policy observation key '{}'".format(key))
            values = np.asarray(episode["obs"][key])
            result["obs"][key].append(values[anchor - 1 : anchor + 1])
        result["positive_actions"].append(positive_chunk)
        result["negative_actions"].append(negative_chunk)
        result["anchor_step"].append(anchor)
        result["intervention_id"].append(
            int(np.asarray(episode["intervention_id"])[anchor])
        )

    return {
        "obs": {
            key: np.asarray(values)
            for key, values in result["obs"].items()
        },
        "positive_actions": np.asarray(
            result["positive_actions"], dtype=np.float32
        ).reshape(-1, prediction_horizon, 7),
        "negative_actions": np.asarray(
            result["negative_actions"], dtype=np.float32
        ).reshape(-1, prediction_horizon, 7),
        "anchor_step": np.asarray(result["anchor_step"], dtype=np.int64),
        "intervention_id": np.asarray(
            result["intervention_id"], dtype=np.int32
        ),
    }


def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def initialize_raw_file(
    output,
    *,
    source_checkpoint,
    source_checkpoint_sha256,
    env_metadata,
    policy_obs_keys,
    raw_obs_keys,
    action_offset,
    action_scale,
):
    """Initialize or validate the root of an append-only raw correction file."""

    if "data" in output:
        if output.attrs.get("schema_version") != RAW_SCHEMA_VERSION:
            raise ValueError("Existing output has a different raw schema")
        if (
            str(output.attrs.get("source_checkpoint_sha256"))
            != source_checkpoint_sha256
        ):
            raise ValueError("Existing output uses a different checkpoint")
        return

    output.attrs["schema_version"] = RAW_SCHEMA_VERSION
    output.attrs["source_checkpoint"] = str(source_checkpoint)
    output.attrs["source_checkpoint_sha256"] = source_checkpoint_sha256
    output.attrs["action_key"] = "delta_eef_pose_action"
    output.attrs["observation_horizon"] = 2
    output.attrs["prediction_horizon"] = 16
    output.attrs["action_horizon"] = 8
    output.attrs["policy_obs_keys_json"] = json.dumps(list(policy_obs_keys))
    output.attrs["raw_obs_keys_json"] = json.dumps(list(raw_obs_keys))
    output.attrs["source_action_offset"] = np.asarray(
        action_offset, dtype=np.float32
    ).reshape(-1)
    output.attrs["source_action_scale"] = np.asarray(
        action_scale, dtype=np.float32
    ).reshape(-1)
    output.attrs["source_action_stats_sha256"] = sha256_arrays(
        output.attrs["source_action_offset"],
        output.attrs["source_action_scale"],
    )
    data = output.create_group("data")
    data.attrs["env_args"] = json.dumps(env_metadata)
    data.attrs["total"] = 0


def _create_array(group, key, value):
    value = np.asarray(value)
    kwargs = {}
    if value.ndim >= 3:
        kwargs["compression"] = "gzip"
    group.create_dataset(key, data=value, **kwargs)


def append_raw_episode(output, episode_id, episode, attrs):
    """Append one accepted in-memory episode to an initialized raw HDF5."""

    data = output["data"]
    name = "demo_{}".format(episode_id)
    if name in data:
        raise KeyError("Raw episode '{}' already exists".format(name))
    num_samples = len(episode["actions"])
    if num_samples <= 0:
        raise ValueError("Cannot append an empty episode")

    group = data.create_group(name)
    group.attrs["num_samples"] = num_samples
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            group.attrs[key] = json.dumps(
                value,
                default=lambda item: (
                    item.item()
                    if isinstance(item, np.generic)
                    else item.tolist()
                    if isinstance(item, np.ndarray)
                    else str(item)
                ),
            )
        else:
            group.attrs[key] = value

    for key in (
        "states",
        "actions",
        "delta_eef_pose_action",
        "rewards",
        "dones",
        "controller",
        "intervention_id",
        "robot_query_action",
        "robot_query_valid",
        "robot_query_seed",
        "target_contact",
        "distractor_contact",
    ):
        values = np.asarray(episode[key])
        if len(values) != num_samples:
            raise ValueError("{} does not align with actions".format(key))
        _create_array(group, key, values)

    for prefix in ("obs", "next_obs"):
        obs_group = group.create_group(prefix)
        for key, values in episode[prefix].items():
            values = np.asarray(values)
            if len(values) != num_samples:
                raise ValueError(
                    "{}/{} does not align with actions".format(prefix, key)
                )
            _create_array(obs_group, key, values)

    data.attrs.modify("total", int(data.attrs["total"]) + num_samples)
    output.flush()
    return name


def _read_episode(group):
    return {
        "obs": {key: group["obs/{}".format(key)][()] for key in group["obs"]},
        "delta_eef_pose_action": group["delta_eef_pose_action"][()],
        "controller": group["controller"][()],
        "intervention_id": group["intervention_id"][()],
        "robot_query_action": group["robot_query_action"][()],
        "robot_query_valid": group["robot_query_valid"][()],
    }


def sorted_demo_names(data_group):
    return sorted(data_group, key=lambda name: int(name.split("_")[-1]))


def _string_array(values):
    return np.asarray([str(value) for value in values], dtype=object)


def _write_pair_arrays(group, arrays):
    for key, value in arrays.items():
        if key == "obs":
            obs_group = group.create_group("obs")
            for obs_key, obs_value in value.items():
                _create_array(obs_group, obs_key, obs_value)
        elif np.asarray(value).dtype.kind in ("U", "O"):
            group.create_dataset(key, data=_string_array(value), dtype=STRING_DTYPE)
        else:
            _create_array(group, key, value)


def extract_pair_file(raw_path, output_path, overwrite=False):
    """Create a compact pair file from accepted raw episodes."""

    raw_path = Path(raw_path).resolve()
    output_path = Path(output_path).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "{} exists; pass --overwrite to replace it".format(output_path)
        )

    with h5py.File(raw_path, "r") as raw:
        if raw.attrs.get("schema_version") != RAW_SCHEMA_VERSION:
            raise ValueError("Input is not a paired-correction raw file")
        obs_keys = json.loads(str(raw.attrs["policy_obs_keys_json"]))
        episode_names = sorted_demo_names(raw["data"])
        pair_parts = []
        for episode_name in episode_names:
            group = raw["data/{}".format(episode_name)]
            if not bool(group.attrs.get("train_eligible", False)):
                continue
            pairs = derive_episode_pairs(_read_episode(group), obs_keys)
            count = len(pairs["anchor_step"])
            if count == 0:
                raise ValueError(
                    "{} is train-eligible but yields no pairs".format(
                        episode_name
                    )
                )
            pairs["episode_id"] = _string_array([episode_name] * count)
            pairs["layout_id"] = _string_array(
                [group.attrs["layout_id"]] * count
            )
            pairs["selected_yaw_deg"] = np.full(
                count,
                float(group.attrs["selected_yaw_deg"]),
                dtype=np.float32,
            )
            pairs["split"] = _string_array(
                [group.attrs.get("split", "train")] * count
            )
            pair_parts.append(pairs)

        if not pair_parts:
            raise ValueError("No train-eligible correction pairs found")

        combined = {
            "obs": {
                key: np.concatenate(
                    [part["obs"][key] for part in pair_parts], axis=0
                )
                for key in obs_keys
            }
        }
        for key in (
            "positive_actions",
            "negative_actions",
            "anchor_step",
            "intervention_id",
            "episode_id",
            "layout_id",
            "selected_yaw_deg",
            "split",
        ):
            combined[key] = np.concatenate(
                [part[key] for part in pair_parts], axis=0
            )

        root_attrs = {
            "schema_version": PAIR_SCHEMA_VERSION,
            "raw_file_sha256": sha256_file(raw_path),
            "source_checkpoint_sha256": str(
                raw.attrs["source_checkpoint_sha256"]
            ),
            "source_action_stats_sha256": str(
                raw.attrs["source_action_stats_sha256"]
            ),
            "observation_horizon": 2,
            "prediction_horizon": 16,
            "action_horizon": 8,
            "policy_obs_keys_json": json.dumps(obs_keys),
            "source_action_offset": raw.attrs["source_action_offset"],
            "source_action_scale": raw.attrs["source_action_scale"],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as output:
            for key, value in root_attrs.items():
                output.attrs[key] = value
            _write_pair_arrays(output.create_group("pairs"), combined)
            mask = output.create_group("mask")
            episode_ids = np.asarray(combined["episode_id"]).astype(str)
            split = np.asarray(combined["split"]).astype(str)
            for split_name in ("train", "valid"):
                selected = sorted(set(episode_ids[split == split_name]))
                mask.create_dataset(
                    "{}_episode_ids".format(split_name),
                    data=_string_array(selected),
                    dtype=STRING_DTYPE,
                )
            output.attrs["num_pairs"] = len(combined["positive_actions"])
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output": str(output_path),
        "episodes": len(
            set(np.asarray(combined["episode_id"]).astype(str).tolist())
        ),
        "pairs": len(combined["positive_actions"]),
        "train_pairs": int(np.sum(np.asarray(combined["split"]) == "train")),
        "valid_pairs": int(np.sum(np.asarray(combined["split"]) == "valid")),
    }
