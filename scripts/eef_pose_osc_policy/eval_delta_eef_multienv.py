"""
Evaluate a delta-EEF-pose policy across PickPlace variants while recording
trajectory data needed for failure analysis.

The script records per-episode low-dimensional trajectories, object motion,
contacts, obstacle clearances, and diffusion action chunks reconstructed into
EEF pose trajectories. It intentionally does not store image observations.

Smoke test:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 ROBOMIMIC_GPU_ID=0 \
    uv run python scripts/eef_pose_osc_policy/eval_delta_eef_multienv.py \
      --envs PickPlaceCan --seeds 600 --n-rollouts 1 --horizon 24 \
      --out-dir /tmp/delta_eef_eval_smoke --overwrite

Full diagnostic eval:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 ROBOMIMIC_GPU_ID=0 \
    uv run python scripts/eef_pose_osc_policy/eval_delta_eef_multienv.py \
      --seeds 600 601 602 --n-rollouts 50 \
      --out-dir outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from diagnose_delta_eef_policy_traj import (  # noqa: E402
    DEFAULT_AGENT,
    angular_distance_deg,
    as_serializable,
    choose_quat_key,
    clip_action_chunk,
    env_action_limits,
    get_pose_obs,
    pose_error_summary,
    predict_pose_traj,
    sample_action_chunk,
)

import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


DEFAULT_ENVS = [
    "PickPlaceCan",
    "PickPlaceBreadCan",
    "PickPlaceBreadCerealCan",
    "PickPlaceBreadCerealMilkCan",
]
DEFAULT_OUT_DIR = "outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50"
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def git_rev(path):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def shell_command():
    command = shlex.join([sys.executable] + sys.argv)
    prefixes = []
    for key in ("MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "ROBOMIMIC_GPU_ID"):
        if key in os.environ:
            prefixes.append(f"{key}={shlex.quote(os.environ[key])}")
    return " ".join(prefixes + [command])


def json_sanitize(value):
    value = as_serializable(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    return value


def finite_min(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.min(finite)) if finite.size else np.nan


def raw_env_from_wrapper(env):
    raw = getattr(env, "unwrapped", env)
    return getattr(raw, "env", raw)


def sim_geom_id(sim, geom_name):
    try:
        return sim.model.geom_name2id(geom_name)
    except Exception:
        return None


def object_body_name(obj):
    for attr in ("root_body", "body_name", "root_body_name"):
        value = getattr(obj, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            value = value[0]
        return value
    name = getattr(obj, "name", None)
    return f"{name}_main" if name is not None else None


def object_geom_names(obj):
    geom_names = []
    for attr in ("visual_geoms", "contact_geoms"):
        for geom_name in getattr(obj, attr, []):
            if geom_name not in geom_names:
                geom_names.append(geom_name)
    return geom_names


def object_is_active_in_scene(sim, obj):
    body_name = object_body_name(obj)
    if body_name is None:
        return True
    try:
        body_id = sim.model.body_name2id(body_name)
    except Exception:
        return True
    return bool(sim.model.body_pos[body_id][2] > -10.0)


def geom_xy_radius(sim, geom_id):
    size = np.asarray(sim.model.geom_size[geom_id], dtype=np.float32)
    geom_type = int(sim.model.geom_type[geom_id])
    if geom_type in (2, 3, 5):
        return float(size[0])
    if geom_type in (4, 6):
        return float(np.linalg.norm(size[:2]))
    return float(np.linalg.norm(size[:2])) if size.shape[0] >= 2 else float(size[0])


def object_center_and_radius(sim, obj):
    centers = []
    radii = []
    for geom_name in object_geom_names(obj):
        geom_id = sim_geom_id(sim, geom_name)
        if geom_id is None:
            continue
        centers.append(np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float32))
        radii.append(geom_xy_radius(sim, geom_id))
    if centers:
        return np.mean(np.stack(centers, axis=0), axis=0), max(radii)

    body_name = object_body_name(obj)
    if body_name is None:
        return None, None
    try:
        body_id = sim.model.body_name2id(body_name)
    except Exception:
        return None, None
    return np.asarray(sim.data.body_xpos[body_id], dtype=np.float32), 0.0


def active_objects(raw_env):
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        return []
    obj_to_use = getattr(raw_env, "obj_to_use", None)
    single_object_mode = int(getattr(raw_env, "single_object_mode", 0))
    result = []
    for obj in getattr(raw_env, "objects", []):
        name = getattr(obj, "name", None)
        body_name = object_body_name(obj)
        if name is None or body_name is None:
            continue
        if single_object_mode in {1, 2} and obj_to_use is not None and name != obj_to_use:
            continue
        if not object_is_active_in_scene(sim, obj):
            continue
        try:
            body_id = sim.model.body_name2id(body_name)
        except Exception:
            continue
        result.append((name, obj, body_id))
    return result


def object_pose_snapshot(raw_env, object_records):
    sim = raw_env.sim
    pos = []
    quat = []
    for _, _, body_id in object_records:
        pos.append(np.asarray(sim.data.body_xpos[body_id], dtype=np.float32).copy())
        quat.append(np.asarray(sim.data.body_xquat[body_id], dtype=np.float32).copy())
    if not pos:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    return np.stack(pos, axis=0), np.stack(quat, axis=0)


def object_geometry_snapshot(raw_env, object_records, target_object_name):
    sim = raw_env.sim
    target_lower = target_object_name.lower() if target_object_name is not None else None
    centers = []
    radii = []
    names = []
    for name, obj, _ in object_records:
        if target_lower is not None and name.lower() == target_lower:
            continue
        center, radius = object_center_and_radius(sim, obj)
        if center is None:
            continue
        centers.append(center)
        radii.append(radius)
        names.append(name)
    if not centers:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), names
    return np.stack(centers, axis=0).astype(np.float32), np.asarray(radii, dtype=np.float32), names


def contact_geom_ids_for_object(sim, obj):
    geom_ids = []
    for geom_name in getattr(obj, "contact_geoms", []):
        geom_id = sim_geom_id(sim, geom_name)
        if geom_id is not None:
            geom_ids.append(geom_id)
    return sorted(set(geom_ids))


def robot_contact_geom_ids(raw_env):
    sim = raw_env.sim
    geom_names = []
    for robot in getattr(raw_env, "robots", []):
        robot_model = getattr(robot, "robot_model", None)
        if robot_model is not None:
            geom_names.extend(getattr(robot_model, "contact_geoms", []))
        gripper = getattr(robot, "gripper", None)
        grippers = gripper.values() if isinstance(gripper, dict) else [gripper]
        for grip in grippers:
            if grip is not None:
                geom_names.extend(getattr(grip, "contact_geoms", []))
    geom_ids = []
    for geom_name in geom_names:
        geom_id = sim_geom_id(sim, geom_name)
        if geom_id is not None:
            geom_ids.append(geom_id)
    return sorted(set(geom_ids))


def contact_maps(raw_env, object_records, target_object_name):
    sim = raw_env.sim
    target_lower = target_object_name.lower() if target_object_name is not None else None
    robot_geom_ids = set(robot_contact_geom_ids(raw_env))
    target_geom_ids = set()
    non_target_by_geom = {}
    for name, obj, _ in object_records:
        geom_ids = contact_geom_ids_for_object(sim, obj)
        if target_lower is not None and name.lower() == target_lower:
            target_geom_ids.update(geom_ids)
        else:
            for geom_id in geom_ids:
                non_target_by_geom[geom_id] = name
    return robot_geom_ids, target_geom_ids, non_target_by_geom


def contact_snapshot(raw_env, robot_geom_ids, target_geom_ids, non_target_by_geom):
    sim = raw_env.sim
    target_count = 0
    non_target_count = 0
    non_target_names = set()
    for contact_i in range(sim.data.ncon):
        contact = sim.data.contact[contact_i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in robot_geom_ids and geom2 in target_geom_ids:
            target_count += 1
        elif geom2 in robot_geom_ids and geom1 in target_geom_ids:
            target_count += 1
        elif geom1 in robot_geom_ids and geom2 in non_target_by_geom:
            non_target_count += 1
            non_target_names.add(non_target_by_geom[geom2])
        elif geom2 in robot_geom_ids and geom1 in non_target_by_geom:
            non_target_count += 1
            non_target_names.add(non_target_by_geom[geom1])
    return target_count, non_target_count, sorted(non_target_names)


def latest_obs_value(obs, key, default=None):
    if key not in obs:
        return default
    value = np.asarray(obs[key])
    if value.ndim >= 2:
        value = value[-1]
    return value


def min_eef_obstacle_clearance(eef_pos, centers, radii):
    if centers.shape[0] == 0:
        return np.nan
    dists = np.linalg.norm(centers[:, :2] - eef_pos[:2], axis=1) - radii
    return float(np.min(dists))


def write_dataset(group, name, data, **kwargs):
    arr = np.asarray(data)
    group.create_dataset(name, data=arr, **kwargs)


def write_string_dataset(group, name, values):
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=STRING_DTYPE)


def task_success(env):
    return bool(env.is_success().get("task", False))


def rollout_episode(policy, env, horizon, quat_key, target_object_name, terminate_on_success, save_states):
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)
    quat_key = choose_quat_key(obs, quat_key)
    raw_env = raw_env_from_wrapper(env)
    object_records = active_objects(raw_env)
    object_names = [name for name, _, _ in object_records]
    initial_object_pos, initial_object_quat = object_pose_snapshot(raw_env, object_records)
    robot_geom_ids, target_geom_ids, non_target_by_geom = contact_maps(raw_env, object_records, target_object_name)
    low, high = env_action_limits(env, action_dim=policy.policy.ac_dim)

    eef_pos0, eef_quat0 = get_pose_obs(obs, quat_key)
    gripper0 = latest_obs_value(obs, "robot0_gripper_qpos", default=np.zeros((0,), dtype=np.float32))

    traj = {
        "raw_actions": [],
        "clipped_actions": [],
        "action_clip_delta_abs": [],
        "eef_pos": [eef_pos0],
        "eef_quat_xyzw": [eef_quat0],
        "gripper_qpos": [np.asarray(gripper0, dtype=np.float32).reshape(-1)],
        "object_pos": [initial_object_pos],
        "object_quat_wxyz": [initial_object_quat],
        "states": [state_dict["states"]] if save_states else None,
        "rewards": [],
        "dones": [],
        "target_contact_count": [],
        "non_target_contact_count": [],
        "non_target_contact_names": [],
        "min_eef_obstacle_clearance": [],
        "chunks": [],
    }
    all_raw_pos_err = []
    all_raw_ori_err = []
    all_clipped_pos_err = []
    all_clipped_ori_err = []
    touched_non_targets = set()
    total_reward = 0.0
    step_i = 0
    success = task_success(env)

    while step_i < horizon:
        start_step = step_i
        start_pos, start_quat = get_pose_obs(obs, quat_key)
        raw_chunk = sample_action_chunk(policy, obs)
        clipped_chunk = clip_action_chunk(raw_chunk, low, high)
        raw_pred_pos, raw_pred_quat = predict_pose_traj(start_pos, start_quat, raw_chunk)
        clipped_pred_pos, clipped_pred_quat = predict_pose_traj(start_pos, start_quat, clipped_chunk)

        chunk_actual_pos = []
        chunk_actual_quat = []
        chunk_dones = []
        for chunk_i in range(raw_chunk.shape[0]):
            if step_i >= horizon:
                break
            raw_action = raw_chunk[chunk_i]
            clipped_action = clipped_chunk[chunk_i]
            next_obs, reward, done, _ = env.step(raw_action)
            total_reward += float(reward)
            success = success or task_success(env)

            pos, quat = get_pose_obs(next_obs, quat_key)
            gripper = latest_obs_value(next_obs, "robot0_gripper_qpos", default=np.zeros((0,), dtype=np.float32))
            object_pos, object_quat = object_pose_snapshot(raw_env, object_records)
            centers, radii, _ = object_geometry_snapshot(raw_env, object_records, target_object_name)
            target_contacts, non_target_contacts, contact_names = contact_snapshot(
                raw_env=raw_env,
                robot_geom_ids=robot_geom_ids,
                target_geom_ids=target_geom_ids,
                non_target_by_geom=non_target_by_geom,
            )
            touched_non_targets.update(contact_names)

            traj["raw_actions"].append(raw_action)
            traj["clipped_actions"].append(clipped_action)
            traj["action_clip_delta_abs"].append(np.abs(raw_action - clipped_action))
            traj["eef_pos"].append(pos)
            traj["eef_quat_xyzw"].append(quat)
            traj["gripper_qpos"].append(np.asarray(gripper, dtype=np.float32).reshape(-1))
            traj["object_pos"].append(object_pos)
            traj["object_quat_wxyz"].append(object_quat)
            if save_states:
                traj["states"].append(env.get_state()["states"])
            traj["rewards"].append(float(reward))
            traj["dones"].append(bool(done))
            traj["target_contact_count"].append(int(target_contacts))
            traj["non_target_contact_count"].append(int(non_target_contacts))
            traj["non_target_contact_names"].append(contact_names)
            traj["min_eef_obstacle_clearance"].append(min_eef_obstacle_clearance(pos, centers, radii))
            chunk_actual_pos.append(pos)
            chunk_actual_quat.append(quat)
            chunk_dones.append(bool(done))

            obs = next_obs
            step_i += 1
            if done or (terminate_on_success and success):
                break

        executed_len = len(chunk_actual_pos)
        actual_pos = np.asarray(chunk_actual_pos, dtype=np.float64)
        actual_quat = np.asarray(chunk_actual_quat, dtype=np.float64)
        raw_pos_err = np.linalg.norm(raw_pred_pos[:executed_len] - actual_pos, axis=1) * 100.0
        clipped_pos_err = np.linalg.norm(clipped_pred_pos[:executed_len] - actual_pos, axis=1) * 100.0
        raw_ori_err = np.asarray(
            [angular_distance_deg(raw_pred_quat[i], actual_quat[i]) for i in range(executed_len)],
            dtype=np.float64,
        )
        clipped_ori_err = np.asarray(
            [angular_distance_deg(clipped_pred_quat[i], actual_quat[i]) for i in range(executed_len)],
            dtype=np.float64,
        )
        all_raw_pos_err.extend(raw_pos_err.tolist())
        all_raw_ori_err.extend(raw_ori_err.tolist())
        all_clipped_pos_err.extend(clipped_pos_err.tolist())
        all_clipped_ori_err.extend(clipped_ori_err.tolist())

        clip_abs_delta = np.abs(raw_chunk - clipped_chunk)
        traj["chunks"].append(
            {
                "start_step": int(start_step),
                "executed_len": int(executed_len),
                "action_horizon": int(raw_chunk.shape[0]),
                "start_pos": start_pos,
                "start_quat_xyzw": start_quat,
                "raw_action_chunk": raw_chunk,
                "clipped_action_chunk": clipped_chunk,
                "raw_pred_eef_pos": raw_pred_pos,
                "raw_pred_eef_quat_xyzw": raw_pred_quat,
                "clipped_pred_eef_pos": clipped_pred_pos,
                "clipped_pred_eef_quat_xyzw": clipped_pred_quat,
                "actual_eef_pos_executed": actual_pos,
                "actual_eef_quat_xyzw_executed": actual_quat,
                "raw_pos_err_cm": raw_pos_err,
                "raw_ori_err_deg": raw_ori_err,
                "clipped_pos_err_cm": clipped_pos_err,
                "clipped_ori_err_deg": clipped_ori_err,
                "action_clip_count": int(np.sum(clip_abs_delta > 1e-5)),
                "action_clip_max_abs_delta": float(np.max(clip_abs_delta)) if clip_abs_delta.size else 0.0,
                "dones": chunk_dones,
            }
        )
        if executed_len == 0 or bool(chunk_dones[-1]) or (terminate_on_success and success):
            break

    raw_actions = np.asarray(traj["raw_actions"], dtype=np.float32)
    action_clip_delta_abs = np.asarray(traj["action_clip_delta_abs"], dtype=np.float32)
    object_pos_arr = np.asarray(traj["object_pos"], dtype=np.float32)
    object_disp = np.linalg.norm(object_pos_arr - object_pos_arr[0:1], axis=-1) if object_pos_arr.size else np.zeros((0, 0))
    target_idx = next((i for i, name in enumerate(object_names) if name.lower() == target_object_name.lower()), None)
    non_target_indices = [i for i in range(len(object_names)) if i != target_idx]
    target_disp = object_disp[:, target_idx] if target_idx is not None and object_disp.size else np.asarray([])
    non_target_disp = object_disp[:, non_target_indices] if non_target_indices and object_disp.size else np.zeros((object_disp.shape[0], 0))
    first_non_target_collision_step = next(
        (i for i, count in enumerate(traj["non_target_contact_count"]) if count > 0),
        None,
    )

    summary = {
        "success": bool(success),
        "horizon": int(step_i),
        "return": float(total_reward),
        "num_chunks": len(traj["chunks"]),
        "quat_obs_key": quat_key,
        "action_clip_count": int(np.sum(action_clip_delta_abs > 1e-5)) if action_clip_delta_abs.size else 0,
        "action_clip_max_abs_delta": float(np.max(action_clip_delta_abs)) if action_clip_delta_abs.size else 0.0,
        "target_contact_count": int(np.sum(traj["target_contact_count"])),
        "non_target_contact_count": int(np.sum(traj["non_target_contact_count"])),
        "non_target_collision_any": bool(np.any(np.asarray(traj["non_target_contact_count"]) > 0)),
        "first_non_target_collision_step": first_non_target_collision_step,
        "non_target_contact_names": sorted(touched_non_targets),
        "min_eef_obstacle_clearance": finite_min(traj["min_eef_obstacle_clearance"]),
        "target_max_displacement": float(np.max(target_disp)) if target_disp.size else np.nan,
        "non_target_max_displacement": float(np.max(non_target_disp)) if non_target_disp.size else 0.0,
        "raw_chunk_error": pose_error_summary(all_raw_pos_err, all_raw_ori_err),
        "clipped_chunk_error": pose_error_summary(all_clipped_pos_err, all_clipped_ori_err),
        "object_names": object_names,
    }
    return summary, traj


def write_episode_hdf5(root_group, episode_key, summary, traj, save_states):
    ep = root_group.create_group(episode_key)
    ep.attrs["success"] = int(summary["success"])
    ep.attrs["horizon"] = summary["horizon"]
    ep.attrs["return"] = summary["return"]
    ep.attrs["num_chunks"] = summary["num_chunks"]

    write_dataset(ep, "raw_actions", traj["raw_actions"])
    write_dataset(ep, "clipped_actions", traj["clipped_actions"])
    write_dataset(ep, "action_clip_delta_abs", traj["action_clip_delta_abs"])
    write_dataset(ep, "rewards", traj["rewards"])
    write_dataset(ep, "dones", traj["dones"])
    write_dataset(ep, "eef_pos", traj["eef_pos"])
    write_dataset(ep, "eef_quat_xyzw", traj["eef_quat_xyzw"])
    write_dataset(ep, "gripper_qpos", traj["gripper_qpos"])
    write_dataset(ep, "object_pos", traj["object_pos"])
    write_dataset(ep, "object_quat_wxyz", traj["object_quat_wxyz"])
    object_pos = np.asarray(traj["object_pos"], dtype=np.float32)
    object_disp = np.linalg.norm(object_pos - object_pos[0:1], axis=-1) if object_pos.size else np.zeros((0, 0))
    write_dataset(ep, "object_displacement_from_initial", object_disp)
    write_string_dataset(ep, "object_names", summary["object_names"])
    write_dataset(ep, "target_contact_count", traj["target_contact_count"])
    write_dataset(ep, "non_target_contact_count", traj["non_target_contact_count"])
    write_dataset(ep, "min_eef_obstacle_clearance", traj["min_eef_obstacle_clearance"])
    write_string_dataset(ep, "non_target_contact_names_per_step", [",".join(names) for names in traj["non_target_contact_names"]])
    if save_states and traj["states"] is not None:
        write_dataset(ep, "states", traj["states"])

    chunks_group = ep.create_group("chunks")
    for chunk_i, chunk in enumerate(traj["chunks"]):
        grp = chunks_group.create_group(f"chunk_{chunk_i}")
        grp.attrs["start_step"] = chunk["start_step"]
        grp.attrs["executed_len"] = chunk["executed_len"]
        grp.attrs["action_horizon"] = chunk["action_horizon"]
        grp.attrs["action_clip_count"] = chunk["action_clip_count"]
        grp.attrs["action_clip_max_abs_delta"] = chunk["action_clip_max_abs_delta"]
        for key in (
            "start_pos",
            "start_quat_xyzw",
            "raw_action_chunk",
            "clipped_action_chunk",
            "raw_pred_eef_pos",
            "raw_pred_eef_quat_xyzw",
            "clipped_pred_eef_pos",
            "clipped_pred_eef_quat_xyzw",
            "actual_eef_pos_executed",
            "actual_eef_quat_xyzw_executed",
            "raw_pos_err_cm",
            "raw_ori_err_deg",
            "clipped_pos_err_cm",
            "clipped_ori_err_deg",
            "dones",
        ):
            write_dataset(grp, key, chunk[key])


def aggregate_summaries(summaries):
    if not summaries:
        return {}

    def mean(key):
        return float(np.mean([s[key] for s in summaries]))

    def max_value(key):
        return float(np.nanmax([s[key] for s in summaries]))

    return {
        "num_episodes": len(summaries),
        "success_rate": mean("success"),
        "mean_horizon": mean("horizon"),
        "mean_return": mean("return"),
        "non_target_collision_rate": mean("non_target_collision_any"),
        "mean_non_target_contact_count": mean("non_target_contact_count"),
        "mean_target_contact_count": mean("target_contact_count"),
        "min_eef_obstacle_clearance": finite_min([s["min_eef_obstacle_clearance"] for s in summaries]),
        "max_target_displacement": max_value("target_max_displacement"),
        "max_non_target_displacement": max_value("non_target_max_displacement"),
        "action_clip_count": int(sum(s["action_clip_count"] for s in summaries)),
        "action_clip_max_abs_delta": max_value("action_clip_max_abs_delta"),
        "raw_pos_err_mean_cm": float(np.nanmean([s["raw_chunk_error"]["pos_cm"]["mean"] for s in summaries])),
        "raw_ori_err_mean_deg": float(np.nanmean([s["raw_chunk_error"]["ori_deg"]["mean"] for s in summaries])),
    }


def write_stats_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_sanitize(payload), f, indent=2)


def write_summary_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_md(path, rows):
    lines = ["# Delta EEF Multi-Env Eval Summary", ""]
    if not rows:
        lines.append("No rows.")
    else:
        headers = ["env", "seed", "num_episodes", "success_rate", "non_target_collision_rate", "mean_horizon"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append(
                "| {env} | {seed} | {num_episodes} | {success_rate:.3f} | "
                "{non_target_collision_rate:.3f} | {mean_horizon:.1f} |".format(**row)
            )
    path.write_text("\n".join(lines) + "\n")


def close_env(env):
    raw_env = raw_env_from_wrapper(env)
    close_fn = getattr(raw_env, "close", None)
    if close_fn is not None:
        close_fn()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=Path(DEFAULT_AGENT))
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[600, 601, 602])
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--quat-obs-key", type=str, default="robot0_eef_quat_site")
    parser.add_argument("--target-object-name", type=str, default="Can")
    parser.add_argument("--terminate-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-states", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.out_dir} already exists and is not empty; pass --overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "agent": str(args.agent),
        "envs": args.envs,
        "seeds": args.seeds,
        "n_rollouts": args.n_rollouts,
        "horizon": args.horizon,
        "target_object_name": args.target_object_name,
        "terminate_on_success": args.terminate_on_success,
        "save_states": args.save_states,
        "command": shell_command(),
        "git_commit": git_rev(repo_root),
        "robomimic_commit": git_rev(repo_root / "third_party" / "robomimic"),
    }
    write_stats_json(args.out_dir / "manifest.json", manifest)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(args.agent), device=device, verbose=False)

    summary_rows = []
    for env_name in args.envs:
        for seed in args.seeds:
            print(f"=== env={env_name} seed={seed} ===", flush=True)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=ckpt_dict,
                env_name=env_name,
                render=False,
                render_offscreen=False,
                verbose=False,
            )
            seed_dir = args.out_dir / env_name / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            episode_summaries = []
            hdf5_path = seed_dir / "episodes.hdf5"
            jsonl_path = seed_dir / "episode_metrics.jsonl"
            with h5py.File(hdf5_path, "w") as h5, open(jsonl_path, "w") as jsonl:
                h5.attrs["env"] = env_name
                h5.attrs["seed"] = seed
                h5.attrs["agent"] = str(args.agent)
                h5.attrs["horizon"] = args.horizon
                data_group = h5.create_group("episodes")
                for rollout_i in range(args.n_rollouts):
                    print(f"{env_name} seed={seed} rollout {rollout_i + 1}/{args.n_rollouts}", flush=True)
                    summary, traj = rollout_episode(
                        policy=policy,
                        env=env,
                        horizon=args.horizon,
                        quat_key=args.quat_obs_key,
                        target_object_name=args.target_object_name,
                        terminate_on_success=args.terminate_on_success,
                        save_states=args.save_states,
                    )
                    summary = {"episode": rollout_i, "env": env_name, "seed": seed, **summary}
                    episode_summaries.append(summary)
                    jsonl.write(json.dumps(json_sanitize(summary)) + "\n")
                    jsonl.flush()
                    write_episode_hdf5(data_group, f"episode_{rollout_i}", summary, traj, save_states=args.save_states)

            aggregate = aggregate_summaries(episode_summaries)
            stats = {"env": env_name, "seed": seed, "aggregate": aggregate, "episodes": episode_summaries}
            write_stats_json(seed_dir / "stats.json", stats)
            row = {"env": env_name, "seed": seed, **aggregate}
            summary_rows.append(row)
            print(
                "done env={} seed={} success_rate={:.3f} collision_rate={:.3f}".format(
                    env_name,
                    seed,
                    row["success_rate"],
                    row["non_target_collision_rate"],
                ),
                flush=True,
            )
            close_env(env)

    write_summary_csv(args.out_dir / "summary.csv", summary_rows)
    write_summary_md(args.out_dir / "summary.md", summary_rows)
    print(f"Wrote eval outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
