"""Shared helpers for delta-EEF guided-denoising diagnostics and rollouts."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from robomimic.utils.guided_denoising_utils import (
    guidance_context_from_rollout_policy,
    guided_policy_from_checkpoint,
)


DEFAULT_AGENT = Path(
    "outputs/robomimic/train/dp_can_delta_pose_osc/20260707222943/models/"
    "model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth"
)
DEFAULT_ENVS = ["PickPlaceBreadCerealCan", "PickPlaceBreadCerealMilkCan"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def raw_env_from_wrapper(env):
    raw = getattr(env, "unwrapped", env)
    return getattr(raw, "env", raw)


def latest_obs_value(obs, key):
    value = np.asarray(obs[key])
    return value[-1] if value.ndim >= 2 else value


def current_eef_position(obs):
    return np.asarray(latest_obs_value(obs, "robot0_eef_pos"), dtype=np.float32).copy()


def object_body_name(obj):
    for attr in ("root_body", "body_name", "root_body_name"):
        value = getattr(obj, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        return value
    return None


def object_is_active(raw_env, obj):
    body_name = object_body_name(obj)
    if body_name is None:
        return False
    try:
        body_id = raw_env.sim.model.body_name2id(body_name)
    except Exception:
        return False
    return bool(raw_env.sim.model.body_pos[body_id][2] > -10.0)


def active_object_records(raw_env):
    records = []
    obj_to_use = getattr(raw_env, "obj_to_use", None)
    single_object_mode = int(getattr(raw_env, "single_object_mode", 0))
    for obj in getattr(raw_env, "objects", []):
        name = getattr(obj, "name", None)
        body_name = object_body_name(obj)
        if name is None or body_name is None:
            continue
        if single_object_mode in (1, 2) and obj_to_use is not None and name != obj_to_use:
            continue
        if not object_is_active(raw_env, obj):
            continue
        body_id = raw_env.sim.model.body_name2id(body_name)
        records.append((name, obj, body_id))
    return records


def oracle_obstacle_geometry(raw_env, object_records, target_object_name="Can"):
    centers = []
    radii = []
    names = []
    target = target_object_name.lower()
    for name, obj, body_id in object_records:
        if name.lower() == target:
            continue
        centers.append(np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=np.float32).copy())
        radii.append(float(obj.horizontal_radius))
        names.append(name)
    if not centers:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            names,
        )
    return np.stack(centers), np.asarray(radii, dtype=np.float32), names


def target_object_position(raw_env, object_records, target_object_name="Can"):
    target = target_object_name.lower()
    for name, _, body_id in object_records:
        if name.lower() == target:
            return np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=np.float32).copy()
    raise KeyError("Target object '{}' is not active".format(target_object_name))


def _geom_ids(raw_env, geom_names):
    result = []
    for name in geom_names:
        try:
            result.append(raw_env.sim.model.geom_name2id(name))
        except Exception:
            pass
    return result


def contact_maps(raw_env, object_records, target_object_name="Can"):
    robot_geom_names = []
    for robot in getattr(raw_env, "robots", []):
        model = getattr(robot, "robot_model", None)
        if model is not None:
            robot_geom_names.extend(getattr(model, "contact_geoms", []))
        gripper = getattr(robot, "gripper", None)
        grippers = gripper.values() if isinstance(gripper, dict) else [gripper]
        for item in grippers:
            if item is not None:
                robot_geom_names.extend(getattr(item, "contact_geoms", []))

    robot_ids = set(_geom_ids(raw_env, robot_geom_names))
    target_ids = set()
    obstacle_by_geom = {}
    target = target_object_name.lower()
    for name, obj, _ in object_records:
        ids = _geom_ids(raw_env, getattr(obj, "contact_geoms", []))
        if name.lower() == target:
            target_ids.update(ids)
        else:
            for geom_id in ids:
                obstacle_by_geom[geom_id] = name
    return robot_ids, target_ids, obstacle_by_geom


def contact_snapshot(raw_env, maps):
    robot_ids, target_ids, obstacle_by_geom = maps
    target_count = 0
    obstacle_count = 0
    names = set()
    for index in range(raw_env.sim.data.ncon):
        contact = raw_env.sim.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        pair = ((geom1, geom2), (geom2, geom1))
        if any(a in robot_ids and b in target_ids for a, b in pair):
            target_count += 1
        for robot_geom, other_geom in pair:
            if robot_geom in robot_ids and other_geom in obstacle_by_geom:
                obstacle_count += 1
                names.add(obstacle_by_geom[other_geom])
                break
    return target_count, obstacle_count, sorted(names)


def minimum_clearance(eef_position, centers, radii, margin=0.0):
    if len(radii) == 0:
        return float("inf")
    distances = np.linalg.norm(centers[:, :2] - np.asarray(eef_position)[:2], axis=1)
    return float(np.min(distances - radii - float(margin)))


def action_limits(env, action_dim):
    raw_env = raw_env_from_wrapper(env)
    low, high = raw_env.action_spec
    low = np.asarray(low, dtype=np.float32).reshape(-1)
    high = np.asarray(high, dtype=np.float32).reshape(-1)
    if low.shape != (action_dim,) or high.shape != (action_dim,):
        raise ValueError("Environment action limits do not match policy action dimension")
    return low, high


def policy_observation(obs, rollout_policy):
    return {
        key: np.asarray(obs[key]).copy()
        for key in rollout_policy.policy.global_config.all_obs_keys
    }


def make_guidance_context(
    rollout_policy,
    obs,
    centers,
    radii,
    scale,
    clearance_margin,
):
    return guidance_context_from_rollout_policy(
        rollout_policy,
        current_eef_pos=current_eef_position(obs),
        obstacle_centers=centers,
        obstacle_radii=radii,
        guidance_scale=scale,
        clearance_margin=clearance_margin,
    )


def sample_action_chunk(rollout_policy, obs, context, noise_seed=None):
    if noise_seed is not None:
        torch.manual_seed(int(noise_seed))
    rollout_policy.policy.set_guidance_context(context)
    prepared = rollout_policy._prepare_observation(obs)
    with torch.no_grad():
        normalized = rollout_policy.policy._get_action_trajectory(prepared)[0]
    normalized_np = normalized.detach().cpu().numpy()
    scale = torch.as_tensor(context.action_scale).reshape(-1).cpu().numpy()
    offset = torch.as_tensor(context.action_offset).reshape(-1).cpu().numpy()
    raw = normalized_np * scale.reshape(1, -1) + offset.reshape(1, -1)
    return normalized_np, raw.astype(np.float32), rollout_policy.policy.get_guidance_diagnostics()


def predict_eef_positions(start_position, raw_action_chunk):
    return np.asarray(start_position, dtype=np.float32)[None] + np.cumsum(
        np.asarray(raw_action_chunk, dtype=np.float32)[:, :3], axis=0
    )


def staged_task_progress(raw_env):
    if not hasattr(raw_env, "staged_rewards"):
        return []
    return [float(value) for value in raw_env.staged_rewards()]


def task_success(env):
    return bool(env.is_success().get("task", False))


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_guided_policy_and_checkpoint(agent, device):
    return guided_policy_from_checkpoint(
        ckpt_path=str(agent),
        device=device,
        verbose=False,
    )
