"""Shared environment and policy helpers for sample-refine experiments."""

import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from robomimic.utils.sample_refine_utils import (
    sample_refine_context_from_rollout_policy,
    sample_refine_policy_from_checkpoint,
)


DEFAULT_AGENT = Path(
    "outputs/robomimic/train/dp_can_delta_pose_osc/20260707222943/models/"
    "model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth"
)


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
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is not None:
            return value
    return None


def active_object_records(raw_env):
    records = []
    selected = getattr(raw_env, "obj_to_use", None)
    single_mode = int(getattr(raw_env, "single_object_mode", 0))
    for obj in getattr(raw_env, "objects", []):
        name = getattr(obj, "name", None)
        body_name = object_body_name(obj)
        if name is None or body_name is None:
            continue
        if single_mode in (1, 2) and selected is not None and name != selected:
            continue
        body_id = raw_env.sim.model.body_name2id(body_name)
        if raw_env.sim.model.body_pos[body_id][2] <= -10.0:
            continue
        records.append((name, obj, body_id))
    return records


def target_record(raw_env, records, target_object_name="Can"):
    target = target_object_name.lower()
    for name, obj, body_id in records:
        if name.lower() == target:
            return name, obj, body_id
    raise KeyError("Target object '{}' is not active".format(target_object_name))


def object_position(raw_env, record):
    return np.asarray(raw_env.sim.data.body_xpos[record[2]], dtype=np.float32).copy()


def target_box_position(raw_env, target_object_name="Can"):
    if hasattr(raw_env, "canonical_object_bins"):
        index = raw_env.canonical_object_bins[target_object_name]
    else:
        index = raw_env.object_to_id[target_object_name.lower()]
    return np.asarray(raw_env.target_bin_placements[index], dtype=np.float32).copy()


def oracle_obstacle_geometry(raw_env, records, target_object_name="Can"):
    centers, radii, names = [], [], []
    target = target_object_name.lower()
    for name, obj, body_id in records:
        if name.lower() == target:
            continue
        centers.append(np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=np.float32))
        radii.append(float(obj.horizontal_radius))
        names.append(name)
    if not centers:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32), names
    return np.stack(centers), np.asarray(radii, np.float32), names


def target_is_grasped(raw_env, target):
    return bool(
        raw_env._check_grasp(
            gripper=raw_env.robots[0].gripper,
            object_geoms=target[1].contact_geoms,
        )
    )


def _geom_ids(raw_env, names):
    result = []
    for name in names:
        try:
            result.append(raw_env.sim.model.geom_name2id(name))
        except Exception:
            pass
    return result


def contact_maps(raw_env, records, target_object_name="Can"):
    robot_names = []
    for robot in raw_env.robots:
        robot_names.extend(getattr(robot.robot_model, "contact_geoms", []))
        grippers = robot.gripper.values() if isinstance(robot.gripper, dict) else [robot.gripper]
        for gripper in grippers:
            robot_names.extend(getattr(gripper, "contact_geoms", []))
    robot_ids = set(_geom_ids(raw_env, robot_names))
    target_ids = set()
    obstacle_by_geom = {}
    for name, obj, _ in records:
        ids = _geom_ids(raw_env, obj.contact_geoms)
        if name.lower() == target_object_name.lower():
            target_ids.update(ids)
        else:
            obstacle_by_geom.update({geom_id: name for geom_id in ids})
    return robot_ids, target_ids, obstacle_by_geom


def contact_snapshot(raw_env, maps):
    robot_ids, target_ids, obstacle_by_geom = maps
    target_count = 0
    obstacle_names = set()
    for index in range(raw_env.sim.data.ncon):
        contact = raw_env.sim.data.contact[index]
        pairs = ((int(contact.geom1), int(contact.geom2)), (int(contact.geom2), int(contact.geom1)))
        if any(a in robot_ids and b in target_ids for a, b in pairs):
            target_count += 1
        for robot_geom, other_geom in pairs:
            if robot_geom in robot_ids and other_geom in obstacle_by_geom:
                obstacle_names.add(obstacle_by_geom[other_geom])
    return target_count, sorted(obstacle_names)


def action_limits(env, action_dim):
    low, high = raw_env_from_wrapper(env).action_spec
    low = np.asarray(low, np.float32).reshape(-1)
    high = np.asarray(high, np.float32).reshape(-1)
    if low.shape != (action_dim,) or high.shape != (action_dim,):
        raise ValueError("Environment action limits do not match policy action dimension")
    return low, high


def make_context(
    rollout_policy,
    env,
    obs,
    *,
    candidate_count,
    margin,
    refine_timestep,
    guidance_scale,
    candidate_seed,
    refinement_seed,
    sample_enabled,
    refinement_enabled,
    target_object_name="Can",
    eef_radius=0.0,
):
    raw_env = raw_env_from_wrapper(env)
    records = active_object_records(raw_env)
    target = target_record(raw_env, records, target_object_name)
    centers, radii, _ = oracle_obstacle_geometry(raw_env, records, target_object_name)
    low, high = action_limits(env, rollout_policy.policy.ac_dim)
    return sample_refine_context_from_rollout_policy(
        rollout_policy,
        current_eef_pos=current_eef_position(obs),
        can_pos=object_position(raw_env, target),
        target_box_pos=target_box_position(raw_env, target_object_name),
        grasped=target_is_grasped(raw_env, target),
        obstacle_centers=centers,
        obstacle_radii=radii,
        candidate_count=candidate_count,
        clearance_margin=margin,
        refine_timestep=refine_timestep,
        guidance_scale=guidance_scale,
        candidate_seed=candidate_seed,
        refinement_seed=refinement_seed,
        sample_enabled=sample_enabled,
        refinement_enabled=refinement_enabled,
        eef_radius=eef_radius,
        can_radius=float(target[1].horizontal_radius),
        action_low=low,
        action_high=high,
    )


def policy_observation(obs, rollout_policy):
    return {
        key: np.asarray(obs[key]).copy()
        for key in rollout_policy.policy.global_config.all_obs_keys
    }


def sample_action_chunk(rollout_policy, obs, context):
    rollout_policy.policy.set_sample_refine_context(context)
    prepared = rollout_policy._prepare_observation(obs)
    with torch.no_grad():
        normalized = rollout_policy.policy._get_action_trajectory(prepared)[0]
    normalized_np = normalized.detach().cpu().numpy()
    scale = torch.as_tensor(context.action_scale).reshape(-1).cpu().numpy()
    offset = torch.as_tensor(context.action_offset).reshape(-1).cpu().numpy()
    raw = normalized_np * scale[None] + offset[None]
    return normalized_np, raw.astype(np.float32), rollout_policy.policy.get_sample_refine_diagnostics()


def regime(progress, clearance):
    if progress <= 0:
        return "non_target_directed"
    if clearance <= 0:
        return "target_directed_obstructed"
    return "target_directed_clear"


def task_success(env):
    return bool(env.is_success().get("task", False))


def state_hash(state):
    digest = hashlib.sha256()
    digest.update(np.asarray(state["states"]).tobytes())
    digest.update(state["model"].encode("utf-8"))
    digest.update(state.get("ep_meta", "").encode("utf-8"))
    return digest.hexdigest()


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


def load_policy(agent, device):
    return sample_refine_policy_from_checkpoint(
        ckpt_path=str(agent), device=device, verbose=False
    )
