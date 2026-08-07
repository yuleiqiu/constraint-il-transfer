"""Paired baseline / guided rollout implementation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .environment import EnvEmpty2D, EnvEmpty2DExtraSphere, EnvEmpty2DExtraSquare
from .metrics import paired_summary


ENVIRONMENTS = {
    "circle": EnvEmpty2DExtraSphere,
    "square": EnvEmpty2DExtraSquare,
    "empty": EnvEmpty2D,
}


def load_guidable_policy(checkpoint: Path, device: torch.device):
    from robomimic.utils.paper_lan_guidance_utils import point_guided_policy_from_checkpoint

    policy, checkpoint_dict = point_guided_policy_from_checkpoint(
        ckpt_path=str(checkpoint),
        device=device,
    )
    return policy, checkpoint_dict


def _raw_predicted_chunk(policy) -> np.ndarray | None:
    normalized = policy.policy.last_predicted_action_chunk
    if normalized is None:
        return None
    normalized = normalized[0].numpy()
    stats = policy.action_normalization_stats["actions"]
    scale = np.asarray(stats["scale"]).reshape(-1)
    offset = np.asarray(stats["offset"]).reshape(-1)
    return normalized * scale + offset


def rollout(
    policy,
    *,
    environment_name: str,
    policy_seed: int,
    guidance_scale: float,
    safety_distance: float,
    record_chunks: bool = False,
) -> dict:
    from robomimic.utils.paper_lan_guidance_utils import (
        point_trajectory_guidance_context_from_rollout_policy,
    )

    environment = ENVIRONMENTS[environment_name]()
    observation = environment.reset()
    observation_history = [observation["agent_pos"].copy()] * 2
    policy.start_episode()
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)

    chunks = []
    active_steps = 0
    diagnostic_steps = 0
    chunk_count = 0
    triggered_chunks = 0
    while True:
        new_chunk = len(policy.policy.action_queue) == 0
        if new_chunk:
            context = point_trajectory_guidance_context_from_rollout_policy(
                policy,
                current_position=environment.position,
                obstacle_points=environment.oracle_obstacle_points(),
                guidance_scale=guidance_scale,
                safety_distance=safety_distance,
                action_key="actions",
                position_indices=(0, 1),
                enabled=guidance_scale > 0,
            )
            policy.policy.set_guidance_context(context)
        policy_observation = {
            "agent_pos": np.stack(observation_history, axis=0),
        }
        action = policy(policy_observation)
        if new_chunk:
            diagnostics = policy.policy.get_guidance_diagnostics()
            active = sum(item["active_waypoint_count"] > 0 for item in diagnostics)
            active_steps += active
            diagnostic_steps += len(diagnostics)
            chunk_count += 1
            triggered_chunks += int(active > 0)
            if record_chunks:
                points = environment.oracle_obstacle_points()
                closest = None
                if len(points):
                    closest = points[
                        np.argmin(np.linalg.norm(points - environment.position[None], axis=1))
                    ].tolist()
                chunks.append(
                    {
                        "start_step": environment.step_count,
                        "predicted_chunk": (
                            None
                            if _raw_predicted_chunk(policy) is None
                            else _raw_predicted_chunk(policy).tolist()
                        ),
                        "closest_obstacle_point": closest,
                        "diagnostics": diagnostics,
                    }
                )
        observation, _, done, info = environment.step(action)
        observation_history.append(observation["agent_pos"].copy())
        observation_history = observation_history[-2:]
        if done:
            break

    trajectory = np.asarray(environment.trajectory)
    length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    row = {
        "environment": environment_name,
        "policy_seed": int(policy_seed),
        "guidance_scale": float(guidance_scale),
        "safety_distance": float(safety_distance),
        "success": bool(environment.is_success()),
        "collision_any": bool(environment.collision_any),
        "safe_success": bool(environment.is_success() and not environment.collision_any),
        "ncr": bool((not environment.is_success()) and (not environment.collision_any)),
        "steps": environment.step_count,
        "minimum_clearance": float(environment.minimum_clearance),
        "trigger_rate": triggered_chunks / chunk_count if chunk_count else 0.0,
        "reverse_step_trigger_rate": (
            active_steps / diagnostic_steps if diagnostic_steps else 0.0
        ),
        "trajectory_length": length,
        "clip_rate": environment.clip_count / environment.step_count,
    }
    if record_chunks:
        row["trajectory"] = trajectory.tolist()
        row["chunks"] = chunks
    return row


def evaluate_pairs(
    checkpoint: Path,
    *,
    environment_name: str,
    seeds: list[int],
    guidance_scale: float,
    safety_distance: float,
    device: torch.device,
    record_chunks: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    baseline_policy, _ = load_guidable_policy(checkpoint, device)
    guided_policy, _ = load_guidable_policy(checkpoint, device)
    baseline = []
    guided = []
    for seed in seeds:
        baseline.append(
            rollout(
                baseline_policy,
                environment_name=environment_name,
                policy_seed=seed,
                guidance_scale=0.0,
                safety_distance=safety_distance,
                record_chunks=record_chunks,
            )
        )
        guided.append(
            rollout(
                guided_policy,
                environment_name=environment_name,
                policy_seed=seed,
                guidance_scale=guidance_scale,
                safety_distance=safety_distance,
                record_chunks=record_chunks,
            )
        )
    return baseline, guided, paired_summary(baseline, guided)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=True, sort_keys=True) + "\n")
