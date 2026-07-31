from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from .config import EnvironmentConfig
from .data import Trajectory, normalize_actions, write_trajectories
from .env import LayoutSpec, PointNavEnv, make_layout
from .models import SourceDiffusionPolicy
from .training import DynamicsBatch


def rgb_history_tensor(
    history: Sequence[np.ndarray], device: torch.device
) -> torch.Tensor:
    images = list(history[-2:])
    while len(images) < 2:
        images.insert(0, images[0])
    array = np.stack(images)
    return (
        torch.from_numpy(array)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .float()
        .to(device)
        / 255.0
    )


def rollout_policy(
    policy: SourceDiffusionPolicy,
    layout: LayoutSpec,
    *,
    policy_seed: int,
    config: EnvironmentConfig | None = None,
    device: str | torch.device | None = None,
    source_name: str = "source",
) -> Trajectory:
    cfg = config or EnvironmentConfig()
    target_device = torch.device(device or next(policy.parameters()).device)
    env = PointNavEnv(layout=layout, config=cfg)
    obs = env.reset(layout.seed, layout)
    history = [obs["rgb"]]
    rgbs, states = [obs["rgb"]], [obs["state"]]
    actions, collisions, successes = [], [], []
    generator = torch.Generator(device=target_device).manual_seed(policy_seed)
    done = False
    policy.eval()
    while not done:
        rgb = rgb_history_tensor(history, target_device)
        normalized_chunk = policy.sample(rgb, generator=generator)
        chunk = (
            normalized_chunk[0, : policy.config.action_horizon]
            .detach()
            .cpu()
            .numpy()
            * cfg.max_step
        )
        for action in chunk:
            obs, _, done, info = env.step(action)
            history.append(obs["rgb"])
            rgbs.append(obs["rgb"])
            states.append(obs["state"])
            actions.append(info["executed_action"])
            collisions.append(info["collision"])
            successes.append(info["success"])
            if done:
                break
    trajectory = Trajectory(
        rgb=np.asarray(rgbs, dtype=np.uint8),
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        collision=np.asarray(collisions, dtype=np.bool_),
        success=np.asarray(successes, dtype=np.bool_),
        layout=layout,
        policy_seed=policy_seed,
        source=source_name,
    )
    trajectory.validate()
    return trajectory


def collect_dynamics_rollouts(
    checkpoint_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    count: int = 150,
    seed: int = 0,
    config: EnvironmentConfig | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Collect unlabeled D0/D1/D2 rollouts from all five source checkpoints."""

    if len(checkpoint_paths) != 5:
        raise ValueError("dynamics collection requires exactly five checkpoints")
    cfg = config or EnvironmentConfig()
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite dynamics data: {target}")
    policies = [
        SourceDiffusionPolicy.load_checkpoint(path, device=target_device)[0]
        for path in checkpoint_paths
    ]
    if count % 15 != 0:
        raise ValueError("count must balance 3 difficulties x 5 checkpoints")
    trajectories = []
    queries: list[dict[str, object]] = []
    outcome_counts = {
        f"D{difficulty}": {"success": 0, "failure": 0}
        for difficulty in range(3)
    }
    for index in range(count):
        difficulty = index % 3
        checkpoint_index = (index // 3) % 5
        layout_seed = seed * 1_000_000 + 100_000 + index
        policy_seed = seed * 1_000_000 + 200_000 + index
        layout = make_layout(difficulty, layout_seed, cfg)
        policy = policies[checkpoint_index]
        trajectory, rollout_queries = _rollout_with_open_loop_queries(
            policy=policy,
            layout=layout,
            rollout_index=index,
            checkpoint_index=checkpoint_index,
            policy_seed=policy_seed,
            config=cfg,
            device=target_device,
        )
        trajectories.append(trajectory)
        queries.extend(rollout_queries)
        outcome = "success" if trajectory.success.any() else "failure"
        outcome_counts[f"D{difficulty}"][outcome] += 1
    write_trajectories(target, trajectories)
    with h5py.File(target, "a") as handle:
        group = handle.create_group("dynamics_queries")
        group.attrs["schema"] = "open-loop-predicted-chunk-v1"
        group.attrs["absorbing_terminal_state"] = True
        group.create_dataset(
            "current_rgb",
            data=np.stack([query["current_rgb"] for query in queries]),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=1,
        )
        group.create_dataset(
            "future_rgb",
            data=np.stack([query["future_rgb"] for query in queries]),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=1,
        )
        group.create_dataset(
            "normalized_action_chunk",
            data=np.stack([query["normalized_action_chunk"] for query in queries]),
            dtype=np.float32,
        )
        for key in (
            "rollout_index",
            "difficulty",
            "checkpoint_index",
            "layout_seed",
            "policy_seed",
            "main_step",
            "open_loop_executed_steps",
        ):
            group.create_dataset(
                key,
                data=np.asarray([query[key] for query in queries], dtype=np.int64),
            )
    return {
        "schema": "toy-recovery-dynamics-collection-v1",
        "count": count,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "difficulty_counts": {
            f"D{difficulty}": sum(
                1 for trajectory in trajectories if trajectory.layout.difficulty == difficulty
            )
            for difficulty in range(3)
        },
        "outcome_counts": outcome_counts,
        "dynamics_queries": len(queries),
        "dynamics_contract": (
            "clone each main-rollout replan state, execute its predicted "
            "16-action chunk open loop, and encode the absorbing terminal future"
        ),
        "uses_reward_or_correction_labels": False,
    }


def _rollout_with_open_loop_queries(
    *,
    policy: SourceDiffusionPolicy,
    layout: LayoutSpec,
    rollout_index: int,
    checkpoint_index: int,
    policy_seed: int,
    config: EnvironmentConfig,
    device: str | torch.device | None,
) -> tuple[Trajectory, list[dict[str, object]]]:
    target_device = torch.device(device or next(policy.parameters()).device)
    env = PointNavEnv(layout=layout, config=config)
    obs = env.reset(layout.seed, layout)
    history = [obs["rgb"]]
    rgbs, states = [obs["rgb"]], [obs["state"]]
    actions, collisions, successes = [], [], []
    queries: list[dict[str, object]] = []
    generator = torch.Generator(device=target_device).manual_seed(policy_seed)
    done = False
    policy.eval()
    while not done:
        rgb = rgb_history_tensor(history, target_device)
        normalized_chunk = policy.sample(rgb, generator=generator)
        planned = (
            normalized_chunk[0].detach().cpu().numpy() * config.max_step
        )
        norms = np.linalg.norm(planned, axis=1, keepdims=True)
        planned = planned * np.minimum(
            1.0, config.max_step / np.maximum(norms, 1e-12)
        )

        branch = PointNavEnv(layout=layout, config=config)
        branch.restore_state(env.clone_state())
        branch_done = False
        executed_steps = 0
        for action in planned:
            if branch_done:
                break
            _, _, branch_done, _ = branch.step(action)
            executed_steps += 1
        queries.append(
            {
                "current_rgb": history[-1].copy(),
                "future_rgb": branch.render().copy(),
                "normalized_action_chunk": normalize_actions(
                    planned, config.max_step
                ).astype(np.float32),
                "rollout_index": rollout_index,
                "difficulty": layout.difficulty,
                "checkpoint_index": checkpoint_index,
                "layout_seed": layout.seed,
                "policy_seed": policy_seed,
                "main_step": env.state.step_count,
                "open_loop_executed_steps": executed_steps,
            }
        )

        for action in planned[: policy.config.action_horizon]:
            obs, _, done, info = env.step(action)
            history.append(obs["rgb"])
            rgbs.append(obs["rgb"])
            states.append(obs["state"])
            actions.append(info["executed_action"])
            collisions.append(info["collision"])
            successes.append(info["success"])
            if done:
                break
    trajectory = Trajectory(
        rgb=np.asarray(rgbs, dtype=np.uint8),
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        collision=np.asarray(collisions, dtype=np.bool_),
        success=np.asarray(successes, dtype=np.bool_),
        layout=layout,
        policy_seed=policy_seed,
        source=f"source_checkpoint_{checkpoint_index}",
    )
    trajectory.validate()
    return trajectory, queries


def load_dynamics_batches(
    rollout_path: str | Path,
    encoder: torch.nn.Module,
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    split_seed: int = 0,
    validation_fraction: float = 0.2,
) -> tuple[list[DynamicsBatch], list[DynamicsBatch], dict[str, object]]:
    """Encode open-loop queries with a stratified rollout-level split."""

    target_device = torch.device(device)
    encoder.eval()
    with h5py.File(rollout_path, "r") as handle, torch.no_grad():
        queries = handle["dynamics_queries"]
        if queries.attrs.get("schema") != "open-loop-predicted-chunk-v1":
            raise ValueError("dynamics data does not use the open-loop chunk contract")
        rollout_ids = queries["rollout_index"][...]
        difficulties = queries["difficulty"][...]
        checkpoints = queries["checkpoint_index"][...]
        validation_rollouts: set[int] = set()
        rng = np.random.default_rng(split_seed)
        for difficulty in range(3):
            for checkpoint in range(5):
                members = np.unique(
                    rollout_ids[
                        (difficulties == difficulty) & (checkpoints == checkpoint)
                    ]
                )
                if len(members) < 2:
                    raise ValueError(
                        "each difficulty/checkpoint stratum needs at least two rollouts"
                    )
                members = members.copy()
                rng.shuffle(members)
                validation_count = max(1, round(len(members) * validation_fraction))
                validation_rollouts.update(
                    int(value) for value in members[:validation_count]
                )
        is_validation = np.asarray(
            [int(value) in validation_rollouts for value in rollout_ids]
        )
        current_rgb = queries["current_rgb"][...]
        future_rgb = queries["future_rgb"][...]
        action_chunks = queries["normalized_action_chunk"][...]

    def encode(indices: np.ndarray) -> list[DynamicsBatch]:
        result = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                current = (
                    torch.from_numpy(current_rgb[selected])
                    .permute(0, 3, 1, 2)
                    .float()
                    .to(target_device)
                    / 255.0
                )
                future = (
                    torch.from_numpy(future_rgb[selected])
                    .permute(0, 3, 1, 2)
                    .float()
                    .to(target_device)
                    / 255.0
                )
                result.append(
                    DynamicsBatch(
                        latent=encoder(current).cpu(),
                        action=torch.from_numpy(action_chunks[selected].copy()),
                        future_latent=encoder(future).cpu(),
                    )
                )
        return result

    train_indices = np.flatnonzero(~is_validation)
    validation_indices = np.flatnonzero(is_validation)
    manifest = {
        "split_seed": split_seed,
        "split_unit": "rollout",
        "stratified_by": ["difficulty", "checkpoint_index"],
        "uses_outcome_labels": False,
        "train_rollouts": len(set(int(v) for v in rollout_ids[~is_validation])),
        "validation_rollouts": len(validation_rollouts),
        "train_rollout_ids": sorted(
            set(int(v) for v in rollout_ids[~is_validation])
        ),
        "validation_rollout_ids": sorted(validation_rollouts),
        "train_queries": len(train_indices),
        "validation_queries": len(validation_indices),
    }
    return encode(train_indices), encode(validation_indices), manifest


def build_expert_latent_bank(
    d0_demo_path: str | Path,
    encoder: torch.nn.Module,
    output_path: str | Path,
    *,
    stride: int = 4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    target_device = torch.device(device)
    latents = []
    encoder.eval()
    with h5py.File(d0_demo_path, "r") as handle, torch.no_grad():
        for episode_name in sorted(handle["episodes"]):
            episode = handle["episodes"][episode_name]
            layout = json.loads(episode.attrs["layout"])
            if int(layout["difficulty"]) != 0:
                raise ValueError("expert latent bank may only use D0 demonstrations")
            frames = episode["rgb"][::stride]
            images = (
                torch.from_numpy(frames)
                .permute(0, 3, 1, 2)
                .float()
                .to(target_device)
                / 255.0
            )
            latents.append(encoder(images).cpu())
    bank = torch.cat(latents, dim=0)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "toy-recovery-d0-latent-bank-v1",
            "latents": bank,
            "source": str(d0_demo_path),
            "stride": stride,
        },
        target,
    )
    return bank
