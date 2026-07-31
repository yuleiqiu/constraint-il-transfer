#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from toy_recovery.config import EnvironmentConfig, PolicyConfig
from toy_recovery.data import (
    TrajectoryWindowDataset,
    generate_oracle_demonstrations,
    replay_trajectory,
    write_trajectories,
)
from toy_recovery.env import PointNavEnv, make_layout
from toy_recovery.evaluation import SOURCE_METHOD, evaluate_episode
from toy_recovery.guidance import (
    GeometryObjective,
    GlobalLatentObjective,
    geometry_context,
    objective_cost_and_gradient_finite,
)
from toy_recovery.models import SourceDiffusionPolicy
from toy_recovery.rollout import (
    build_expert_latent_bank,
    collect_dynamics_rollouts,
    load_dynamics_batches,
)
from toy_recovery.training import train_latent_dynamics, train_source_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    owned_temp = None
    if args.work_dir is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="toy-recovery-smoke-")
        work_dir = Path(owned_temp.name)
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env_config = EnvironmentConfig()
    policy_config = PolicyConfig(latent_dim=16, unet_down_dims=(16, 32))

    demos = generate_oracle_demonstrations(
        6, difficulty=0, seed_start=900_000, config=env_config
    )
    assert all(replay_trajectory(trajectory, env_config) for trajectory in demos)
    data_path = work_dir / "d0_smoke.hdf5"
    write_trajectories(data_path, demos)
    dataset = TrajectoryWindowDataset(
        [data_path], policy_config, max_step=env_config.max_step
    )
    result = train_source_policy(
        dataset,
        work_dir / "source",
        seed=7,
        updates=5,
        batch_size=4,
        device=device,
    )
    assert len(result.checkpoint_paths) == 5
    resumed = train_source_policy(
        dataset,
        work_dir / "source_resume",
        seed=7,
        updates=10,
        batch_size=4,
        resume_checkpoint=result.checkpoint_paths[-1],
        device=device,
    )
    assert resumed.checkpoint_paths[-1].endswith("source_update_000010.pth")
    policy, checkpoint_payload = SourceDiffusionPolicy.load_checkpoint(
        result.checkpoint_paths[-1], device=device
    )
    assert checkpoint_payload["weights"] == "ema"
    dynamics_data = work_dir / "dynamics_smoke.hdf5"
    collection = collect_dynamics_rollouts(
        result.checkpoint_paths,
        dynamics_data,
        count=30,
        seed=8,
        config=env_config,
        device=device,
    )
    train_batches, validation_batches, split_manifest = load_dynamics_batches(
        dynamics_data,
        policy.encoder,
        batch_size=64,
        device=device,
        split_seed=8,
    )
    dynamics, dynamics_metrics = train_latent_dynamics(
        train_batches,
        validation_batches,
        work_dir / "latent_dynamics.pth",
        latent_dim=policy_config.latent_dim,
        seed=8,
        epochs=3,
        early_stopping_patience=2,
        shuffle_permutations=2,
        device=device,
    )
    assert collection["dynamics_contract"].startswith("clone each")
    assert split_manifest["train_rollouts"] == 15
    assert split_manifest["validation_rollouts"] == 15
    assert np.isfinite(dynamics_metrics["normal_mse"])
    bank = build_expert_latent_bank(
        data_path,
        policy.encoder,
        work_dir / "bank.pth",
        stride=8,
        device=device,
    ).to(device)
    latent_objective = GlobalLatentObjective(dynamics, bank)

    layout = make_layout(1, 910_000, env_config)
    env = PointNavEnv(layout=layout, config=env_config)
    initial_rgb = env.reset(layout.seed, layout)["rgb"]
    history = torch.from_numpy(
        # The reset render is duplicated for observation horizon 2.
        np.stack([initial_rgb, initial_rgb])
    ).permute(0, 3, 1, 2).unsqueeze(0).float().to(device) / 255.0
    noise = torch.randn(
        (1, policy_config.prediction_horizon, 2),
        generator=torch.Generator(device=device).manual_seed(123),
        device=device,
    )
    source = policy.sample(history, noise=noise)
    geometry = GeometryObjective()
    context = geometry_context(
        [layout],
        np.asarray(layout.start),
        source * env_config.max_step,
        device=torch.device(device),
    )
    assert objective_cost_and_gradient_finite(
        geometry, source * env_config.max_step, context
    )
    latest_latent = policy.encoder(history[:, -1])
    assert objective_cost_and_gradient_finite(
        latent_objective,
        source * env_config.max_step,
        {"latent": latest_latent},
    )
    disabled = policy.sample(
        history,
        noise=noise,
        objective=geometry,
        objective_context=context,
        guidance_scale=0.0,
    )
    assert torch.equal(source, disabled)
    geometry_guided = policy.sample(
        history,
        noise=noise,
        objective=geometry,
        objective_context=context,
        guidance_scale=0.01,
    )
    latent_guided = policy.sample(
        history,
        noise=noise,
        objective=latent_objective,
        objective_context={"latent": latest_latent},
        guidance_scale=0.01,
    )
    assert torch.isfinite(geometry_guided).all()
    assert torch.isfinite(latent_guided).all()
    first = evaluate_episode(
        policy,
        make_layout(0, 920_000, env_config),
        method=SOURCE_METHOD,
        policy_seed=19,
        environment_config=env_config,
        device=device,
    )
    second = evaluate_episode(
        policy,
        make_layout(0, 920_000, env_config),
        method=SOURCE_METHOD,
        policy_seed=19,
        environment_config=env_config,
        device=device,
    )
    assert (
        first.success,
        first.collision,
        first.timeout,
        first.steps,
        first.path_length,
        first.minimum_clearance,
    ) == (
        second.success,
        second.collision,
        second.timeout,
        second.steps,
        second.path_length,
        second.minimum_clearance,
    )
    summary = {
        "device": device,
        "demos": len(demos),
        "windows": len(dataset),
        "checkpoints": len(result.checkpoint_paths),
        "dynamics_queries": collection["dynamics_queries"],
        "dynamics_split_rollouts": [
            split_manifest["train_rollouts"],
            split_manifest["validation_rollouts"],
        ],
        "final_loss": result.final_loss,
        "source_rollout_reproducible": True,
        "disabled_guidance_bitwise_parity": True,
        "geometry_gradient_finite": True,
        "guided_ddim_finite": True,
        "latent_gradient_finite": True,
        "source_checkpoint_weights": "ema",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if owned_temp is not None:
        owned_temp.cleanup()


if __name__ == "__main__":
    main()
