#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toy_recovery.config import ExperimentConfig
from toy_recovery.models import SourceDiffusionPolicy
from toy_recovery.rollout import build_expert_latent_bank, load_dynamics_batches
from toy_recovery.training import train_latent_dynamics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--d0-demos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = SourceDiffusionPolicy.load_checkpoint(
        args.source_checkpoint, device=device
    )
    for parameter in policy.encoder.parameters():
        parameter.requires_grad_(False)
    train_batches, validation_batches, split_manifest = load_dynamics_batches(
        args.rollouts,
        policy.encoder,
        device=device,
        split_seed=args.seed,
    )
    model, metrics = train_latent_dynamics(
        train_batches,
        validation_batches,
        args.output_dir / "latent_dynamics.pth",
        latent_dim=policy.config.latent_dim,
        seed=args.seed,
        epochs=args.epochs or config.dynamics_max_epochs,
        shuffle_permutations=config.dynamics_shuffle_permutations,
        device=device,
    )
    metrics["data_split"] = split_manifest
    model.save_checkpoint(
        args.output_dir / "latent_dynamics.pth", metadata=metrics
    )
    (args.output_dir / "dynamics_validation.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    if not metrics["passed"]:
        raise SystemExit(
            "dynamics failed action-shuffle or persistence gate; guidance is blocked"
        )
    build_expert_latent_bank(
        args.d0_demos,
        policy.encoder,
        args.output_dir / "d0_expert_latent_bank.pth",
        device=device,
    )


if __name__ == "__main__":
    main()
