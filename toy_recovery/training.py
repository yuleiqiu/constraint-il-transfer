from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TrajectoryWindowDataset
from .models import LatentDynamics, SourceDiffusionPolicy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class SourceTrainingResult:
    checkpoint_paths: tuple[str, ...]
    final_loss: float
    updates: int


def train_source_policy(
    dataset: TrajectoryWindowDataset,
    output_dir: str | Path,
    *,
    seed: int = 0,
    updates: int = 5_000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    ema_decay: float = 0.995,
    resume_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
) -> SourceTrainingResult:
    if updates < 5:
        raise ValueError("at least five updates are required for five checkpoints")
    if dataset.difficulties != {0}:
        raise ValueError(
            "source policy training is strictly limited to D0 demonstrations"
        )
    seed_everything(seed)
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    start_update = 0
    resume_payload: dict[str, object] | None = None
    model = SourceDiffusionPolicy(dataset.policy_config).to(target_device)
    if resume_checkpoint is not None:
        resume_payload = torch.load(
            resume_checkpoint, map_location=target_device, weights_only=False
        )
        if resume_payload.get("schema") != "toy-recovery-source-v2":
            raise ValueError(
                "source resume requires a v2 ResNet18-SpatialSoftmax checkpoint"
            )
        start_update = int(resume_payload["update"])
        if updates <= start_update:
            raise ValueError("target updates must exceed resume checkpoint update")
        model.load_state_dict(
            resume_payload.get("online_model", resume_payload["model"])
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if resume_payload is not None and "optimizer" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])
    ema_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    if resume_payload is not None:
        for name in ema_parameters:
            ema_parameters[name].copy_(resume_payload["model"][name])
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )
    iterator = iter(loader)
    checkpoint_interval = max(1, updates // 5)
    checkpoint_updates = list(
        range(
            ((start_update // checkpoint_interval) + 1) * checkpoint_interval,
            updates + 1,
            checkpoint_interval,
        )
    )
    checkpoint_paths = []
    final_loss = float("nan")
    model.train()
    output = Path(output_dir)
    for update in range(start_update + 1, updates + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        rgb = batch["rgb"].to(target_device)
        actions = batch["action"].to(target_device)
        loss = model.training_loss(rgb, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                ema_parameters[name].lerp_(parameter.detach(), 1.0 - ema_decay)
        final_loss = float(loss.detach().cpu())
        if update in checkpoint_updates:
            path = output / f"source_update_{update:06d}.pth"
            ema_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            ema_state.update(
                {
                    name: tensor.detach().clone()
                    for name, tensor in ema_parameters.items()
                }
            )
            model.save_checkpoint(
                path,
                update=update,
                optimizer=optimizer,
                ema_state=ema_state,
            )
            checkpoint_paths.append(str(path))
            print(
                f"source checkpoint update={update} loss={final_loss:.6f} "
                f"path={path}",
                flush=True,
            )
    return SourceTrainingResult(
        checkpoint_paths=tuple(checkpoint_paths),
        final_loss=final_loss,
        updates=updates,
    )


@dataclass(frozen=True)
class DynamicsBatch:
    latent: torch.Tensor
    action: torch.Tensor
    future_latent: torch.Tensor


def dynamics_validation(
    model: LatentDynamics,
    batches: Iterable[DynamicsBatch],
    *,
    device: str | torch.device = "cpu",
    shuffle_seed: int = 0,
    shuffle_permutations: int = 10,
) -> dict[str, float | bool]:
    target_device = torch.device(device)
    rng = torch.Generator(device=target_device).manual_seed(shuffle_seed)
    cached = list(batches)
    if not cached:
        raise ValueError("dynamics validation requires held-out batches")
    latent = torch.cat([batch.latent for batch in cached]).to(target_device)
    actions = torch.cat([batch.action for batch in cached]).to(target_device)
    target = torch.cat([batch.future_latent for batch in cached]).to(target_device)
    model.eval()
    with torch.no_grad():
        prediction = model(latent, actions)
        normal = float(torch.mean((prediction - target) ** 2).cpu())
        persistence = float(torch.mean((latent - target) ** 2).cpu())
        shuffled_errors = []
        for _ in range(shuffle_permutations):
            permutation = torch.randperm(
                actions.shape[0], generator=rng, device=target_device
            )
            shuffled = model(latent, actions[permutation])
            shuffled_errors.append(
                float(torch.mean((shuffled - target) ** 2).cpu())
            )
    shuffled = float(np.mean(shuffled_errors))
    shuffled_std = float(np.std(shuffled_errors))
    ratio = shuffled / max(normal, 1e-12)
    return {
        "normal_mse": normal,
        "persistence_mse": persistence,
        "action_shuffle_mse": shuffled,
        "action_shuffle_mse_std": shuffled_std,
        "action_shuffle_permutations": shuffle_permutations,
        "action_shuffle_ratio": ratio,
        "beats_persistence": normal < persistence,
        "passed": ratio >= 1.2 and normal < persistence,
    }


def train_latent_dynamics(
    train_batches: Iterable[DynamicsBatch],
    validation_batches: Iterable[DynamicsBatch],
    output_path: str | Path,
    *,
    latent_dim: int = 64,
    seed: int = 0,
    epochs: int = 50,
    learning_rate: float = 3e-4,
    early_stopping_patience: int = 8,
    shuffle_permutations: int = 10,
    device: str | torch.device | None = None,
) -> tuple[LatentDynamics, dict[str, object]]:
    seed_everything(seed)
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cached_train = list(train_batches)
    cached_validation = list(validation_batches)
    if not cached_train or not cached_validation:
        raise ValueError("dynamics training requires non-empty rollout splits")
    action_horizon = int(cached_train[0].action.shape[1])
    if any(batch.action.shape[1] != action_horizon for batch in cached_train):
        raise ValueError("dynamics training batches disagree on action horizon")
    model = LatentDynamics(
        latent_dim=latent_dim,
        prediction_horizon=action_horizon,
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    epochs_completed = 0
    for epoch in range(epochs):
        model.train()
        for batch in cached_train:
            prediction = model(
                batch.latent.to(target_device), batch.action.to(target_device)
            )
            target = batch.future_latent.to(target_device)
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_error = []
            for batch in cached_validation:
                prediction = model(
                    batch.latent.to(target_device),
                    batch.action.to(target_device),
                )
                target = batch.future_latent.to(target_device)
                validation_error.append(
                    torch.mean((prediction - target) ** 2).cpu()
                )
            mean_validation = float(torch.stack(validation_error).mean())
        epochs_completed = epoch + 1
        if mean_validation < best_validation - 1e-8:
            best_validation = mean_validation
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("latent dynamics produced no valid checkpoint")
    model.load_state_dict(best_state)
    metrics = dynamics_validation(
        model,
        cached_validation,
        device=target_device,
        shuffle_seed=seed + 1,
        shuffle_permutations=shuffle_permutations,
    )
    metrics.update(
        {
            "epochs_completed": epochs_completed,
            "best_validation_mse": best_validation,
            "early_stopping_patience": early_stopping_patience,
        }
    )
    model.save_checkpoint(output_path, metadata=metrics)
    return model, metrics
