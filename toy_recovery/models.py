from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch
from torch import nn
import torch.nn.functional as F

from robomimic.models.diffusion_policy_nets import ConditionalUnet1D

from .config import PolicyConfig


class GuidanceObjective(Protocol):
    def cost(
        self, actions: torch.Tensor, context: dict[str, Any]
    ) -> torch.Tensor:
        """Return one lower-is-better scalar per batch element."""


class ImageEncoder(nn.Module):
    """Small GroupNorm CNN used by both the source policy and latent study."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.Mish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.Mish(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, latent_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(image).flatten(1))


class SourceDiffusionPolicy(nn.Module):
    def __init__(self, config: PolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyConfig()
        self.encoder = ImageEncoder(self.config.latent_dim)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=self.config.latent_dim
            * self.config.observation_horizon,
            diffusion_step_embed_dim=64,
            down_dims=list(self.config.unet_down_dims),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

    @staticmethod
    def scheduler(train_timesteps: int = 100) -> DDIMScheduler:
        return DDIMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

    def encode_history(self, rgb_history: torch.Tensor) -> torch.Tensor:
        batch, horizon, channels, height, width = rgb_history.shape
        encoded = self.encoder(
            rgb_history.reshape(batch * horizon, channels, height, width)
        )
        return encoded.reshape(batch, horizon * self.config.latent_dim)

    def training_loss(
        self,
        rgb_history: torch.Tensor,
        normalized_actions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        scheduler = self.scheduler(self.config.train_timesteps)
        conditioning = self.encode_history(rgb_history)
        noise = torch.randn(
            normalized_actions.shape,
            device=normalized_actions.device,
            dtype=normalized_actions.dtype,
            generator=generator,
        )
        timesteps = torch.randint(
            0,
            self.config.train_timesteps,
            (normalized_actions.shape[0],),
            device=normalized_actions.device,
            generator=generator,
        ).long()
        noisy = scheduler.add_noise(normalized_actions, noise, timesteps)
        prediction = self.noise_pred_net(noisy, timesteps, global_cond=conditioning)
        return F.mse_loss(prediction, noise)

    def _predicted_clean(
        self,
        sample: torch.Tensor,
        noise_prediction: torch.Tensor,
        timestep: int | torch.Tensor,
        scheduler: DDIMScheduler,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        alpha = scheduler.alphas_cumprod[index].to(
            device=sample.device, dtype=sample.dtype
        )
        sqrt_alpha = alpha.sqrt()
        sqrt_one_minus_alpha = (1.0 - alpha).sqrt()
        clean = (sample - sqrt_one_minus_alpha * noise_prediction) / sqrt_alpha
        return clean, sqrt_alpha, sqrt_one_minus_alpha

    @torch.no_grad()
    def sample(
        self,
        rgb_history: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        objective: GuidanceObjective | None = None,
        objective_context: dict[str, Any] | None = None,
        guidance_scale: float = 0.0,
        gradient_clip: float = 1.0,
        max_step: float = 0.05,
    ) -> torch.Tensor:
        """Shared predicted-clean DDIM sampler for source and both guides."""

        batch = rgb_history.shape[0]
        device, dtype = rgb_history.device, rgb_history.dtype
        conditioning = self.encode_history(rgb_history)
        sample = (
            noise.clone()
            if noise is not None
            else torch.randn(
                (batch, self.config.prediction_horizon, 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
        scheduler = self.scheduler(self.config.train_timesteps)
        scheduler.set_timesteps(self.config.inference_steps, device=device)
        context = objective_context or {}
        for timestep in scheduler.timesteps:
            noise_prediction = self.noise_pred_net(
                sample, timestep, global_cond=conditioning
            )
            if objective is not None and guidance_scale != 0.0:
                clean, sqrt_alpha, sqrt_one_minus_alpha = self._predicted_clean(
                    sample, noise_prediction, timestep, scheduler
                )
                with torch.enable_grad():
                    clean_for_cost = clean.detach().requires_grad_(True)
                    costs = objective.cost(clean_for_cost * max_step, context)
                    if costs.shape != (batch,):
                        raise ValueError(
                            "guidance objective must return shape (batch,)"
                        )
                    gradient = torch.autograd.grad(
                        costs.sum(), clean_for_cost, only_inputs=True
                    )[0]
                flat = gradient.flatten(1)
                norms = flat.norm(dim=1).clamp_min(1e-12)
                scales = torch.clamp(gradient_clip / norms, max=1.0)
                clipped = gradient * scales[:, None, None]
                guided_clean = clean - guidance_scale * clipped
                if float(sqrt_one_minus_alpha) > 0.0:
                    noise_prediction = (
                        sample - sqrt_alpha * guided_clean
                    ) / sqrt_one_minus_alpha
            sample = scheduler.step(
                noise_prediction,
                timestep,
                sample,
                generator=generator,
            ).prev_sample
        return sample

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        update: int,
        optimizer: torch.optim.Optimizer | None = None,
        ema_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "schema": "toy-recovery-source-v1",
            "config": asdict(self.config),
            "update": update,
            "model": ema_state or self.state_dict(),
            "weights": "ema" if ema_state is not None else "online",
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if ema_state is not None:
            payload["online_model"] = self.state_dict()
        torch.save(payload, target)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple["SourceDiffusionPolicy", dict[str, object]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("schema") != "toy-recovery-source-v1":
            raise ValueError("unsupported source checkpoint")
        config_raw = dict(payload["config"])
        config_raw["unet_down_dims"] = tuple(config_raw["unet_down_dims"])
        model = cls(PolicyConfig(**config_raw)).to(device)
        model.load_state_dict(payload["model"])
        return model, payload


class LatentDynamics(nn.Module):
    """Action-conditioned 16-step latent predictor."""

    def __init__(
        self,
        latent_dim: int = 64,
        prediction_horizon: int = 16,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.prediction_horizon = prediction_horizon
        self.network = nn.Sequential(
            nn.Linear(latent_dim + prediction_horizon * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, latent: torch.Tensor, normalized_actions: torch.Tensor
    ) -> torch.Tensor:
        if normalized_actions.shape[1:] != (self.prediction_horizon, 2):
            raise ValueError("latent dynamics requires a full 16-action chunk")
        inputs = torch.cat((latent, normalized_actions.flatten(1)), dim=1)
        return self.network(inputs)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, object]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "toy-recovery-latent-dynamics-v1",
                "latent_dim": self.latent_dim,
                "prediction_horizon": self.prediction_horizon,
                "model": self.state_dict(),
                "metadata": metadata,
            },
            target,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> tuple["LatentDynamics", dict[str, object]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("schema") != "toy-recovery-latent-dynamics-v1":
            raise ValueError("unsupported latent dynamics checkpoint")
        model = cls(
            latent_dim=int(payload["latent_dim"]),
            prediction_horizon=int(payload["prediction_horizon"]),
        ).to(device)
        model.load_state_dict(payload["model"])
        return model, dict(payload["metadata"])
