from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvironmentConfig:
    workspace_low: float = -1.0
    workspace_high: float = 1.0
    agent_radius: float = 0.04
    goal_radius: float = 0.08
    max_step: float = 0.05
    horizon: int = 80
    planner_margin: float = 0.02
    image_size: int = 64


@dataclass(frozen=True)
class PolicyConfig:
    observation_horizon: int = 2
    prediction_horizon: int = 16
    action_horizon: int = 8
    train_timesteps: int = 100
    inference_steps: int = 10
    latent_dim: int = 64
    unet_down_dims: tuple[int, ...] = (64, 128, 256)


@dataclass(frozen=True)
class GuidanceConfig:
    scales: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1)
    gradient_clip: float = 1.0
    inflated_margin: float = 0.02
    progress_tolerance: float = 0.02
    obstruction_margin: float = 0.01
    softmin_temperature: float = 0.1


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    d0_demonstrations: int = 200
    source_updates: int = 5_000
    source_update_cap: int = 10_000
    dynamics_rollouts: int = 150
    dynamics_max_epochs: int = 50
    dynamics_shuffle_permutations: int = 10
    calibration_layouts_per_difficulty: int = 50
    scale_tuning_layouts_per_difficulty: int = 30
    screen_layouts_per_difficulty: int = 100
    test_layouts_per_difficulty: int = 200
    recovery_states_per_difficulty: int = 10
    recovery_candidates_per_state: int = 16
    recovery_guided_candidates_per_state: int = 4
    recovery_continuation_seeds: int = 8
    recovery_required_successes: int = 6
    policy_seeds: tuple[int, ...] = (0,)
    dynamics_seeds: tuple[int, ...] = (0,)
    confirmation_policy_seeds: tuple[int, ...] = (10, 11, 12)
    confirmation_dynamics_seeds: tuple[int, ...] = (20, 21, 22)
    layout_seed_base: int = 10_000
    output_dir: str = "outputs/toy_recovery_objectives"

    def validate(self) -> None:
        p = self.policy
        if (p.observation_horizon, p.prediction_horizon, p.action_horizon) != (
            2,
            16,
            8,
        ):
            raise ValueError("the rapid prototype fixes horizons at 2/16/8")
        if p.train_timesteps != 100 or p.inference_steps != 10:
            raise ValueError("the rapid prototype fixes DDIM at 100/10")
        if self.environment.horizon < p.prediction_horizon:
            raise ValueError("episode horizon must cover a prediction chunk")
        if self.dynamics_rollouts % 15 != 0:
            raise ValueError(
                "dynamics rollouts must balance 3 difficulties x 5 checkpoints"
            )
        if self.source_updates < 5 or self.source_updates % 5 != 0:
            raise ValueError("source updates must support five equal checkpoints")
        if self.source_update_cap < self.source_updates:
            raise ValueError("source update cap cannot precede the initial budget")
        if not (
            0 < self.recovery_required_successes
            <= self.recovery_continuation_seeds
        ):
            raise ValueError("invalid recoverability continuation threshold")
        if not (
            0 < self.recovery_guided_candidates_per_state
            <= self.recovery_candidates_per_state
        ):
            raise ValueError("guided coverage candidates must fit the ranking pool")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        env = EnvironmentConfig(**raw.pop("environment"))
        policy_raw = raw.pop("policy")
        policy_raw["unet_down_dims"] = tuple(policy_raw["unet_down_dims"])
        policy = PolicyConfig(**policy_raw)
        guidance_raw = raw.pop("guidance")
        guidance_raw["scales"] = tuple(guidance_raw["scales"])
        guidance = GuidanceConfig(**guidance_raw)
        for key in (
            "policy_seeds",
            "dynamics_seeds",
            "confirmation_policy_seeds",
            "confirmation_dynamics_seeds",
        ):
            if key in raw:
                raw[key] = tuple(raw[key])
        config = cls(
            environment=env,
            policy=policy,
            guidance=guidance,
            **raw,
        )
        config.validate()
        return config
