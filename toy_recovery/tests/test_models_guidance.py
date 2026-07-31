from __future__ import annotations

import numpy as np
import torch

from toy_recovery.config import PolicyConfig
from toy_recovery.env import make_layout
from toy_recovery.guidance import (
    GeometryObjective,
    GlobalLatentObjective,
    geometry_context,
    objective_cost_and_gradient_finite,
)
from toy_recovery.models import LatentDynamics, SourceDiffusionPolicy


def small_policy() -> SourceDiffusionPolicy:
    return SourceDiffusionPolicy(
        PolicyConfig(latent_dim=16, unet_down_dims=(16, 32))
    ).eval()


def test_disabled_guidance_has_bitwise_source_parity() -> None:
    torch.manual_seed(0)
    policy = small_policy()
    history = torch.rand(1, 2, 3, 64, 64)
    noise = torch.randn(1, 16, 2)
    source = policy.sample(history, noise=noise)
    layout = make_layout(1, 4)
    context = geometry_context(
        [layout], np.asarray(layout.start), source * 0.05
    )
    disabled = policy.sample(
        history,
        noise=noise,
        objective=GeometryObjective(),
        objective_context=context,
        guidance_scale=0.0,
    )
    assert torch.equal(source, disabled)


def test_geometry_and_latent_cost_gradients_are_finite() -> None:
    torch.manual_seed(1)
    actions = torch.randn(2, 16, 2) * 0.01
    layouts = [make_layout(1, 10), make_layout(2, 11)]
    context = geometry_context(
        layouts,
        np.asarray([layout.start for layout in layouts]),
        actions,
    )
    assert objective_cost_and_gradient_finite(
        GeometryObjective(), actions, context
    )
    dynamics = LatentDynamics(latent_dim=8, hidden_dim=16)
    latent = torch.randn(2, 8)
    bank = torch.randn(12, 8)
    objective = GlobalLatentObjective(dynamics, bank)
    assert objective_cost_and_gradient_finite(
        objective, actions, {"latent": latent}
    )
