from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .env import LayoutSpec
from .models import LatentDynamics


def radial_clip_actions(actions: torch.Tensor, max_step: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(actions, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_step / norms, max=1.0)
    return actions * scale


def radial_clip_actions_numpy(actions: np.ndarray, max_step: float) -> np.ndarray:
    values = np.asarray(actions)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_step / np.maximum(norms, 1e-12))
    return values * scale


def geometry_context(
    layouts: Sequence[LayoutSpec],
    positions: np.ndarray | torch.Tensor,
    source_actions: torch.Tensor,
    *,
    agent_radius: float = 0.04,
    margin: float = 0.02,
    progress_tolerance: float = 0.02,
    max_step: float = 0.05,
    device: torch.device | None = None,
) -> dict[str, Any]:
    target_device = device or source_actions.device
    position_tensor = torch.as_tensor(
        positions, dtype=source_actions.dtype, device=target_device
    )
    if position_tensor.ndim == 1:
        position_tensor = position_tensor.unsqueeze(0)
    max_obstacles = max((len(layout.obstacles) for layout in layouts), default=0)
    centers = torch.zeros(
        (len(layouts), max_obstacles, 2),
        dtype=source_actions.dtype,
        device=target_device,
    )
    radii = torch.zeros(
        (len(layouts), max_obstacles),
        dtype=source_actions.dtype,
        device=target_device,
    )
    mask = torch.zeros(
        (len(layouts), max_obstacles),
        dtype=torch.bool,
        device=target_device,
    )
    goals = torch.zeros(
        (len(layouts), 2), dtype=source_actions.dtype, device=target_device
    )
    for batch_index, layout in enumerate(layouts):
        goals[batch_index] = torch.tensor(
            layout.goal, dtype=source_actions.dtype, device=target_device
        )
        for obstacle_index, obstacle in enumerate(layout.obstacles):
            centers[batch_index, obstacle_index] = torch.tensor(
                (obstacle.x, obstacle.y),
                dtype=source_actions.dtype,
                device=target_device,
            )
            radii[batch_index, obstacle_index] = obstacle.radius
            mask[batch_index, obstacle_index] = True
    return {
        "position": position_tensor,
        "goal": goals,
        "obstacle_centers": centers,
        "obstacle_radii": radii,
        "obstacle_mask": mask,
        "source_actions": radial_clip_actions(source_actions.detach(), max_step),
        "agent_radius": agent_radius,
        "margin": margin,
        "progress_tolerance": progress_tolerance,
        "max_step": max_step,
    }


@dataclass(frozen=True)
class GeometryObjective:
    clearance_temperature: float = 0.02
    progress_weight: float = 4.0

    def cost(
        self, actions: torch.Tensor, context: dict[str, Any]
    ) -> torch.Tensor:
        actions = radial_clip_actions(actions, float(context["max_step"]))
        position = context["position"]
        goal = context["goal"]
        centers = context["obstacle_centers"]
        radii = context["obstacle_radii"]
        mask = context["obstacle_mask"]
        source_actions = context["source_actions"]
        if centers.shape[1] == 0:
            clearance_cost = torch.zeros(
                actions.shape[0], device=actions.device, dtype=actions.dtype
            )
        else:
            endpoints = position[:, None, :] + torch.cumsum(actions, dim=1)
            starts = torch.cat((position[:, None, :], endpoints[:, :-1]), dim=1)
            midpoints = 0.5 * (starts + endpoints)
            samples = torch.cat((endpoints, midpoints), dim=1)
            distance = torch.linalg.vector_norm(
                samples[:, :, None, :] - centers[:, None, :, :], dim=-1
            )
            inflated = (
                radii
                + float(context["agent_radius"])
                + float(context["margin"])
            )
            violation = F.softplus(
                (inflated[:, None, :] - distance)
                / self.clearance_temperature
            ) * self.clearance_temperature
            active = mask[:, None, :].to(violation.dtype)
            clearance_cost = (violation.square() * active).sum((1, 2)) / active.sum(
                (1, 2)
            ).clamp_min(1.0)
        candidate_final = position + actions.sum(dim=1)
        source_final = position + source_actions.sum(dim=1)
        candidate_distance = torch.linalg.vector_norm(
            candidate_final - goal, dim=-1
        )
        source_distance = torch.linalg.vector_norm(source_final - goal, dim=-1)
        retreat = F.relu(
            candidate_distance
            - source_distance
            - float(context["progress_tolerance"])
        )
        return clearance_cost + self.progress_weight * retreat.square()


@dataclass(frozen=True)
class GlobalLatentObjective:
    dynamics: LatentDynamics
    expert_bank: torch.Tensor
    max_step: float = 0.05
    softmin_temperature: float = 0.1

    def cost(
        self, actions: torch.Tensor, context: dict[str, Any]
    ) -> torch.Tensor:
        latent = context["latent"]
        actions = radial_clip_actions(actions, self.max_step)
        normalized_actions = actions / self.max_step
        predicted = self.dynamics(latent, normalized_actions)
        bank = self.expert_bank.to(
            device=predicted.device, dtype=predicted.dtype
        )
        squared_distance = torch.mean(
            (predicted[:, None, :] - bank[None, :, :]).square(), dim=-1
        )
        temperature = self.softmin_temperature
        return -temperature * torch.logsumexp(
            -squared_distance / temperature, dim=1
        )


def obstruction_trigger(
    layout: LayoutSpec,
    position: np.ndarray,
    actions: np.ndarray,
    *,
    agent_radius: float = 0.04,
    margin: float = 0.01,
    max_step: float = 0.05,
) -> bool:
    current = np.asarray(position, dtype=np.float64)
    for action in radial_clip_actions_numpy(np.asarray(actions), max_step=max_step):
        end = current + action
        for obstacle in layout.obstacles:
            delta = end - current
            denominator = float(np.dot(delta, delta))
            if denominator == 0.0:
                distance = float(np.linalg.norm(obstacle.center - current))
            else:
                fraction = float(
                    np.dot(obstacle.center - current, delta) / denominator
                )
                projection = current + np.clip(fraction, 0.0, 1.0) * delta
                distance = float(np.linalg.norm(obstacle.center - projection))
            if distance <= obstacle.radius + agent_radius + margin:
                return True
        current = end
    return False


def objective_cost_and_gradient_finite(
    objective: GeometryObjective | GlobalLatentObjective,
    actions: torch.Tensor,
    context: dict[str, Any],
) -> bool:
    candidate = actions.detach().requires_grad_(True)
    cost = objective.cost(candidate, context)
    gradient = torch.autograd.grad(cost.sum(), candidate)[0]
    return bool(torch.isfinite(cost).all() and torch.isfinite(gradient).all())
