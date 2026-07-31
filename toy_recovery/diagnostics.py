from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import EnvironmentConfig, GuidanceConfig
from .env import LayoutSpec, PointNavEnv, PointNavState
from .guidance import (
    GeometryObjective,
    GlobalLatentObjective,
    geometry_context,
    obstruction_trigger,
)
from .models import SourceDiffusionPolicy
from .rollout import rgb_history_tensor


@dataclass
class ObstructionState:
    layout: LayoutSpec
    state: PointNavState
    rgb_history: tuple[np.ndarray, np.ndarray]
    policy_seed: int


def _continue_source(
    policy: SourceDiffusionPolicy,
    env: PointNavEnv,
    history: list[np.ndarray],
    *,
    policy_seed: int,
    device: torch.device,
) -> bool:
    generator = torch.Generator(device=device).manual_seed(policy_seed)
    done = env.state.collision or env.state.success
    while not done and env.state.step_count < env.config.horizon:
        rgb = rgb_history_tensor(history, device)
        chunk = (
            policy.sample(rgb, generator=generator)[
                0, : policy.config.action_horizon
            ]
            .detach()
            .cpu()
            .numpy()
            * env.config.max_step
        )
        for action in chunk:
            obs, _, done, _ = env.step(action)
            history.append(obs["rgb"])
            if done:
                break
    return bool(env.state.success and not env.state.collision)


def capture_failure_obstruction_states(
    policy: SourceDiffusionPolicy,
    layouts: Sequence[LayoutSpec],
    *,
    count: int = 25,
    policy_seed: int = 0,
    config: EnvironmentConfig | None = None,
    guidance_config: GuidanceConfig | None = None,
    device: str | torch.device | None = None,
) -> list[ObstructionState]:
    cfg = config or EnvironmentConfig()
    guide_cfg = guidance_config or GuidanceConfig()
    target_device = torch.device(device or next(policy.parameters()).device)
    selected = []
    for layout in layouts:
        env = PointNavEnv(layout=layout, config=cfg)
        obs = env.reset(layout.seed, layout)
        history = [obs["rgb"]]
        generator = torch.Generator(device=target_device).manual_seed(policy_seed)
        first_obstruction: ObstructionState | None = None
        done = False
        while not done:
            rgb = rgb_history_tensor(history, target_device)
            normalized = policy.sample(rgb, generator=generator)
            actions = normalized[0].detach().cpu().numpy() * cfg.max_step
            if first_obstruction is None and obstruction_trigger(
                layout,
                env.state.position,
                actions,
                agent_radius=cfg.agent_radius,
                margin=guide_cfg.obstruction_margin,
                max_step=cfg.max_step,
            ):
                padded = ([history[0]] + history)[-2:]
                first_obstruction = ObstructionState(
                    layout=layout,
                    state=env.clone_state(),
                    rgb_history=(padded[0].copy(), padded[1].copy()),
                    policy_seed=policy_seed,
                )
            for action in actions[: policy.config.action_horizon]:
                obs, _, done, _ = env.step(action)
                history.append(obs["rgb"])
                if done:
                    break
        if not env.state.success and first_obstruction is not None:
            selected.append(first_obstruction)
        if len(selected) >= count:
            break
    return selected


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    # Lower cost should rank safe-recoverable candidates first.
    wins = 0.0
    for p_value in positive:
        wins += np.sum(p_value < negative)
        wins += 0.5 * np.sum(p_value == negative)
    return float(wins / (len(positive) * len(negative)))


def fixed_state_recovery_diagnostic(
    policy: SourceDiffusionPolicy,
    states: Sequence[ObstructionState],
    geometry_objective: GeometryObjective,
    latent_objective: GlobalLatentObjective,
    *,
    candidates_per_state: int = 32,
    guided_candidates_per_state: int = 4,
    continuation_seeds: int = 8,
    required_successes: int = 6,
    top_k: int = 5,
    guidance_scales: dict[str, float] | None = None,
    config: EnvironmentConfig | None = None,
    guidance_config: GuidanceConfig | None = None,
    device: str | torch.device | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    cfg = config or EnvironmentConfig()
    guide_cfg = guidance_config or GuidanceConfig()
    target_device = torch.device(device or next(policy.parameters()).device)
    scale_map = guidance_scales or {"geometry": 0.0, "global_latent": 0.0}
    rows = []

    def evaluate_actions(
        *,
        captured: ObstructionState,
        state_index: int,
        candidate_index: int,
        noise_seed: int,
        method: str,
        actions: torch.Tensor,
        geometry_ctx: dict[str, object],
        latent_ctx: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        geometry_cost = float(
            geometry_objective.cost(actions, geometry_ctx).detach().cpu()[0]
        )
        latent_cost = float(
            latent_objective.cost(actions, latent_ctx).detach().cpu()[0]
        )
        prefix_env = PointNavEnv(layout=captured.layout, config=cfg)
        prefix_env.restore_state(captured.state)
        prefix_history = [
            captured.rgb_history[0].copy(),
            captured.rgb_history[1].copy(),
        ]
        prefix_safe = True
        for action in actions[
            0, : policy.config.action_horizon
        ].detach().cpu().numpy():
            obs, _, done, _ = prefix_env.step(action)
            prefix_history.append(obs["rgb"])
            if prefix_env.state.collision:
                prefix_safe = False
                break
            if done:
                break
        continuation_successes = 0
        if prefix_safe:
            endpoint = prefix_env.clone_state()
            endpoint_history = [image.copy() for image in prefix_history[-2:]]
            for continuation_index in range(continuation_seeds):
                continuation_env = PointNavEnv(
                    layout=captured.layout, config=cfg
                )
                continuation_env.restore_state(endpoint)
                success = _continue_source(
                    policy,
                    continuation_env,
                    [image.copy() for image in endpoint_history],
                    policy_seed=(
                        captured.policy_seed * 100_000
                        + state_index * continuation_seeds
                        + continuation_index
                    ),
                    device=target_device,
                )
                continuation_successes += int(success)
        return {
            "state_index": state_index,
            "difficulty": captured.layout.difficulty,
            "layout_seed": captured.layout.seed,
            "candidate_index": candidate_index,
            "noise_seed": noise_seed,
            "method": method,
            "geometry_cost": geometry_cost,
            "latent_cost": latent_cost,
            "prefix_safe": prefix_safe,
            "continuation_successes": continuation_successes,
            "continuation_trials": continuation_seeds,
            "safe_recoverable": (
                prefix_safe and continuation_successes >= required_successes
            ),
        }

    for state_index, captured in enumerate(states):
        rgb = rgb_history_tensor(list(captured.rgb_history), target_device)
        latest = (
            torch.from_numpy(captured.rgb_history[-1])
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(target_device)
            / 255.0
        )
        latent_ctx = {"latent": policy.encoder(latest)}
        for candidate_index in range(candidates_per_state):
            noise_seed = (
                captured.policy_seed * 1_000_000
                + state_index * candidates_per_state
                + candidate_index
            )
            noise = torch.randn(
                (1, policy.config.prediction_horizon, 2),
                generator=torch.Generator(device=target_device).manual_seed(
                    noise_seed
                ),
                device=target_device,
            )
            source_normalized = policy.sample(rgb, noise=noise)
            source_actions = source_normalized * cfg.max_step
            geometry_ctx = geometry_context(
                [captured.layout],
                captured.state.position,
                source_actions,
                agent_radius=cfg.agent_radius,
                margin=guide_cfg.inflated_margin,
                progress_tolerance=guide_cfg.progress_tolerance,
                max_step=cfg.max_step,
                device=target_device,
            )
            method_actions = [("source", source_actions)]
            if candidate_index < guided_candidates_per_state:
                geometry_actions = policy.sample(
                    rgb,
                    noise=noise,
                    objective=geometry_objective,
                    objective_context=geometry_ctx,
                    guidance_scale=float(scale_map["geometry"]),
                    gradient_clip=guide_cfg.gradient_clip,
                    max_step=cfg.max_step,
                ) * cfg.max_step
                latent_actions = policy.sample(
                    rgb,
                    noise=noise,
                    objective=latent_objective,
                    objective_context=latent_ctx,
                    guidance_scale=float(scale_map["global_latent"]),
                    gradient_clip=guide_cfg.gradient_clip,
                    max_step=cfg.max_step,
                ) * cfg.max_step
                method_actions.extend(
                    (
                        ("geometry", geometry_actions),
                        ("global_latent", latent_actions),
                    )
                )
            for method, actions in method_actions:
                rows.append(
                    evaluate_actions(
                        captured=captured,
                        state_index=state_index,
                        candidate_index=candidate_index,
                        noise_seed=noise_seed,
                        method=method,
                        actions=actions,
                        geometry_ctx=geometry_ctx,
                        latent_ctx=latent_ctx,
                    )
                )
    ranking = {}
    source_rows = [row for row in rows if row["method"] == "source"]
    labels = np.asarray(
        [row["safe_recoverable"] for row in source_rows], dtype=bool
    )
    random_precision = float(labels.mean()) if len(labels) else float("nan")
    for name, key in (
        ("geometry", "geometry_cost"),
        ("global_latent", "latent_cost"),
    ):
        scores = np.asarray([row[key] for row in source_rows], dtype=np.float64)
        precisions = []
        state_aurocs = []
        covered = 0
        any_covered = 0
        for state_index in sorted({row["state_index"] for row in source_rows}):
            indices = np.asarray(
                [
                    index
                    for index, row in enumerate(source_rows)
                    if row["state_index"] == state_index
                ]
            )
            order = indices[np.argsort(scores[indices], kind="stable")]
            precisions.append(float(labels[order[:top_k]].mean()))
            covered += int(labels[order[:top_k]].any())
            any_covered += int(labels[indices].any())
            state_auc = _auroc(scores[indices], labels[indices])
            if np.isfinite(state_auc):
                state_aurocs.append(state_auc)
        ranking[name] = {
            "auroc": (
                float(np.mean(state_aurocs)) if state_aurocs else None
            ),
            "auroc_states": len(state_aurocs),
            "pooled_auroc_diagnostic_only": (
                float(_auroc(scores, labels))
                if np.isfinite(_auroc(scores, labels))
                else None
            ),
            "top_k": top_k,
            "top_k_precision": float(np.mean(precisions)) if precisions else 0.0,
            "random_precision": random_precision,
            "selected_state_coverage": covered / max(len(states), 1),
            "candidate_set_state_coverage": any_covered / max(len(states), 1),
        }
    achieved_coverage = {}
    for method in ("source", "geometry", "global_latent"):
        method_rows = [row for row in rows if row["method"] == method]
        method_labels = np.asarray(
            [row["safe_recoverable"] for row in method_rows], dtype=bool
        )
        state_coverage = 0
        for state_index in sorted({row["state_index"] for row in method_rows}):
            state_coverage += int(
                any(
                    row["safe_recoverable"]
                    for row in method_rows
                    if row["state_index"] == state_index
                )
            )
        achieved_coverage[method] = {
            "candidate_recoverable_rate": float(method_labels.mean()),
            "state_coverage": state_coverage / max(len(states), 1),
        }
    result = {
        "schema": "toy-recovery-fixed-state-v1",
        "definition": (
            f"prefix collision-free and at least {required_successes}/"
            f"{continuation_seeds} frozen-source continuations safely complete"
        ),
        "states": len(states),
        "candidates_per_state": candidates_per_state,
        "guided_candidates_per_state": guided_candidates_per_state,
        "guidance_scales": scale_map,
        "ranking": ranking,
        "achieved_coverage": achieved_coverage,
        "rows": rows,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
