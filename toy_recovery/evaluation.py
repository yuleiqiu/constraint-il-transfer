from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np
import torch

from .config import EnvironmentConfig, GuidanceConfig
from .data import EvaluationRecord, write_evaluation_records
from .env import LayoutSpec, PointNavEnv
from .guidance import (
    GeometryObjective,
    GlobalLatentObjective,
    geometry_context,
    obstruction_trigger,
    radial_clip_actions,
)
from .models import SourceDiffusionPolicy
from .rollout import rgb_history_tensor
from .planner import oracle_actions


SOURCE_METHOD = "source"
GEOMETRY_METHOD = "geometry guidance"
LATENT_METHOD = "global latent-manifold guidance (LPB-style objective)"
ORACLE_METHOD = "shortest-path oracle"


def _latent_context(
    policy: SourceDiffusionPolicy,
    latest_rgb: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image = (
        torch.from_numpy(latest_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
        / 255.0
    )
    return {"latent": policy.encoder(image)}


def evaluate_episode(
    policy: SourceDiffusionPolicy,
    layout: LayoutSpec,
    *,
    method: str,
    policy_seed: int,
    guidance_scale: float = 0.0,
    geometry_objective: GeometryObjective | None = None,
    latent_objective: GlobalLatentObjective | None = None,
    environment_config: EnvironmentConfig | None = None,
    guidance_config: GuidanceConfig | None = None,
    device: str | torch.device | None = None,
) -> EvaluationRecord:
    cfg = environment_config or EnvironmentConfig()
    guide_cfg = guidance_config or GuidanceConfig()
    target_device = torch.device(device or next(policy.parameters()).device)
    env = PointNavEnv(layout=layout, config=cfg)
    obs = env.reset(layout.seed, layout)
    history = [obs["rgb"]]
    generator = torch.Generator(device=target_device).manual_seed(policy_seed)
    activations = 0
    inference_seconds = 0.0
    done = False
    replan_index = 0
    policy.eval()
    while not done:
        rgb = rgb_history_tensor(history, target_device)
        noise = torch.randn(
            (1, policy.config.prediction_horizon, 2),
            generator=generator,
            device=target_device,
            dtype=rgb.dtype,
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        start_time = time.perf_counter()
        source_normalized = policy.sample(rgb, noise=noise)
        source_actions = radial_clip_actions(
            source_normalized * cfg.max_step, cfg.max_step
        )
        triggered = obstruction_trigger(
            layout,
            env.state.position,
            source_actions[0].detach().cpu().numpy(),
            agent_radius=cfg.agent_radius,
            margin=guide_cfg.obstruction_margin,
            max_step=cfg.max_step,
        )
        selected = source_normalized
        if method != SOURCE_METHOD and triggered and guidance_scale != 0.0:
            activations += 1
            if method == GEOMETRY_METHOD:
                if geometry_objective is None:
                    raise ValueError("geometry method requires GeometryObjective")
                context = geometry_context(
                    [layout],
                    env.state.position,
                    source_actions.detach(),
                    agent_radius=cfg.agent_radius,
                    margin=guide_cfg.inflated_margin,
                    progress_tolerance=guide_cfg.progress_tolerance,
                    max_step=cfg.max_step,
                    device=target_device,
                )
                objective = geometry_objective
            elif method == LATENT_METHOD:
                if latent_objective is None:
                    raise ValueError("latent method requires GlobalLatentObjective")
                context = _latent_context(policy, history[-1], target_device)
                objective = latent_objective
            else:
                raise ValueError(f"unknown method {method}")
            selected = policy.sample(
                rgb,
                noise=noise,
                objective=objective,
                objective_context=context,
                guidance_scale=guidance_scale,
                gradient_clip=guide_cfg.gradient_clip,
                max_step=cfg.max_step,
            )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        inference_seconds += time.perf_counter() - start_time
        chunk = (
            selected[0, : policy.config.action_horizon].detach().cpu().numpy()
            * cfg.max_step
        )
        for action in chunk:
            obs, _, done, _ = env.step(action)
            history.append(obs["rgb"])
            if done:
                break
        replan_index += 1
    state = env.state
    timeout = not state.collision and not state.success
    return EvaluationRecord(
        method=method,
        difficulty=layout.difficulty,
        layout_seed=layout.seed,
        layout_id=layout.identifier,
        policy_seed=policy_seed,
        success=state.success,
        collision=state.collision,
        timeout=timeout,
        steps=state.step_count,
        path_length=state.path_length,
        minimum_clearance=(
            state.minimum_clearance
            if math.isfinite(state.minimum_clearance)
            else None
        ),
        guidance_activations=activations,
        inference_seconds=inference_seconds,
    )


def evaluate_oracle_episode(
    layout: LayoutSpec,
    *,
    environment_config: EnvironmentConfig | None = None,
) -> EvaluationRecord:
    cfg = environment_config or EnvironmentConfig()
    env = PointNavEnv(layout=layout, config=cfg)
    env.reset(layout.seed, layout)
    done = False
    for action in oracle_actions(layout, cfg):
        _, _, done, _ = env.step(action)
        if done:
            break
    while not done:
        delta = np.asarray(layout.goal) - env.state.position
        _, _, done, _ = env.step(delta)
    state = env.state
    return EvaluationRecord(
        method=ORACLE_METHOD,
        difficulty=layout.difficulty,
        layout_seed=layout.seed,
        layout_id=layout.identifier,
        policy_seed=0,
        success=state.success,
        collision=state.collision,
        timeout=not state.success and not state.collision,
        steps=state.step_count,
        path_length=state.path_length,
        minimum_clearance=(
            state.minimum_clearance
            if math.isfinite(state.minimum_clearance)
            else None
        ),
        guidance_activations=0,
        inference_seconds=0.0,
    )


def evaluate_paired(
    policy: SourceDiffusionPolicy,
    layouts: Sequence[LayoutSpec],
    methods: Sequence[str],
    *,
    policy_seed: int,
    scales: dict[str, float] | None = None,
    geometry_objective: GeometryObjective | None = None,
    latent_objective: GlobalLatentObjective | None = None,
    environment_config: EnvironmentConfig | None = None,
    guidance_config: GuidanceConfig | None = None,
    output_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> list[EvaluationRecord]:
    scale_map = scales or {}
    records = [
        evaluate_episode(
            policy,
            layout,
            method=method,
            policy_seed=policy_seed,
            guidance_scale=scale_map.get(method, 0.0),
            geometry_objective=geometry_objective,
            latent_objective=latent_objective,
            environment_config=environment_config,
            guidance_config=guidance_config,
            device=device,
        )
        for layout in layouts
        for method in methods
    ]
    if output_path is not None:
        write_evaluation_records(output_path, records)
    return records


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def exact_paired_pvalue(failure_to_success: int, success_to_failure: int) -> float:
    discordant = failure_to_success + success_to_failure
    if discordant == 0:
        return 1.0
    extreme = min(failure_to_success, success_to_failure)
    lower_tail = sum(
        math.comb(discordant, k) for k in range(extreme + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def summarize_records(
    records: Sequence[EvaluationRecord],
    *,
    include_runtime: bool = True,
) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.method, record.difficulty)].append(record)
    summaries: dict[str, object] = {}
    for (method, difficulty), group in sorted(grouped.items()):
        total = len(group)
        successes = sum(record.success for record in group)
        collisions = sum(record.collision for record in group)
        timeouts = sum(record.timeout for record in group)
        completed = [record for record in group if record.success]
        method_summary = {
            "episodes": total,
            "success_rate": successes / total,
            "success_wilson_95": wilson_interval(successes, total),
            "collision_rate": collisions / total,
            "collision_wilson_95": wilson_interval(collisions, total),
            "timeout_rate": timeouts / total,
            "timeout_wilson_95": wilson_interval(timeouts, total),
            "mean_path_length": float(
                np.mean([record.path_length for record in group])
            ),
            "mean_completion_steps": (
                float(np.mean([record.steps for record in completed]))
                if completed
                else None
            ),
            "mean_minimum_clearance": (
                float(
                    np.mean(
                        [
                            record.minimum_clearance
                            for record in group
                            if record.minimum_clearance is not None
                        ]
                    )
                )
                if any(record.minimum_clearance is not None for record in group)
                else None
            ),
            "mean_guidance_activations": float(
                np.mean([record.guidance_activations for record in group])
            ),
        }
        if include_runtime:
            method_summary["mean_inference_seconds"] = float(
                np.mean([record.inference_seconds for record in group])
            )
        summaries[f"{method}/D{difficulty}"] = method_summary

    paired: dict[str, object] = {}
    source = {
        (record.difficulty, record.layout_seed, record.policy_seed): record
        for record in records
        if record.method == SOURCE_METHOD
    }
    for method in sorted({record.method for record in records} - {SOURCE_METHOD}):
        for difficulty in sorted({record.difficulty for record in records}):
            guided = {
                (record.difficulty, record.layout_seed, record.policy_seed): record
                for record in records
                if record.method == method and record.difficulty == difficulty
            }
            keys = sorted(set(source) & set(guided))
            keys = [key for key in keys if key[0] == difficulty]
            if not keys:
                continue
            f_to_s = sum(
                not source[key].success and guided[key].success for key in keys
            )
            s_to_f = sum(
                source[key].success and not guided[key].success for key in keys
            )
            paired[f"{method}/D{difficulty}"] = {
                "pairs": len(keys),
                "failure_to_success": f_to_s,
                "success_to_failure": s_to_f,
                "net_transitions": f_to_s - s_to_f,
                "exact_paired_pvalue": exact_paired_pvalue(f_to_s, s_to_f),
            }
    return {"metrics": summaries, "paired": paired}


def rapid_gate(
    records: Sequence[EvaluationRecord],
    recovery_ranking: dict[str, dict[str, object]],
) -> dict[str, object]:
    summary = summarize_records(records)
    source_d0_records = {
        (record.layout_seed, record.policy_seed): record
        for record in records
        if record.method == SOURCE_METHOD and record.difficulty == 0
    }
    d0_ok = bool(source_d0_records)
    for method in (GEOMETRY_METHOD, LATENT_METHOD):
        guided_d0 = {
            (record.layout_seed, record.policy_seed): record
            for record in records
            if record.method == method and record.difficulty == 0
        }
        if set(guided_d0) != set(source_d0_records):
            d0_ok = False
            continue
        for key, source_record in source_d0_records.items():
            guided_record = guided_d0[key]
            d0_ok &= guided_record.guidance_activations == 0
            d0_ok &= (
                guided_record.success,
                guided_record.collision,
                guided_record.timeout,
                guided_record.steps,
                guided_record.path_length,
                guided_record.minimum_clearance,
            ) == (
                source_record.success,
                source_record.collision,
                source_record.timeout,
                source_record.steps,
                source_record.path_length,
                source_record.minimum_clearance,
            )
    transition_ok = any(
        value["net_transitions"] >= 10
        for value in summary["paired"].values()
    )
    ranking_ok = any(
        values.get("auroc") is not None
        and values["auroc"] > 0.6
        and values.get("top_k_precision", 0.0)
        > values.get("random_precision", 1.0)
        for values in recovery_ranking.values()
    )
    passed = bool(d0_ok and transition_ok and ranking_ok)
    return {
        "passed": passed,
        "d0_retention_passed": bool(d0_ok),
        "paired_transition_passed": bool(transition_ok),
        "recovery_ranking_passed": bool(ranking_ok),
        "decision": (
            "advance to three-seed confirmation"
            if passed
            else "stop expansion and archive mechanism boundary; do not implement ReGuide"
        ),
    }


def select_guidance_scales(
    policy: SourceDiffusionPolicy,
    tuning_layouts: Sequence[LayoutSpec],
    *,
    candidate_scales: Sequence[float],
    policy_seed: int,
    geometry_objective: GeometryObjective,
    latent_objective: GlobalLatentObjective,
    environment_config: EnvironmentConfig | None = None,
    guidance_config: GuidanceConfig | None = None,
    device: str | torch.device | None = None,
) -> tuple[dict[str, float], dict[str, object]]:
    """Use the independent tuning split and the same grid for both guides."""

    if not candidate_scales or candidate_scales[0] != 0.0:
        raise ValueError("the shared candidate grid must begin with zero")
    selected: dict[str, float] = {}
    audit: dict[str, object] = {
        "split": "scale_tuning",
        "candidate_scales": list(candidate_scales),
        "methods": {},
    }
    for method in (GEOMETRY_METHOD, LATENT_METHOD):
        candidates = []
        for scale in candidate_scales:
            records = evaluate_paired(
                policy,
                tuning_layouts,
                [SOURCE_METHOD if scale == 0.0 else method],
                policy_seed=policy_seed,
                scales={method: scale},
                geometry_objective=geometry_objective,
                latent_objective=latent_objective,
                environment_config=environment_config,
                guidance_config=guidance_config,
                device=device,
            )
            rates = {
                difficulty: float(
                    np.mean(
                        [
                            record.success
                            for record in records
                            if record.difficulty == difficulty
                        ]
                    )
                )
                for difficulty in range(3)
            }
            # D0 retention is a hard tie-break; target performance is D1+D2.
            score = rates[1] + rates[2] + 0.25 * rates[0]
            candidates.append(
                {"scale": scale, "success_rates": rates, "score": score}
            )
        winner = max(candidates, key=lambda item: (item["score"], -item["scale"]))
        selected[method] = float(winner["scale"])
        audit["methods"][method] = {
            "candidates": candidates,
            "selected_scale": winner["scale"],
        }
    return selected, audit


def source_environment_gate(
    oracle_records: Sequence[EvaluationRecord],
    source_records: Sequence[EvaluationRecord],
) -> dict[str, object]:
    oracle_by_difficulty = {
        difficulty: np.mean(
            [
                record.success
                for record in oracle_records
                if record.difficulty == difficulty
            ]
        )
        for difficulty in range(3)
    }
    source_by_difficulty = {
        difficulty: np.mean(
            [
                record.success
                for record in source_records
                if record.difficulty == difficulty
            ]
        )
        for difficulty in range(3)
    }
    passed = (
        all(value >= 0.99 for value in oracle_by_difficulty.values())
        and source_by_difficulty[0] >= 0.95
        and 0.20 <= source_by_difficulty[1] <= 0.80
        and 0.05 <= source_by_difficulty[2] <= 0.50
        and source_by_difficulty[1] - source_by_difficulty[2] >= 0.10
    )
    return {
        "passed": bool(passed),
        "oracle_success": oracle_by_difficulty,
        "source_success": source_by_difficulty,
        "layout_tuning_allowed": "calibration only; test manifest is immutable",
    }
