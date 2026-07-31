#!/usr/bin/env python3
"""Run the geometry-only tuning and paired-screen diagnostic.

This intentionally does not load latent dynamics. It is an explicitly
requested diagnostic after the latent-dynamics funnel stopped, and therefore
does not reopen the latent or confirmation stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toy_recovery.config import ExperimentConfig
from toy_recovery.data import write_evaluation_records
from toy_recovery.env import LayoutSpec, load_layout_manifest
from toy_recovery.evaluation import (
    GEOMETRY_METHOD,
    SOURCE_METHOD,
    evaluate_episode,
    summarize_records,
)
from toy_recovery.guidance import GeometryObjective
from toy_recovery.models import SourceDiffusionPolicy


def _layouts(path: Path, split: str) -> list[LayoutSpec]:
    manifest = load_layout_manifest(path)
    return [LayoutSpec.from_dict(raw) for raw in manifest["splits"][split]]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _evaluate(
    policy: SourceDiffusionPolicy,
    layouts: list[LayoutSpec],
    *,
    method: str,
    scale: float,
    policy_seed: int,
    config: ExperimentConfig,
    device: str,
):
    objective = GeometryObjective()
    records = []
    for index, layout in enumerate(layouts, start=1):
        records.append(
            evaluate_episode(
                policy,
                layout,
                method=method,
                policy_seed=policy_seed,
                guidance_scale=scale,
                geometry_objective=objective,
                environment_config=config.environment,
                guidance_config=config.guidance,
                device=device,
            )
        )
        if index % 10 == 0 or index == len(layouts):
            successes = sum(record.success for record in records)
            print(
                f"{method} scale={scale:g}: {index}/{len(layouts)} "
                f"episodes, {successes} successes",
                flush=True,
            )
    return records


def tune(
    policy: SourceDiffusionPolicy,
    layouts: list[LayoutSpec],
    *,
    output_dir: Path,
    policy_seed: int,
    config: ExperimentConfig,
    device: str,
) -> None:
    candidates = []
    for scale in config.guidance.scales:
        method = SOURCE_METHOD if scale == 0.0 else GEOMETRY_METHOD
        records = _evaluate(
            policy,
            layouts,
            method=method,
            scale=scale,
            policy_seed=policy_seed,
            config=config,
            device=device,
        )
        write_evaluation_records(
            output_dir / f"scale_{scale:g}_episodes.jsonl", records
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
        score = rates[1] + rates[2] + 0.25 * rates[0]
        candidates.append(
            {"scale": scale, "success_rates": rates, "score": score}
        )
        _write_json(output_dir / "tuning_progress.json", {"candidates": candidates})
    winner = max(candidates, key=lambda item: (item["score"], -item["scale"]))
    _write_json(
        output_dir / "scale_selection.json",
        {
            "schema": "toy-recovery-geometry-only-scale-selection-v1",
            "split": "scale_tuning",
            "policy_seed": policy_seed,
            "candidate_scales": list(config.guidance.scales),
            "candidates": candidates,
            "selected_scale": winner["scale"],
            "selection_rule": "maximize D1 SR + D2 SR + 0.25 * D0 SR; prefer smaller scale on ties",
            "scope": "geometry-only diagnostic after latent-dynamics gate failure",
        },
    )
    print(f"selected geometry scale: {winner['scale']:g}", flush=True)


def screen(
    policy: SourceDiffusionPolicy,
    layouts: list[LayoutSpec],
    *,
    output_dir: Path,
    scale_selection: Path,
    policy_seed: int,
    config: ExperimentConfig,
    device: str,
) -> None:
    selection = json.loads(scale_selection.read_text())
    scale = float(selection["selected_scale"])
    source = _evaluate(
        policy,
        layouts,
        method=SOURCE_METHOD,
        scale=0.0,
        policy_seed=policy_seed,
        config=config,
        device=device,
    )
    geometry = _evaluate(
        policy,
        layouts,
        method=GEOMETRY_METHOD,
        scale=scale,
        policy_seed=policy_seed,
        config=config,
        device=device,
    )
    records = [record for pair in zip(source, geometry) for record in pair]
    write_evaluation_records(output_dir / "screen_episodes.jsonl", records)
    summary = summarize_records(records)
    summary["selected_scale"] = scale
    summary["split"] = "screen"
    summary["scope"] = "geometry-only diagnostic after latent-dynamics gate failure"
    _write_json(output_dir / "screen_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("tune", "screen"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-selection", type=Path)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.phase == "screen" and args.scale_selection is None:
        parser.error("--scale-selection is required for --phase screen")
    config = ExperimentConfig.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = SourceDiffusionPolicy.load_checkpoint(
        args.source_checkpoint, device=device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = "scale_tuning" if args.phase == "tune" else "screen"
    layouts = _layouts(args.manifest, split)
    if args.phase == "tune":
        tune(
            policy,
            layouts,
            output_dir=args.output_dir,
            policy_seed=args.policy_seed,
            config=config,
            device=device,
        )
    else:
        screen(
            policy,
            layouts,
            output_dir=args.output_dir,
            scale_selection=args.scale_selection,
            policy_seed=args.policy_seed,
            config=config,
            device=device,
        )


if __name__ == "__main__":
    main()
