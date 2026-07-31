#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toy_recovery.config import ExperimentConfig
from toy_recovery.diagnostics import (
    capture_failure_obstruction_states,
    fixed_state_recovery_diagnostic,
)
from toy_recovery.env import LayoutSpec, load_layout_manifest
from toy_recovery.evaluation import GEOMETRY_METHOD, LATENT_METHOD
from toy_recovery.guidance import GeometryObjective, GlobalLatentObjective
from toy_recovery.models import LatentDynamics, SourceDiffusionPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-bank", type=Path, required=True)
    parser.add_argument("--scale-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = SourceDiffusionPolicy.load_checkpoint(
        args.source_checkpoint, device=device
    )
    dynamics, metadata = LatentDynamics.load_checkpoint(
        args.dynamics_checkpoint, device=device
    )
    if not metadata.get("passed", False):
        raise SystemExit("action-shuffle gate failed; diagnostic is blocked")
    bank = torch.load(args.latent_bank, map_location=device, weights_only=False)
    selected_scales = json.loads(args.scale_selection.read_text())["selected_scales"]
    diagnostic_scales = {
        "geometry": float(selected_scales[GEOMETRY_METHOD]),
        "global_latent": float(selected_scales[LATENT_METHOD]),
    }
    manifest = load_layout_manifest(args.manifest)
    screen_layouts = [
        LayoutSpec.from_dict(raw)
        for raw in manifest["splits"]["screen"]
    ]
    states = []
    for difficulty in (1, 2):
        layouts = [
            layout for layout in screen_layouts if layout.difficulty == difficulty
        ]
        captured = capture_failure_obstruction_states(
            policy,
            layouts,
            count=config.recovery_states_per_difficulty,
            policy_seed=args.policy_seed,
            config=config.environment,
            guidance_config=config.guidance,
            device=device,
        )
        if len(captured) != config.recovery_states_per_difficulty:
            raise RuntimeError(
                f"only found {len(captured)}/"
                f"{config.recovery_states_per_difficulty} D{difficulty} failure states"
            )
        states.extend(captured)
    fixed_state_recovery_diagnostic(
        policy,
        states,
        GeometryObjective(),
        GlobalLatentObjective(
            dynamics,
            bank["latents"].to(device),
            max_step=config.environment.max_step,
            softmin_temperature=config.guidance.softmin_temperature,
        ),
        candidates_per_state=config.recovery_candidates_per_state,
        guided_candidates_per_state=(
            config.recovery_guided_candidates_per_state
        ),
        continuation_seeds=config.recovery_continuation_seeds,
        required_successes=config.recovery_required_successes,
        guidance_scales=diagnostic_scales,
        config=config.environment,
        guidance_config=config.guidance,
        device=device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
