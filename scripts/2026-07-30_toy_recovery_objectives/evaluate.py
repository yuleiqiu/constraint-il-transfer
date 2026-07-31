#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toy_recovery.config import ExperimentConfig
from toy_recovery.env import LayoutSpec, load_layout_manifest
from toy_recovery.evaluation import (
    GEOMETRY_METHOD,
    LATENT_METHOD,
    SOURCE_METHOD,
    evaluate_paired,
    select_guidance_scales,
    summarize_records,
)
from toy_recovery.guidance import GeometryObjective, GlobalLatentObjective
from toy_recovery.models import LatentDynamics, SourceDiffusionPolicy


def layouts_from_manifest(path: Path, split: str) -> list[LayoutSpec]:
    manifest = load_layout_manifest(path)
    return [
        LayoutSpec.from_dict(raw) for raw in manifest["splits"][split]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("scale_tuning", "screen", "test"), required=True
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--tune-scales", action="store_true")
    parser.add_argument("--scale-selection", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.tune_scales and args.split != "scale_tuning":
        raise SystemExit("guidance scales may only be tuned on scale_tuning layouts")
    if not args.tune_scales and args.split == "scale_tuning":
        raise SystemExit("scale_tuning split is only for selecting scales")
    config = ExperimentConfig.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = SourceDiffusionPolicy.load_checkpoint(
        args.source_checkpoint, device=device
    )
    dynamics, metadata = LatentDynamics.load_checkpoint(
        args.dynamics_checkpoint, device=device
    )
    if not metadata.get("passed", False):
        raise SystemExit(
            "latent-dynamics action-shuffle gate did not pass; evaluation blocked"
        )
    bank_payload = torch.load(args.latent_bank, map_location=device, weights_only=False)
    if bank_payload.get("schema") != "toy-recovery-d0-latent-bank-v1":
        raise ValueError("unsupported latent-bank schema")
    geometry = GeometryObjective()
    latent = GlobalLatentObjective(
        dynamics=dynamics,
        expert_bank=bank_payload["latents"].to(device),
        max_step=config.environment.max_step,
        softmin_temperature=config.guidance.softmin_temperature,
    )
    layouts = layouts_from_manifest(args.manifest, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.tune_scales:
        scales, audit = select_guidance_scales(
            policy,
            layouts,
            candidate_scales=config.guidance.scales,
            policy_seed=args.policy_seed,
            geometry_objective=geometry,
            latent_objective=latent,
            environment_config=config.environment,
            guidance_config=config.guidance,
            device=device,
        )
        selection_path = args.output_dir / "scale_selection.json"
        selection_path.write_text(
            json.dumps(
                {"selected_scales": scales, "audit": audit},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        if args.scale_selection is None:
            raise SystemExit("--scale-selection is required when not tuning")
        selected = json.loads(args.scale_selection.read_text())
        scales = selected["selected_scales"]
    records = evaluate_paired(
        policy,
        layouts,
        [SOURCE_METHOD, GEOMETRY_METHOD, LATENT_METHOD],
        policy_seed=args.policy_seed,
        scales=scales,
        geometry_objective=geometry,
        latent_objective=latent,
        environment_config=config.environment,
        guidance_config=config.guidance,
        output_path=args.output_dir / f"{args.split}_episodes.jsonl",
        device=device,
    )
    summary = summarize_records(records)
    (args.output_dir / f"{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
