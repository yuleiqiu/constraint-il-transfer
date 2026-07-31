#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toy_recovery.config import ExperimentConfig
from toy_recovery.env import LayoutSpec, load_layout_manifest
from toy_recovery.evaluation import (
    SOURCE_METHOD,
    evaluate_episode,
    evaluate_oracle_episode,
    source_environment_gate,
)
from toy_recovery.models import SourceDiffusionPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = SourceDiffusionPolicy.load_checkpoint(
        args.source_checkpoint, device=device
    )
    manifest = load_layout_manifest(args.candidate_manifest)
    calibration = [
        LayoutSpec.from_dict(raw)
        for raw in manifest["splits"]["calibration"]
    ]
    oracle_records = [
        evaluate_oracle_episode(layout, environment_config=config.environment)
        for layout in calibration
    ]
    source_records = [
        evaluate_episode(
            policy,
            layout,
            method=SOURCE_METHOD,
            policy_seed=args.policy_seed,
            environment_config=config.environment,
            device=device,
        )
        for layout in calibration
    ]
    result = source_environment_gate(oracle_records, source_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if not result["passed"]:
        raise SystemExit(
            "environment/source gate failed; tune templates on calibration only "
            "and do not inspect or modify test results"
        )
    frozen = args.output_dir / "layout_manifest.json"
    frozen.write_bytes(args.candidate_manifest.read_bytes())
    print(f"gate passed; frozen manifest: {frozen}")


if __name__ == "__main__":
    main()
