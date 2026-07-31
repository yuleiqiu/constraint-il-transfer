#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from toy_recovery.config import ExperimentConfig
from toy_recovery.rollout import collect_dynamics_rollouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    manifest = collect_dynamics_rollouts(
        args.checkpoints,
        args.output,
        count=args.count or config.dynamics_rollouts,
        seed=args.seed,
        config=config.environment,
        device=args.device,
    )
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
