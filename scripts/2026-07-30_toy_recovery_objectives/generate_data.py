#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from toy_recovery.config import ExperimentConfig
from toy_recovery.data import generate_oracle_demonstrations, write_trajectories
from toy_recovery.env import write_layout_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--demo-count", type=int)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    output = args.output_dir or Path(config.output_dir)
    demo_count = args.demo_count or config.d0_demonstrations
    trajectories = generate_oracle_demonstrations(
        demo_count,
        difficulty=0,
        seed_start=config.layout_seed_base,
        config=config.environment,
    )
    write_trajectories(output / "data" / "d0_demonstrations.hdf5", trajectories)
    split_counts = {
        "calibration": config.calibration_layouts_per_difficulty,
        "scale_tuning": config.scale_tuning_layouts_per_difficulty,
        "screen": config.screen_layouts_per_difficulty,
        "test": config.test_layouts_per_difficulty,
    }
    split_offsets = {
        "calibration": 100_000,
        "scale_tuning": 200_000,
        "screen": 300_000,
        "test": 500_000,
    }
    split_seeds = {
        split: {
            difficulty: list(
                range(
                    config.layout_seed_base
                    + split_offsets[split]
                    + difficulty * 10_000,
                    config.layout_seed_base
                    + split_offsets[split]
                    + difficulty * 10_000
                    + count,
                )
            )
            for difficulty in range(3)
        }
        for split, count in split_counts.items()
    }
    # Freeze and commit this candidate only after the calibration-only
    # environment/source gate passes. Test layouts remain untouched.
    write_layout_manifest(
        output / "candidate_layout_manifest.json",
        split_seeds=split_seeds,
        config=config.environment,
    )


if __name__ == "__main__":
    main()
