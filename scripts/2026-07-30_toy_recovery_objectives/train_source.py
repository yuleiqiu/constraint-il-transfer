#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from toy_recovery.config import ExperimentConfig
from toy_recovery.data import TrajectoryWindowDataset
from toy_recovery.training import train_source_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    dataset = TrajectoryWindowDataset(
        [args.data],
        config.policy,
        max_step=config.environment.max_step,
    )
    result = train_source_policy(
        dataset,
        args.output_dir,
        seed=args.seed,
        updates=args.updates or config.source_updates,
        batch_size=args.batch_size,
        resume_checkpoint=args.resume,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
