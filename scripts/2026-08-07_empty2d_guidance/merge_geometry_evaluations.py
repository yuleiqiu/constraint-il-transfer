"""Merge independently evaluated Circle and Square shards."""

import argparse
import json
from pathlib import Path

from paired_evaluation import conclusion, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()
    shards = {
        geometry: json.loads((args.eval_dir / f"result_{geometry}.json").read_text())
        for geometry in ("circle", "square")
    }
    seeds = {shard["training_seed"] for shard in shards.values()}
    checkpoints = {shard["checkpoint"] for shard in shards.values()}
    if len(seeds) != 1 or len(checkpoints) != 1:
        raise ValueError("Geometry shards must use one training seed and checkpoint")
    calibration = {
        geometry: shards[geometry]["parameters"][geometry]
        for geometry in shards
    }
    tests = {geometry: shards[geometry]["tests"][geometry] for geometry in shards}
    final_conclusion = conclusion(tests)
    manifest = {
        "checkpoint": checkpoints.pop(),
        "training_seed": seeds.pop(),
        "calibration_seeds": shards["circle"]["calibration_seeds"],
        "test_seeds": shards["circle"]["test_seeds"],
        "parameters": calibration,
        "tests": tests,
        "conclusion": final_conclusion,
    }
    (args.eval_dir / "locked_parameters.json").write_text(
        json.dumps(calibration, indent=2) + "\n"
    )
    (args.eval_dir / "result.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True) + "\n"
    )
    write_report(
        args.eval_dir / "report.md",
        manifest["training_seed"],
        calibration,
        tests,
        final_conclusion,
    )
    print(json.dumps(manifest, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
