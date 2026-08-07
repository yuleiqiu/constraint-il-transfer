"""Require replication in at least two of three training seeds."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs=3)
    parser.add_argument("--out", type=Path, default=Path("outputs/empty2d_guidance/final_result.json"))
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.results]
    seeds = sorted(run["training_seed"] for run in runs)
    if seeds != [42, 43, 44]:
        raise ValueError("Expected exactly training seeds 42, 43, and 44")
    reference = next(run for run in runs if run["training_seed"] == 42)["parameters"]
    if any(run["parameters"] != reference for run in runs):
        raise ValueError("All seeds must use seed-42 locked rho/Q")
    counts = {
        geometry: sum(run["tests"][geometry]["passed"] for run in runs)
        for geometry in ("circle", "square")
    }
    replicated = {geometry: count >= 2 for geometry, count in counts.items()}
    if all(replicated.values()):
        conclusion = "Guidance robustly works in both idealized point geometries in at least 2/3 training seeds."
    elif any(replicated.values()):
        passing = [name for name, value in replicated.items() if value]
        conclusion = f"Shape-dependent replicated partial pass: {passing[0]} only."
    else:
        conclusion = "Current guidance is not replicated in the simplified environments."
    output = {
        "training_seeds": seeds,
        "locked_parameters": reference,
        "pass_count": counts,
        "replicated": replicated,
        "conclusion": conclusion,
        "claim_boundary": "A pass validates idealized point geometry only and cannot prove geometry is the unique D2 failure cause.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
