"""Render one typical held-out pair for each available outcome category."""

import argparse
import json
from pathlib import Path

from empty2d_guidance.video import render_pair


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def category(baseline, guided):
    if baseline["collision_any"] and not baseline["success"] and guided["safe_success"]:
        return "baseline_collision_to_guided_success"
    if baseline["success"] and guided["success"]:
        return "both_success"
    if not baseline["success"] and not guided["success"]:
        return "both_failure"
    if baseline["success"] and not guided["success"]:
        return "baseline_success_to_guided_failure"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rendered = []
    for environment in ("circle", "square"):
        baseline = read_jsonl(args.eval_dir / f"test_{environment}_baseline.jsonl")
        guided = read_jsonl(args.eval_dir / f"test_{environment}_guided.jsonl")
        guided_by_seed = {row["policy_seed"]: row for row in guided}
        selected = {}
        for before in baseline:
            after = guided_by_seed[before["policy_seed"]]
            name = category(before, after)
            if name is not None and name not in selected:
                selected[name] = (before, after)
        for name, (before, after) in selected.items():
            output = args.out_dir / f"{environment}_{name}_seed{before['policy_seed']}.mp4"
            render_pair(output, before, after)
            rendered.append(str(output.resolve()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps({"videos": rendered}, indent=2) + "\n")
    print(json.dumps({"videos": rendered}, indent=2))


if __name__ == "__main__":
    main()
