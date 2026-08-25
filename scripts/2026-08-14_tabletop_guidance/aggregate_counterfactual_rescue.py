"""Merge partitioned Tabletop counterfactual rescue evaluations."""

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "counterfactual_rescue_eval", SCRIPT_DIR / "counterfactual_rescue_eval.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def wilson(successes, trials, z=1.959963984540054):
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [center - radius, center + radius]


def outcome(metrics):
    if metrics["safe_success"]:
        return "safe_success"
    if metrics["success_with_collision"]:
        return "success_with_collision"
    if metrics["collision_failure"]:
        return "collision_failure"
    return "collision_free_non_completion"


def contact_episode_counts(rows, branch):
    counts = Counter()
    for row in rows:
        present = {
            (item["category"], item["obstacle"])
            for item in row[branch]["contact_audit"]["physical_contact_pairs"]
        }
        counts.update(present)
    return {
        f"{category}:{obstacle}": count
        for (category, obstacle), count in sorted(counts.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    rows = []
    manifests = []
    for part in args.parts:
        manifests.append(json.loads((part / "manifest.json").read_text()))
        with (part / "episode_metrics.jsonl").open() as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    keys = [(row["scene"], row["policy_seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate scene / policy-seed pair across partitions")
    rows.sort(key=lambda row: (row["scene"], row["policy_seed"]))
    summary = MODULE.summarize(rows)
    eligible = [
        row for row in rows
        if row["clean"]["success"] and row["obstacle_baseline"]["collision_failure"]
    ]
    summary["rescue_rate_wilson_95"] = wilson(
        summary["guided_safe_success_rescues"], summary["eligible_pairs"]
    )
    summary["all_branch_outcomes"] = {
        branch: dict(Counter(outcome(row[branch]) for row in rows))
        for branch in ("clean", "obstacle_baseline", "obstacle_guided")
    }
    summary["eligible_guided_outcomes"] = dict(
        Counter(outcome(row["obstacle_guided"]) for row in eligible)
    )
    summary["eligible_physical_contact_episode_counts"] = {
        branch: contact_episode_counts(eligible, branch)
        for branch in ("obstacle_baseline", "obstacle_guided")
    }
    summary["eligible_eef_xy_proxy_overlap"] = {
        branch: int(
            sum(
                row[branch]["contact_audit"]["eef_milk_xy_footprint_overlap_proxy"]
                for row in eligible
            )
        )
        for branch in ("obstacle_baseline", "obstacle_guided")
    }
    summary["per_scene"] = []
    for scene in sorted({row["scene"] for row in rows}):
        selected = [row for row in rows if row["scene"] == scene]
        item = MODULE.summarize(selected)
        item["scene"] = scene
        summary["per_scene"].append(item)

    initial_matches = {}
    for manifest in manifests:
        initial_matches.update(manifest["initial_state_match"])
    merged_manifest = {
        "parts": [str(path.resolve()) for path in args.parts],
        "checkpoint": manifests[0]["checkpoint"],
        "state_bank": manifests[0]["state_bank"],
        "scenes": sorted({row["scene"] for row in rows}),
        "policy_seeds": sorted({row["policy_seed"] for row in rows}),
        "horizon": manifests[0]["horizon"],
        "guidance": manifests[0]["guidance"],
        "eligibility": manifests[0]["eligibility"],
        "rescue": manifests[0]["rescue"],
        "initial_state_match": initial_matches,
        "eef_collision_note": manifests[0]["eef_collision_note"],
    }
    (args.out_dir / "episode_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "manifest.json").write_text(json.dumps(merged_manifest, indent=2) + "\n")

    lines = [
        "# Tabletop counterfactual rescue evaluation",
        "",
        f"Paired triads: {summary['pairs']} (five layouts x five diffusion-noise seeds).",
        f"Clean successes: {summary['clean_successes']}/{summary['pairs']}.",
        f"Eligible clean-success / obstacle-collision-failure pairs: {summary['eligible_pairs']}.",
        f"Guided safe task-completing rescues: {summary['guided_safe_success_rescues']}/{summary['eligible_pairs']} "
        + (f"= {summary['rescue_rate']:.3f}." if summary["rescue_rate"] is not None else "(undefined)."),
        "",
        "## Per-scene rescue counts",
        "",
        "| Scene | Pairs | Clean success | Eligible | Safe rescues | Rescue rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["per_scene"]:
        rate = "n/a" if item["rescue_rate"] is None else f"{item['rescue_rate']:.2f}"
        lines.append(
            f"| {item['scene']} | {item['pairs']} | {item['clean_successes']} | "
            f"{item['eligible_pairs']} | {item['guided_safe_success_rescues']} | {rate} |"
        )
    lines += [
        "",
        "## Eligibility-scoped guided outcomes",
        "",
        "```json",
        json.dumps(summary["eligible_guided_outcomes"], indent=2),
        "```",
        "",
        "## Collision-pair audit on eligible pairs",
        "",
        "```json",
        json.dumps(summary["eligible_physical_contact_episode_counts"], indent=2),
        "```",
        "",
        "The EEF is a site rather than a collision geom. EEF-versus-Milk is therefore "
        "reported as an XY footprint-overlap proxy; physical contacts are separated into "
        "gripper hand, gripper finger, arm link, and carried Can.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
