"""Audit and aggregate fixed-state RaisedMilk baseline / guided results."""

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    by_episode = {row["episode"]: row for row in rows}
    if len(by_episode) != len(rows):
        raise ValueError(f"duplicate episodes in {path}")
    return by_episode


def read_jsonl_parts(paths):
    merged = {}
    identity_fields = (
        "episode",
        "layout_seed",
        "policy_seed",
        "initial_state_hash",
        "condition",
        "success",
        "safe_success",
        "collision_any",
        "ncr",
    )
    for path in paths:
        for episode, row in read_jsonl(path).items():
            if episode in merged:
                for field in identity_fields:
                    if merged[episode][field] != row[field]:
                        raise ValueError(
                            f"conflicting duplicate episode {episode} field {field}"
                        )
                continue
            merged[episode] = row
    return merged


def condition_summary(rows):
    count = len(rows)
    return {
        "episodes": count,
        "task_success": sum(row["success"] for row in rows),
        "task_sr": sum(row["success"] for row in rows) / count,
        "safe_success": sum(row["safe_success"] for row in rows),
        "safe_sr": sum(row["safe_success"] for row in rows) / count,
        "collision_any": sum(row["collision_any"] for row in rows),
        "cr": sum(row["collision_any"] for row in rows) / count,
        "ncr_count": sum(row["ncr"] for row in rows),
        "ncr": sum(row["ncr"] for row in rows) / count,
        "four_way": {
            "safe_success": sum(row["safe_success"] for row in rows),
            "success_with_collision": sum(
                row["success_with_collision"] for row in rows
            ),
            "collision_failure": sum(row["collision_failure"] for row in rows),
            "collision_free_non_completion": sum(row["ncr"] for row in rows),
        },
    }


def transitions(reference, candidate, key):
    failure_to_success = 0
    success_to_failure = 0
    for episode in reference:
        before = reference[episode][key]
        after = candidate[episode][key]
        failure_to_success += int(not before and after)
        success_to_failure += int(before and not after)
    return {
        "failure_to_success": failure_to_success,
        "success_to_failure": success_to_failure,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--guided", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = read_jsonl(args.baseline)
    guided = read_jsonl_parts(args.guided)
    if baseline.keys() != guided.keys():
        raise ValueError("episode sets differ")
    for episode in baseline:
        before, after = baseline[episode], guided[episode]
        if before["condition"] != "baseline" or after["condition"] != "guided":
            raise ValueError(f"condition mismatch at episode {episode}")
        for row in (before, after):
            if row["initial_state_hash"] != row["restored_state_hash"]:
                raise ValueError(f"restore hash mismatch at episode {episode}")
        for field in ("episode", "layout_seed", "policy_seed", "initial_state_hash"):
            if before[field] != after[field]:
                raise ValueError(f"paired {field} mismatch at episode {episode}")

    baseline_rows = [baseline[index] for index in sorted(baseline)]
    guided_rows = [guided[index] for index in sorted(guided)]
    summary = {
        "pairing_audit": {
            "episode_sets_match": True,
            "layout_seeds_match": True,
            "policy_seeds_match": True,
            "initial_state_hashes_match": True,
            "all_restore_hashes_match": True,
        },
        "baseline": condition_summary(baseline_rows),
        "guided": condition_summary(guided_rows),
        "task_transitions": transitions(baseline, guided, "success"),
        "safe_transitions": transitions(baseline, guided, "safe_success"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
