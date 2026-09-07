"""Audit and aggregate two independently run fixed-state checkpoint screens."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    left = read_jsonl(args.left)
    right = read_jsonl(args.right)
    if left.keys() != right.keys():
        raise ValueError("episode sets differ")

    counts = {
        "both_success": 0,
        "left_only_success": 0,
        "right_only_success": 0,
        "both_failure": 0,
    }
    paired_rows = []
    for episode in sorted(left):
        lhs, rhs = left[episode], right[episode]
        for row in (lhs, rhs):
            if row["initial_state_hash"] != row["restored_state_hash"]:
                raise ValueError(f"restore hash mismatch at episode {episode}")
        for field in ("episode", "layout_seed", "policy_seed", "initial_state_hash"):
            if lhs[field] != rhs[field]:
                raise ValueError(f"paired {field} mismatch at episode {episode}")
        if lhs["success"] and rhs["success"]:
            outcome = "both_success"
        elif lhs["success"]:
            outcome = "left_only_success"
        elif rhs["success"]:
            outcome = "right_only_success"
        else:
            outcome = "both_failure"
        counts[outcome] += 1
        paired_rows.append(
            {
                "episode": episode,
                "layout_seed": lhs["layout_seed"],
                "policy_seed": lhs["policy_seed"],
                "initial_state_hash": lhs["initial_state_hash"],
                "left_success": lhs["success"],
                "right_success": rhs["success"],
                "outcome": outcome,
            }
        )

    episodes = len(paired_rows)
    failure_to_success = counts["right_only_success"]
    success_to_failure = counts["left_only_success"]
    summary = {
        "left": {
            "label": next(iter(left.values()))["checkpoint"],
            "successes": sum(row["success"] for row in left.values()),
            "episodes": episodes,
        },
        "right": {
            "label": next(iter(right.values()))["checkpoint"],
            "successes": sum(row["success"] for row in right.values()),
            "episodes": episodes,
        },
        "pairing_audit": {
            "episode_sets_match": True,
            "layout_seeds_match": True,
            "policy_seeds_match": True,
            "initial_state_hashes_match": True,
            "all_restore_hashes_match": True,
        },
        "paired_outcomes": counts,
        "right_failure_to_success": failure_to_success,
        "right_success_to_failure": success_to_failure,
        "records": paired_rows,
    }
    for side in ("left", "right"):
        summary[side]["success_rate"] = (
            summary[side]["successes"] / summary[side]["episodes"]
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
