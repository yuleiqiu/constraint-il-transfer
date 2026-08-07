"""Task-level paired metrics for Empty2D rollouts."""

from __future__ import annotations

import math
from collections import Counter


OUTCOMES = (
    "safe_success",
    "success_with_collision",
    "collision_failure",
    "collision_free_non_completion",
)


def outcome(row: dict) -> str:
    if row["success"] and not row["collision_any"]:
        return "safe_success"
    if row["success"] and row["collision_any"]:
        return "success_with_collision"
    if row["collision_any"]:
        return "collision_failure"
    return "collision_free_non_completion"


def exact_mcnemar_p(failure_to_success: int, success_to_failure: int) -> float:
    discordant = failure_to_success + success_to_failure
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(failure_to_success, success_to_failure) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty rollout set")
    count = len(rows)
    outcomes = Counter(outcome(row) for row in rows)
    return {
        "rollouts": count,
        "task_sr": sum(row["success"] for row in rows) / count,
        "safe_sr": sum(row["success"] and not row["collision_any"] for row in rows) / count,
        "cr": sum(row["collision_any"] for row in rows) / count,
        "ncr": sum((not row["success"]) and (not row["collision_any"]) for row in rows) / count,
        "outcomes": {name: outcomes[name] for name in OUTCOMES},
        "minimum_clearance": min(row["minimum_clearance"] for row in rows),
        "mean_trigger_rate": sum(row.get("trigger_rate", 0.0) for row in rows) / count,
        "mean_trajectory_length": sum(row["trajectory_length"] for row in rows) / count,
        "mean_clip_rate": sum(row["clip_rate"] for row in rows) / count,
    }


def paired_summary(baseline: list[dict], guided: list[dict]) -> dict:
    baseline_by_seed = {row["policy_seed"]: row for row in baseline}
    guided_by_seed = {row["policy_seed"]: row for row in guided}
    if baseline_by_seed.keys() != guided_by_seed.keys():
        raise ValueError("Baseline and guided policy seeds do not match")
    failure_to_success = 0
    success_to_failure = 0
    for seed in baseline_by_seed:
        before = baseline_by_seed[seed]["success"]
        after = guided_by_seed[seed]["success"]
        failure_to_success += int((not before) and after)
        success_to_failure += int(before and (not after))
    return {
        "baseline": summarize(baseline),
        "guided": summarize(guided),
        "failure_to_success": failure_to_success,
        "success_to_failure": success_to_failure,
        "net_rescues": failure_to_success - success_to_failure,
        "exact_mcnemar_p": exact_mcnemar_p(failure_to_success, success_to_failure),
    }
