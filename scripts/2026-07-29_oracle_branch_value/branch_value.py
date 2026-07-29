"""Pure aggregation and ranking for simulator-branch value diagnostics."""

from collections import Counter


OUTCOME_ORDER = {
    "collision_failure": 0,
    "ncr": 1,
    "success_with_collision": 2,
    "safe_success": 3,
}


def outcome_label(success, collision_any):
    if success:
        return "success_with_collision" if collision_any else "safe_success"
    return "collision_failure" if collision_any else "ncr"


def oracle_sort_key(branch):
    """Rank by task outcome, then task progress, then source proximity."""

    staged = tuple(float(value) for value in branch["final_staged_rewards"])
    return (
        OUTCOME_ORDER[branch["outcome"]],
        int(branch["target_contact_any"]),
        staged,
        float(branch["maximum_target_lift_m"]),
        float(branch["maximum_target_displacement_m"]),
        -float(branch["source_normalized_rms_deviation"]),
        -int(branch["candidate_index"]),
    )


def select_oracle_branch(branches, *, eligible_only=False):
    selected = [
        branch
        for branch in branches
        if not eligible_only
        or branch["candidate_type"] == "source"
        or branch["geometry_eligible"]
    ]
    if not selected:
        raise ValueError("At least one branch must be selectable")
    return max(selected, key=oracle_sort_key)


def summarize_states(states):
    if not states:
        return {
            "states": 0,
            "source_outcomes": {},
            "any_candidate_outcomes": {},
        }

    source_outcomes = Counter()
    oracle_any_outcomes = Counter()
    oracle_eligible_outcomes = Counter()
    any_success = 0
    any_safe_success = 0
    eligible_success = 0
    eligible_safe_success = 0
    any_failure_to_success = 0
    eligible_failure_to_success = 0
    any_success_to_failure = 0
    eligible_success_to_failure = 0
    states_with_geometry_eligible = 0
    total_geometry_eligible = 0

    for state in states:
        branches = state["branches"]
        source = next(
            branch for branch in branches if branch["candidate_type"] == "source"
        )
        oracle_any = select_oracle_branch(branches)
        oracle_eligible = select_oracle_branch(branches, eligible_only=True)
        source_outcomes[source["outcome"]] += 1
        oracle_any_outcomes[oracle_any["outcome"]] += 1
        oracle_eligible_outcomes[oracle_eligible["outcome"]] += 1

        corrective = [
            branch for branch in branches if branch["candidate_type"] == "corrective"
        ]
        eligible = [branch for branch in corrective if branch["geometry_eligible"]]
        successes = [branch for branch in corrective if branch["success"]]
        safe_successes = [
            branch
            for branch in corrective
            if branch["success"] and not branch["collision_any"]
        ]
        eligible_successes = [branch for branch in eligible if branch["success"]]
        eligible_safe_successes = [
            branch
            for branch in eligible
            if branch["success"] and not branch["collision_any"]
        ]
        any_success += int(bool(successes))
        any_safe_success += int(bool(safe_successes))
        eligible_success += int(bool(eligible_successes))
        eligible_safe_success += int(bool(eligible_safe_successes))
        states_with_geometry_eligible += int(bool(eligible))
        total_geometry_eligible += len(eligible)

        any_failure_to_success += int(
            not source["success"] and oracle_any["success"]
        )
        eligible_failure_to_success += int(
            not source["success"] and oracle_eligible["success"]
        )
        any_success_to_failure += int(
            source["success"] and not oracle_any["success"]
        )
        eligible_success_to_failure += int(
            source["success"] and not oracle_eligible["success"]
        )

    return {
        "states": len(states),
        "source_outcomes": dict(source_outcomes),
        "oracle_any_outcomes": dict(oracle_any_outcomes),
        "oracle_geometry_eligible_outcomes": dict(oracle_eligible_outcomes),
        "states_with_any_corrective_success": any_success,
        "states_with_any_corrective_safe_success": any_safe_success,
        "states_with_eligible_corrective_success": eligible_success,
        "states_with_eligible_corrective_safe_success": eligible_safe_success,
        "states_with_geometry_eligible_candidate": states_with_geometry_eligible,
        "total_geometry_eligible_candidates": total_geometry_eligible,
        "oracle_any_failure_to_success": any_failure_to_success,
        "oracle_any_success_to_failure": any_success_to_failure,
        "oracle_eligible_failure_to_success": eligible_failure_to_success,
        "oracle_eligible_success_to_failure": eligible_success_to_failure,
    }
