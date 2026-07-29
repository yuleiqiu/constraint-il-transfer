"""Unit tests for oracle branch aggregation."""

import unittest

from branch_value import outcome_label, select_oracle_branch, summarize_states


def branch(index, outcome, *, source=False, eligible=False, deviation=0.0):
    success = outcome in ("safe_success", "success_with_collision")
    collision = outcome in ("success_with_collision", "collision_failure")
    return {
        "candidate_index": index,
        "candidate_type": "source" if source else "corrective",
        "geometry_eligible": eligible,
        "outcome": outcome,
        "success": success,
        "collision_any": collision,
        "target_contact_any": success,
        "final_staged_rewards": [0.0, float(success)],
        "maximum_target_lift_m": float(success),
        "maximum_target_displacement_m": float(success),
        "source_normalized_rms_deviation": deviation,
    }


class BranchValueTest(unittest.TestCase):
    def test_outcome_partition(self):
        self.assertEqual(outcome_label(True, False), "safe_success")
        self.assertEqual(outcome_label(True, True), "success_with_collision")
        self.assertEqual(outcome_label(False, True), "collision_failure")
        self.assertEqual(outcome_label(False, False), "ncr")

    def test_oracle_prefers_task_success_then_safety(self):
        branches = [
            branch(-1, "collision_failure", source=True),
            branch(0, "success_with_collision", eligible=True, deviation=0.1),
            branch(1, "safe_success", eligible=True, deviation=0.2),
        ]
        self.assertEqual(select_oracle_branch(branches)["candidate_index"], 1)

    def test_eligible_oracle_retains_source_fallback(self):
        branches = [
            branch(-1, "safe_success", source=True),
            branch(0, "collision_failure", eligible=False),
        ]
        selected = select_oracle_branch(branches, eligible_only=True)
        self.assertEqual(selected["candidate_type"], "source")

    def test_summary_counts_oracle_rescue(self):
        states = [
            {
                "branches": [
                    branch(-1, "collision_failure", source=True),
                    branch(0, "safe_success", eligible=True),
                ]
            }
        ]
        summary = summarize_states(states)
        self.assertEqual(summary["oracle_any_failure_to_success"], 1)
        self.assertEqual(summary["oracle_eligible_failure_to_success"], 1)
        self.assertEqual(summary["states_with_any_corrective_safe_success"], 1)


if __name__ == "__main__":
    unittest.main()
