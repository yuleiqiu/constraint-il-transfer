"""Unit tests for gated corrective-policy selection."""

import unittest

import numpy as np

from selector import (
    SelectorConfig,
    chunk_noise_seeds,
    select_candidate,
    should_trigger,
)


class SelectorTest(unittest.TestCase):
    def test_gate_requires_obstructed_progress_before_contact(self):
        kwargs = {
            "nominal_progress_m": 0.02,
            "nominal_clearance_m": 0.005,
            "start_clearance_m": 0.03,
            "target_contact_seen": False,
            "obstacle_contact_seen": False,
            "obstacle_count": 2,
        }
        self.assertEqual(should_trigger(**kwargs), (True, []))
        active, reasons = should_trigger(**{**kwargs, "target_contact_seen": True})
        self.assertFalse(active)
        self.assertIn("target_contact_seen", reasons)

    def test_selects_least_deviating_eligible_candidate(self):
        nominal = np.zeros((2, 7), dtype=np.float64)
        raw = np.zeros((3, 2, 7), dtype=np.float64)
        # Move toward +x target. Candidate 0 is unsafe, 1 and 2 are safe.
        raw[0, :, :3] = [[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]
        raw[1, :, :3] = [[0.02, 0.03, 0.0], [0.02, 0.0, 0.0]]
        raw[2, :, :3] = [[0.02, -0.03, 0.0], [0.02, 0.0, 0.0]]
        normalized = raw.copy()
        normalized[1] *= 2.0
        normalized[2] *= 3.0
        index, rows = select_candidate(
            nominal_progress_m=0.032,
            nominal_normalized_chunk=nominal,
            candidate_raw_chunks=raw,
            candidate_normalized_chunks=normalized,
            start_position=np.array([0.0, 0.0, 0.0]),
            target_position=np.array([0.1, 0.0, 0.0]),
            centers=np.array([[0.02, 0.0, 0.0]]),
            radii=np.array([0.01]),
            config=SelectorConfig(
                candidate_clearance_m=0.0,
                progress_rule="source_relative",
            ),
        )
        self.assertFalse(rows[0]["eligible"])
        self.assertTrue(rows[1]["eligible"])
        self.assertTrue(rows[2]["eligible"])
        self.assertEqual(index, 1)

    def test_falls_back_when_no_candidate_is_eligible(self):
        chunks = np.zeros((2, 2, 7), dtype=np.float64)
        index, rows = select_candidate(
            nominal_progress_m=0.05,
            nominal_normalized_chunk=np.zeros((2, 7)),
            candidate_raw_chunks=chunks,
            candidate_normalized_chunks=chunks,
            start_position=np.zeros(3),
            target_position=np.array([1.0, 0.0, 0.0]),
            centers=np.zeros((0, 3)),
            radii=np.zeros(0),
            config=SelectorConfig(progress_rule="source_relative"),
        )
        self.assertIsNone(index)
        self.assertFalse(any(row["eligible"] for row in rows))

    def test_seed_schedule_is_stable_and_matches_fixed_state_sampling(self):
        first = chunk_noise_seeds(600_000_007, 3, 4)
        second = chunk_noise_seeds(600_000_007, 3, 4)
        self.assertEqual(first, second)
        self.assertEqual(first[0], first[1][0])
        self.assertEqual(len(set(first[1])), 4)


if __name__ == "__main__":
    unittest.main()
