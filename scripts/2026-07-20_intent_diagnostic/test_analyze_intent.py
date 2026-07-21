"""Directly runnable pure-Python tests for analyze_intent.py.

Run with:
    uv run python scripts/2026-07-20_intent_diagnostic/test_analyze_intent.py
"""

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze_intent.py")
SPEC = importlib.util.spec_from_file_location("intent_diagnostic", MODULE_PATH)
intent = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(intent)


class SegmentCircleTests(unittest.TestCase):
    def test_crossing_tangent_and_clear(self):
        crossing = intent.segment_circle_clearance_xy((-2, 0), (2, 0), (0, 0), 1)
        tangent = intent.segment_circle_clearance_xy((-2, 1), (2, 1), (0, 0), 1)
        clear = intent.segment_circle_clearance_xy((-2, 2), (2, 2), (0, 0), 1)
        self.assertLess(crossing, 0)
        self.assertTrue(math.isclose(tangent, 0.0, abs_tol=1e-12))
        self.assertGreater(clear, 0)


class ProgressTests(unittest.TestCase):
    def test_closest_progress_can_be_positive_when_endpoint_is_negative(self):
        path = np.asarray([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        target = np.asarray([1.0, 0.0, 0.0])
        self.assertAlmostEqual(intent.closest_progress(path, target), 0.6)
        self.assertAlmostEqual(intent.endpoint_progress(path, target), -0.5)

    def test_closest_progress_is_negative_when_all_waypoints_move_away(self):
        path = np.asarray([[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0], [-0.2, 0.0, 0.0]])
        target = np.asarray([1.0, 0.0, 0.0])
        self.assertAlmostEqual(intent.closest_progress(path, target), -0.1)


class CriticalChunkTests(unittest.TestCase):
    def setUp(self):
        self.starts = [0, 8, 16]
        self.lengths = [8, 8, 4]

    def test_main_contact_selects_containing_chunk(self):
        selected, reason = intent.select_critical_chunk(self.starts, self.lengths, "main_analysis", None, 9)
        self.assertEqual((selected, reason), (1, "first_obstacle_contact"))

    def test_main_no_contact_selects_last_chunk(self):
        selected, reason = intent.select_critical_chunk(self.starts, self.lengths, "main_analysis", None, None)
        self.assertEqual((selected, reason), (2, "last_chunk_no_contact"))

    def test_success_and_downstream_select_target_chunk(self):
        for cohort in ("success_calibration", "downstream_failure"):
            selected, reason = intent.select_critical_chunk(self.starts, self.lengths, cohort, 16, 3)
            self.assertEqual((selected, reason), (2, "first_target_contact"))


class ClassificationTests(unittest.TestCase):
    def test_three_categories_are_mutually_exclusive_and_exhaustive(self):
        cases = [
            (0.01, -0.001, "target_directed_obstructed"),
            (0.0, -0.001, "non_target_directed"),
            (-0.01, 0.02, "non_target_directed"),
            (0.01, 0.02, "target_directed_clear"),
        ]
        observed = []
        for progress, clearance, expected in cases:
            category = intent.classify_chunk(progress, clearance)
            self.assertIn(category, intent.CATEGORIES)
            self.assertEqual(category, expected)
            observed.append(category)
        self.assertEqual(set(observed), set(intent.CATEGORIES))

    def test_threshold_and_margin_sensitivity(self):
        self.assertEqual(intent.classify_chunk(0.0006, 0.005, 0.0005, 0.01), "target_directed_obstructed")
        self.assertEqual(intent.classify_chunk(0.0006, 0.005, 0.001, 0.01), "non_target_directed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
