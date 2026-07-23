"""Directly runnable tests for the Stage-1 feasibility solver."""

import unittest

import numpy as np

from feasibility import (
    FeasibilityConfig,
    closest_target_progress,
    positions_to_actions,
    reconstruct_positions,
    segment_circle_clearances_xy,
    solve_eight_step_feasibility,
)


class GeometryTests(unittest.TestCase):
    def test_position_action_round_trip_preserves_other_dimensions(self):
        start = np.array((0.1, -0.2, 0.9))
        actions = np.zeros((8, 7), dtype=np.float64)
        actions[:, :3] = np.array((0.01, -0.002, 0.003))
        actions[:, 3:] = np.arange(32).reshape(8, 4)
        positions = reconstruct_positions(start, actions)
        reconstructed = positions_to_actions(start, positions, actions)
        np.testing.assert_allclose(reconstructed, actions, atol=1e-12)
        np.testing.assert_array_equal(reconstructed[:, 3:], actions[:, 3:])

    def test_segment_circle_clearance(self):
        start = np.array((-2.0, 0.0, 0.0))
        positions = np.array(((2.0, 0.0, 0.0), (2.0, 2.0, 0.0)))
        centers = np.array(((0.0, 0.0), (0.0, 2.0)))
        radii = np.array((1.0, 1.0))
        clearances = segment_circle_clearances_xy(start, positions, centers, radii)
        self.assertAlmostEqual(clearances[0, 0], -1.0)
        self.assertAlmostEqual(clearances[0, 1], 1.0)
        self.assertAlmostEqual(clearances[1, 0], 1.0)

    def test_closest_progress_uses_all_waypoints(self):
        start = np.array((0.0, 0.0, 0.0))
        target = np.array((1.0, 0.0, 0.0))
        positions = np.array(((0.6, 0.0, 0.0), (-0.5, 0.0, 0.0)))
        self.assertAlmostEqual(closest_target_progress(start, positions, target), 0.6)


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.start = np.zeros(3)
        self.actions = np.zeros((8, 7), dtype=np.float64)
        self.actions[:, 0] = 0.05
        self.actions[:, 3:] = np.linspace(-0.5, 0.5, 32).reshape(8, 4)
        self.target = np.array((0.4, 0.0, 0.0))
        self.low = np.full(7, -1.0)
        self.high = np.full(7, 1.0)

    def test_feasible_baseline_is_identity(self):
        result = solve_eight_step_feasibility(
            start_position=self.start,
            baseline_actions=self.actions,
            target_position=self.target,
            obstacle_centers=np.array(((0.0, 1.0, 0.0),)),
            obstacle_radii=np.array((0.05,)),
            action_low=self.low,
            action_high=self.high,
        )
        self.assertTrue(result.feasible, result.solver_message)
        np.testing.assert_allclose(result.corrected_actions, self.actions, atol=1e-6)

    def test_solver_routes_around_blocking_circle_and_preserves_progress(self):
        config = FeasibilityConfig(
            clearance_margin_m=0.01,
            progress_tolerance_m=0.01,
            trust_radius_m=0.12,
            smoothness_weight=0.2,
        )
        result = solve_eight_step_feasibility(
            start_position=self.start,
            baseline_actions=self.actions,
            target_position=self.target,
            obstacle_centers=np.array(((0.2, 0.0, 0.0),)),
            obstacle_radii=np.array((0.05,)),
            action_low=self.low,
            action_high=self.high,
            config=config,
        )
        self.assertTrue(result.feasible, result.to_dict())
        self.assertGreaterEqual(result.corrected_min_clearance_m, 0.01 - 1e-5)
        self.assertGreaterEqual(
            result.corrected_target_progress_m,
            result.baseline_target_progress_m - 0.01 - 1e-5,
        )
        self.assertLessEqual(result.max_waypoint_displacement_m, 0.12 + 1e-5)
        np.testing.assert_array_equal(result.corrected_actions[:, 3:], self.actions[:, 3:])

    def test_solver_reports_infeasible_when_trust_region_is_too_small(self):
        result = solve_eight_step_feasibility(
            start_position=self.start,
            baseline_actions=self.actions,
            target_position=self.target,
            obstacle_centers=np.array(((0.2, 0.0, 0.0),)),
            obstacle_radii=np.array((0.05,)),
            action_low=self.low,
            action_high=self.high,
            config=FeasibilityConfig(
                clearance_margin_m=0.01,
                trust_radius_m=0.005,
            ),
        )
        self.assertFalse(result.feasible)

    def test_recovery_constraint_can_exit_conservative_margin(self):
        start = np.array((0.14, 0.0, 0.0))
        actions = self.actions.copy()
        target = np.array((0.54, 0.0, 0.0))
        result = solve_eight_step_feasibility(
            start_position=start,
            baseline_actions=actions,
            target_position=target,
            obstacle_centers=np.array(((0.2, 0.0, 0.0),)),
            obstacle_radii=np.array((0.05,)),
            action_low=self.low,
            action_high=self.high,
            config=FeasibilityConfig(
                clearance_margin_m=0.01,
                clearance_recovery_steps=2,
                progress_tolerance_m=0.02,
                trust_radius_m=0.12,
                smoothness_weight=0.2,
            ),
        )
        self.assertTrue(result.feasible, result.to_dict())
        self.assertGreaterEqual(result.corrected_min_clearance_m, -1e-5)


if __name__ == "__main__":
    unittest.main()
