"""Directly runnable tests for the policy-manifold feasibility solver."""

import unittest

import numpy as np

from manifold_feasibility import (
    ManifoldConfig,
    fit_manifold,
    reconstruct_positions,
    solve_manifold_projection,
)


class ManifoldTests(unittest.TestCase):
    def test_invalid_support_multiplier_is_rejected(self):
        with self.assertRaises(ValueError):
            ManifoldConfig(support_multiplier=0.0).validate()

    def test_fit_contains_empirical_quantile(self):
        rng = np.random.default_rng(4)
        samples = np.zeros((64, 8, 3), dtype=np.float64)
        samples[:, :, :2] = rng.normal(scale=0.01, size=(64, 8, 2))
        config = ManifoldConfig(support_quantile=0.90)
        model = fit_manifold(samples, config)
        contained = np.mean(
            model.sample_distances_d2 <= model.support_threshold_d2 + 1e-12
        )
        self.assertGreaterEqual(contained, 0.89)
        self.assertLessEqual(contained, 0.92)
        self.assertGreater(model.effective_rank, 1.0)

    def test_safe_nominal_is_preserved(self):
        start = np.zeros(3)
        actions = np.zeros((8, 7))
        actions[:, 0] = 0.02
        nominal_positions = reconstruct_positions(start, actions)
        rng = np.random.default_rng(5)
        samples = np.repeat(nominal_positions[None], 32, axis=0)
        samples[:, :, :2] += rng.normal(scale=0.002, size=(32, 8, 2))
        model, result = solve_manifold_projection(
            start_position=start,
            nominal_actions=actions,
            sample_positions=samples,
            target_position=np.array((0.16, 0.0, 0.0)),
            obstacle_centers=np.array(((0.0, 1.0, 0.0),)),
            obstacle_radii=np.array((0.05,)),
            action_low=np.full(7, -1.0),
            action_high=np.full(7, 1.0),
            clearance_margin_m=0.01,
        )
        self.assertTrue(result.feasible, result.to_dict())
        self.assertLess(result.maximum_waypoint_displacement_m, 1e-5)
        self.assertLessEqual(
            result.support_distance_d2,
            model.support_threshold_d2 + 1e-5,
        )

    def test_safe_mode_inside_support_is_found(self):
        start = np.zeros(3)
        actions = np.zeros((8, 7))
        actions[:, 0] = 0.04
        nominal_positions = reconstruct_positions(start, actions)
        phase = np.sin(np.pi * np.arange(1, 9) / 9.0)
        samples = []
        for amplitude in np.linspace(-0.10, 0.10, 48):
            points = nominal_positions.copy()
            points[:, 1] += amplitude * phase
            samples.append(points)
        model, result = solve_manifold_projection(
            start_position=start,
            nominal_actions=actions,
            sample_positions=np.asarray(samples),
            target_position=np.array((0.32, 0.0, 0.0)),
            obstacle_centers=np.array(((0.16, 0.0, 0.0),)),
            obstacle_radii=np.array((0.035,)),
            action_low=np.full(7, -1.0),
            action_high=np.full(7, 1.0),
            clearance_margin_m=0.01,
            config=ManifoldConfig(progress_tolerance_m=0.01),
        )
        self.assertTrue(result.feasible, result.to_dict())
        self.assertGreaterEqual(result.corrected_min_clearance_m, 0.01 - 1e-5)
        self.assertGreaterEqual(
            result.corrected_target_progress_m,
            result.baseline_target_progress_m - 0.01 - 1e-5,
        )
        self.assertLessEqual(result.support_ratio, 1.0 + 1e-5)


if __name__ == "__main__":
    unittest.main()
