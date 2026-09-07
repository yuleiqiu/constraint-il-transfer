"""Pure checks for the obstacle-free wide-support Tabletop teacher."""

from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import wide_support_teacher as MODULE  # noqa: E402


def test_endpoint_conditioned_routes_are_exact_bounded_and_symmetric():
    config = MODULE.WideSupportTeacherConfig()
    start = np.array([0.01, -0.22, 1.02])
    goal = np.array([-0.02, 0.22, 1.02])
    signed_peaks = []
    midpoint_offsets = []
    for seed in range(15000, 15100):
        route, normal, peak, midpoint, attempts = MODULE.sample_lateral_route(
            start, goal, np.random.default_rng(seed), config
        )
        np.testing.assert_allclose(route[0], start, atol=1e-12)
        np.testing.assert_allclose(route[-1], goal, atol=1e-12)
        assert abs(peak) <= config.route_max_lateral_m
        assert attempts >= 1
        signed_peaks.append(peak)
        midpoint_offsets.append(midpoint)
    signed_peaks = np.asarray(signed_peaks)
    midpoint_offsets = np.asarray(midpoint_offsets)
    assert np.count_nonzero(signed_peaks > 0.065) >= 15
    assert np.count_nonzero(signed_peaks < -0.065) >= 15
    assert np.count_nonzero(np.abs(signed_peaks) < 0.04) >= 15
    assert abs(float(np.mean(midpoint_offsets))) < 0.02


def test_polyline_resampling_bounds_every_step():
    points = np.array(
        [[0.0, -0.2, 1.0], [0.12, 0.0, 1.0], [0.0, 0.2, 1.0]],
        dtype=np.float64,
    )
    route = MODULE.resample_polyline_max_step(points, 0.006)
    np.testing.assert_allclose(route[[0, -1]], points[[0, -1]])
    assert np.linalg.norm(np.diff(route, axis=0), axis=1).max() <= 0.006 + 1e-12

