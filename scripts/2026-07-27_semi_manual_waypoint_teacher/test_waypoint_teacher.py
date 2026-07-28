"""Pure geometry checks for the semi-manual waypoint teacher."""

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_waypoint_teacher.py")
SPEC = importlib.util.spec_from_file_location("evaluate_waypoint_teacher", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_interpolation_reaches_each_waypoint_with_bounded_deltas():
    start = np.array([0.0, 0.0, 1.0])
    waypoints = np.array(
        [
            [0.0, 0.0, 1.1],
            [0.12, -0.08, 1.1],
            [0.12, -0.08, 0.98],
        ]
    )
    deltas, segment_ends = MODULE.interpolate_position_route(
        start, waypoints, max_delta_m=0.006
    )
    positions = start + np.cumsum(deltas, axis=0)
    assert np.all(np.linalg.norm(deltas, axis=1) <= 0.0060001)
    for end, expected in zip(segment_ends, waypoints):
        assert np.allclose(positions[end - 1], expected, atol=1e-6)


def test_overhead_route_lifts_above_objects_and_hands_off_above_target():
    start = np.array([0.0, -0.2, 0.98])
    target = np.array([0.18, -0.34, 0.86])
    obstacles = np.array(
        [
            [0.1, -0.25, 0.84],
            [0.12, -0.30, 0.895],
        ]
    )
    route = MODULE.overhead_waypoints(
        start,
        target,
        obstacles,
        minimum_lift_m=0.06,
        object_overhead_m=0.18,
        handoff_height_m=0.12,
    )
    assert np.isclose(route[0, 2], 1.075)
    assert np.allclose(route[1, :2], target[:2])
    assert np.allclose(route[2], [target[0], target[1], 0.98])


def _candidate(yaw, lift, contact=True, collision=False, error=0.001):
    return {
        "relative_yaw_deg": yaw,
        "target_lift_at_handoff_m": lift,
        "teacher_target_contact_any": contact,
        "teacher_collision_any": collision,
        "handoff_error_m": error,
    }


def test_yaw_selection_uses_smallest_safe_orientation_change():
    candidates = [
        _candidate(0.0, 0.14, collision=True),
        _candidate(-20.0, 0.14),
        _candidate(20.0, 0.13),
        _candidate(-45.0, 0.15),
    ]
    selected = MODULE.select_yaw_candidate(
        candidates, minimum_target_lift_m=0.05
    )
    assert selected["relative_yaw_deg"] == -20.0


def test_yaw_selection_fallback_prefers_a_lifted_pick():
    candidates = [
        _candidate(0.0, 0.0, contact=False, collision=False),
        _candidate(20.0, 0.14, collision=True),
        _candidate(45.0, 0.0, contact=True, collision=False),
    ]
    selected = MODULE.select_yaw_candidate(
        candidates, minimum_target_lift_m=0.05
    )
    assert selected["relative_yaw_deg"] == 20.0
    assert not MODULE.candidate_is_safe_pick(selected, 0.05)


def test_yaw_candidates_are_grouped_by_absolute_change():
    groups = MODULE.yaw_candidate_groups(
        [45.0, 0.0, -20.0, 20.0, -45.0, 30.0]
    )
    assert groups == [[0.0], [-20.0, 20.0], [30.0], [45.0, -45.0]]


if __name__ == "__main__":
    test_interpolation_reaches_each_waypoint_with_bounded_deltas()
    test_overhead_route_lifts_above_objects_and_hands_off_above_target()
    test_yaw_selection_uses_smallest_safe_orientation_change()
    test_yaw_selection_fallback_prefers_a_lifted_pick()
    test_yaw_candidates_are_grouped_by_absolute_change()
    print("waypoint teacher tests passed")
