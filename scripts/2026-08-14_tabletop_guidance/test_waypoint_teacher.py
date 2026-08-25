"""Pure checks for the direct tabletop waypoint teacher."""

from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import waypoint_teacher as MODULE  # noqa: E402


def test_minimum_jerk_reaches_endpoint_and_stays_on_segment():
    start = np.array([-0.1, 0.0, 1.0])
    goal = np.array([0.05, 0.3, 1.0])
    route = MODULE.minimum_jerk_positions(start, goal, max_step_m=0.006)
    np.testing.assert_allclose(route[-1], goal)
    assert np.max(MODULE.point_to_segment_distance(route, start, goal)) < 1e-12
    progress = (route - start) @ (goal - start)
    assert np.all(np.diff(progress) >= 0.0)


def test_controller_uses_absolute_world_frame():
    config = MODULE.absolute_osc_controller_config(kp=500)
    arm = config["body_parts"]["right"]
    assert arm["type"] == "OSC_POSE"
    assert arm["input_type"] == "absolute"
    assert arm["input_ref_frame"] == "world"
    assert arm["kp"] == 500


def test_point_to_segment_distance():
    points = np.array([[0.5, 0.1], [0.5, -0.2], [-0.2, 0.0], [1.2, 0.0]])
    distance = MODULE.point_to_segment_distance(points, [0.0, 0.0], [1.0, 0.0])
    np.testing.assert_allclose(distance, [0.1, 0.2, 0.2, 0.2])
