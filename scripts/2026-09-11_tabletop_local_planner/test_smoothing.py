import numpy as np
from geometry import Ellipsoid, Geometry
from smoothing import compact_bezier, flatten_cubic, advance_path


def test_curve_is_shorter_and_all_segments_clear():
    g = Geometry(
        Ellipsoid([0, 0, 0.95], np.eye(3), [0.04] * 3),
        [Ellipsoid([0, 0, 0], np.eye(3), [0.02] * 3)],
    )
    route = g.bypass(
        np.array([-0.2, 0, 0.95]), np.eye(3), [0.3, 0], np.array([0.4, 0.4])
    )
    smooth = compact_bezier(g, route, np.eye(3))
    assert smooth["length"] < route["length"]
    assert len(smooth["bezier_controls"]) > 0
    assert np.array_equal(smooth["points"][[0, -1]], route["points"][[0, -1]])
    assert np.all(smooth["points"][:, 2] == 0.95)
    for a, b in zip(smooth["points"][:-1], smooth["points"][1:]):
        assert g.segment_clearance(a, np.eye(3), b, np.eye(3)) >= 0


def test_flatten_bounds_actual_curve():
    control = np.array([[0, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0], [0.1, 0, 0]])
    chords = flatten_cubic(control)
    assert len(chords) > 2
    for a, b, error in chords:
        assert error <= 0.0005
    for t in np.linspace(0, 1, 101):
        q = (
            (1 - t) ** 3 * control[0]
            + 3 * (1 - t) ** 2 * t * control[1]
            + 3 * (1 - t) * t * t * control[2]
            + t**3 * control[3]
        )
        distances = []
        for a, b, error in chords:
            d = b - a
            u = np.clip(np.dot(q - a, d) / np.dot(d, d), 0, 1)
            distances.append(np.linalg.norm(q - a - u * d))
        assert min(distances) <= 0.0005


def test_progress_is_monotone_and_targets_cross_dense_vertices():
    points = np.column_stack([np.linspace(0, 0.1, 101), np.zeros(101), np.zeros(101)])
    progress, targets, total = advance_path(points, 0.02, np.array([0.019, 0, 0]), 8)
    assert progress == 0.02 and abs(total - 0.1) < 1e-12
    assert np.allclose(np.diff(targets[:, 0]), 0.006)
    assert abs(targets[-1, 0] - 0.068) < 1e-12
