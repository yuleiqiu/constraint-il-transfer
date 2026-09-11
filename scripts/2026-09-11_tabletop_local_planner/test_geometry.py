import numpy as np
from scipy.spatial.transform import Rotation
from geometry import Ellipsoid, Geometry


def sphere(center, radius):
    return Ellipsoid(np.array(center), np.eye(3), np.full(3, radius))


def test_segment_interior_collision_is_rejected():
    g = Geometry(sphere([0, 0, 0], 0.04), [sphere([0, 0, 0], 0.02)], margin=0.01)
    assert g.segment_clearance([-0.2, 0, 0], np.eye(3), [-0.19, 0, 0], np.eye(3)) > 0
    assert g.segment_clearance([-0.2, 0, 0], np.eye(3), [0.2, 0, 0], np.eye(3)) < 0


def test_rotating_offset_actor_sweeps_into_obstacle():
    g = Geometry(sphere([0.1, 0, 0], 0.02), [sphere([0.1, 0, 0], 0.01)], margin=0)
    left = Rotation.from_euler("z", -90, degrees=True).as_matrix()
    right = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    assert g.segment_clearance([0, 0, 0], left, [0, 0, 0], left) > 0
    assert g.segment_clearance([0, 0, 0], right, [0, 0, 0], right) > 0
    assert g.segment_clearance([0, 0, 0], left, [0, 0, 0], right) < 0


def test_bypass_has_safe_edges_and_forward_endpoint():
    g = Geometry(sphere([0, 0, 0.95], 0.04), [sphere([0, 0, 0], 0.02)], margin=0.01)
    result = g.bypass(
        np.array([-0.2, 0, 0.95]), np.eye(3), [0.3, 0], np.array([0.4, 0.4])
    )
    assert result is not None and len(result["points"]) > 2
    assert result["points"][-1, 0] > 0.04
    for a, b in zip(result["points"][:-1], result["points"][1:]):
        assert g.segment_clearance(a, np.eye(3), b, np.eye(3)) >= 0


def test_workspace_blocks_bypass():
    g = Geometry(sphere([0, 0, 0.95], 0.1), [sphere([0, 0, 0], 0.02)], margin=0.01)
    assert (
        g.bypass(np.array([-0.2, 0, 0.95]), np.eye(3), [0.3, 0], np.array([0.4, 0.05]))
        is None
    )


def test_overflight_not_falsely_blocked_in_xy():
    g = Geometry(sphere([0, 0, 0.8], 0.04), [sphere([0, 0, 0], 0.02)])
    assert g.segment_clearance([-0.2, 0, 1.0], np.eye(3), [0.2, 0, 1.0], np.eye(3)) > 0


def test_zero_motion_matches_existing_mvee_clearance():
    import torch
    from robomimic.utils.ellipsoid_guidance_utils import directional_ellipsoid_clearance

    actor = Ellipsoid(
        [0.01, 0.02, -0.03],
        Rotation.from_euler("z", 0.3).as_matrix(),
        [0.02, 0.04, 0.03],
    )
    obstacle = Ellipsoid(
        [0.1, -0.1, 0.9], Rotation.from_euler("y", 0.5).as_matrix(), [0.03, 0.06, 0.05]
    )
    g = Geometry(obstacle, [actor])
    p, r = np.array([-0.2, 0.1, 0.95]), Rotation.from_euler("x", 0.4).as_matrix()
    world = actor.at(p, r)
    expected = directional_ellipsoid_clearance(
        torch.tensor(world.center),
        torch.tensor(world.rotation),
        torch.tensor(world.axes),
        obstacle.center,
        obstacle.rotation,
        obstacle.axes,
        clearance_margin=0.01,
    ).item()
    assert abs(g.segment_clearance(p, r, p, r) - expected) < 1e-12
