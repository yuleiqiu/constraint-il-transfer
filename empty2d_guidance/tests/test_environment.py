import numpy as np

from empty2d_guidance.environment import (
    GOAL,
    START,
    EnvEmpty2D,
    EnvEmpty2DExtraSphere,
    EnvEmpty2DExtraSquare,
)


def test_environment_contract_and_absolute_step_clip():
    environment = EnvEmpty2D()
    observation = environment.reset()
    np.testing.assert_array_equal(observation["agent_pos"], START.astype(np.float32))
    observation, _, done, info = environment.step(GOAL)
    np.testing.assert_allclose(
        np.linalg.norm(observation["agent_pos"] - START), 0.03, atol=1e-7
    )
    assert not done and info["action_clipped"]
    state = environment.get_state()
    environment.step(np.zeros(2))
    environment.reset_to(state)
    np.testing.assert_allclose(environment.position, state["position"])


def test_swept_collision_detects_segment_crossing_without_endpoint_penetration():
    circle = EnvEmpty2DExtraSphere(robot_radius=0.02)
    assert circle.segment_collides(np.asarray([-0.5, 0.0]), np.asarray([0.5, 0.0]))
    assert not circle.segment_collides(np.asarray([-0.5, 0.5]), np.asarray([0.5, 0.5]))
    square = EnvEmpty2DExtraSquare(robot_radius=0.02)
    assert square.segment_collides(np.asarray([-0.5, 0.31]), np.asarray([0.5, 0.31]))
    assert not square.segment_collides(np.asarray([-0.5, 0.33]), np.asarray([0.5, 0.33]))


def test_swept_clearance_uses_closest_interior_point_of_segment():
    circle = EnvEmpty2DExtraSphere(robot_radius=0.02)
    start = np.asarray([-0.5, 0.4])
    end = np.asarray([0.5, 0.4])
    np.testing.assert_allclose(circle.segment_clearance(start, end), 0.08)
    assert circle.segment_clearance(np.asarray([-0.5, 0.31]), np.asarray([0.5, 0.31])) == 0.0

    square = EnvEmpty2DExtraSquare(robot_radius=0.02)
    np.testing.assert_allclose(square.segment_clearance(start, end), 0.08)
    assert square.segment_clearance(np.asarray([-0.5, 0.31]), np.asarray([0.5, 0.31])) == 0.0


def test_oracle_cloud_is_uninflated_true_boundary():
    circle = EnvEmpty2DExtraSphere()
    circle_points = circle.oracle_obstacle_points()
    assert circle_points.shape == (256, 2)
    np.testing.assert_allclose(np.linalg.norm(circle_points, axis=1), 0.3, atol=1e-6)
    square = EnvEmpty2DExtraSquare()
    square_points = square.oracle_obstacle_points()
    assert square_points.shape == (256, 2)
    assert np.all(np.isclose(np.max(np.abs(square_points), axis=1), 0.3))


def test_shifted_obstacle_translates_collision_clearance_and_oracle_cloud():
    center = np.asarray([0.2, -0.15])
    circle = EnvEmpty2DExtraSphere(obstacle_center=center, robot_radius=0.02)
    assert circle.segment_collides(
        np.asarray([-0.2, -0.15]), np.asarray([0.6, -0.15])
    )
    assert not circle.segment_collides(
        np.asarray([-0.2, 0.3]), np.asarray([0.6, 0.3])
    )
    np.testing.assert_allclose(
        np.linalg.norm(circle.oracle_obstacle_points() - center, axis=1),
        0.3,
        atol=1e-6,
    )

    square = EnvEmpty2DExtraSquare(obstacle_center=center, robot_radius=0.02)
    assert square.segment_collides(
        np.asarray([-0.2, -0.15]), np.asarray([0.6, -0.15])
    )
    assert not square.segment_collides(
        np.asarray([-0.2, 0.3]), np.asarray([0.6, 0.3])
    )
    relative_points = square.oracle_obstacle_points() - center
    assert np.all(np.isclose(np.max(np.abs(relative_points), axis=1), 0.3))
