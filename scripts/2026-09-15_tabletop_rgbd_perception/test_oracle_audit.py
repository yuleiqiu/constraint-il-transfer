import numpy as np

from oracle_audit import audit_perception_frame, binary_mask_metrics, pointcloud_metrics
from perception import PerceptionFrame, RgbdFrame


def test_mask_and_point_metrics():
    predicted = np.array([[1, 1], [0, 0]], dtype=bool)
    oracle = np.array([[1, 0], [1, 0]], dtype=bool)
    metrics = binary_mask_metrics(predicted, oracle)
    assert metrics["intersection"] == 1
    assert metrics["iou"] == 1 / 3
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5

    points = np.array([[0, 0, 1], [0.001, 0, 1]], dtype=np.float32)
    cloud = pointcloud_metrics(points, points.copy())
    assert cloud["nn_5mm_precision"] == 1.0
    assert cloud["nn_10mm_recall"] == 1.0
    assert cloud["chamfer_m"] == 0.0


def test_audit_does_not_modify_policy_frame():
    rgbd = RgbdFrame(
        rgb=np.zeros((3, 3, 3), dtype=np.uint8),
        depth_m=np.ones((3, 3), dtype=np.float32),
        intrinsic=np.eye(3),
        camera_pose=np.eye(4),
    )
    target = np.zeros((3, 3), dtype=bool)
    target[0, 0] = True
    goal = np.zeros((3, 3), dtype=bool)
    goal[2, 2] = True
    empty = np.zeros((3, 3), dtype=bool)
    task_points = np.zeros((256, 3), dtype=np.float32)
    perceived = PerceptionFrame(
        task_points=task_points.copy(),
        target_points=np.array([[0, 0, 1]], dtype=np.float32),
        goal_points=np.array([[2, 2, 1]], dtype=np.float32),
        obstacle_points=np.empty((0, 3), dtype=np.float32),
        masks={"target": target.copy(), "goal": goal.copy(), "obstacle": empty.copy()},
        valid=True,
        diagnostics={"support_detected": False},
    )
    before = perceived.task_points.copy()
    audit = audit_perception_frame(
        perceived,
        rgbd,
        {"target": target, "goal": goal, "obstacle": empty},
    )
    assert audit.diagnostics["mask_metrics"]["target"]["iou"] == 1.0
    np.testing.assert_array_equal(perceived.task_points, before)
