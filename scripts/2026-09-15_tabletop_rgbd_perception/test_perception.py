from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from experiment_common import audit_absolute_eef_actions
from perception import (
    Detection,
    PerceptionHistory,
    PerceptionProvider,
    ProviderConfig,
    RgbdFrame,
    deterministic_farthest_point_sample,
    resolve_mask_overlaps,
    unproject_mask_to_world,
)
from runtime_io import replace_policy_pointcloud


def rgbd(shape=(8, 9)) -> RgbdFrame:
    height, width = shape
    intrinsic = np.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return RgbdFrame(
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth_m=np.ones((height, width), dtype=np.float32),
        intrinsic=intrinsic,
        camera_pose=np.eye(4),
    )


class FakeDetector:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = []

    def detect(self, rgb, prompt, *, box_threshold, text_threshold):
        self.calls.append((prompt, box_threshold, text_threshold, rgb.shape))
        if prompt in self.missing:
            return []
        score = {"can": 0.9, "tray": 0.8, "milk": 0.7, "stand": 0.6}[prompt]
        return [Detection(np.array([0, 0, 4, 4], dtype=np.float32), score, prompt)]


class FakeSegmenter:
    def __init__(self, masks):
        self.masks = masks

    def segment(self, rgb, boxes_by_role):
        return {role: self.masks[role].copy() for role in boxes_by_role}


class FakeTracker:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.initial = None

    def initialize(self, rgb, label_map):
        self.initial = label_map.copy()

    def track(self, rgb):
        return self.outputs.pop(0).copy()


@dataclass
class FakeTrackerFactory:
    outputs: list[np.ndarray]
    created: FakeTracker | None = None

    def create(self):
        self.created = FakeTracker(self.outputs)
        return self.created


def role_masks(shape=(8, 9)):
    masks = {role: np.zeros(shape, dtype=bool) for role in ("target", "goal", "milk", "support")}
    masks["target"][1:3, 1:3] = True
    masks["goal"][4:6, 1:3] = True
    masks["milk"][1:3, 5:7] = True
    masks["support"][3:5, 5:7] = True
    return masks


def make_provider(*, obstacle=True, missing=(), tracker_outputs=None):
    masks = role_masks()
    prompts = {"target": "can", "goal": "tray"}
    if obstacle:
        prompts.update({"milk": "milk", "support": "stand"})
    factory = FakeTrackerFactory(tracker_outputs or [])
    provider = PerceptionProvider(
        detector=FakeDetector(missing),
        segmenter=FakeSegmenter(masks),
        tracker_factory=factory,
        config=ProviderConfig(prompts=prompts, scene_has_obstacle=obstacle),
    )
    return provider, factory


def test_unproject_metric_depth_to_world():
    frame = rgbd((3, 3))
    mask = np.zeros((3, 3), dtype=bool)
    mask[2, 2] = True
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    points = unproject_mask_to_world(frame.depth_m, mask, frame.intrinsic, pose)
    np.testing.assert_allclose(points, [[2.0, 3.0, 4.0]])


def test_fps_and_repeat_padding_are_deterministic():
    small = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    padded = deterministic_farthest_point_sample(small, 5)
    np.testing.assert_array_equal(padded, [small[0], small[1], small[0], small[1], small[0]])
    line = np.stack([np.arange(20), np.zeros(20), np.zeros(20)], axis=-1).astype(np.float32)
    first = deterministic_farthest_point_sample(line, 6)
    second = deterministic_farthest_point_sample(line, 6)
    np.testing.assert_array_equal(first, second)


def test_mask_overlap_uses_score_then_stable_role_id():
    target = np.array([[1, 1], [0, 0]], dtype=bool)
    goal = np.array([[1, 0], [0, 1]], dtype=bool)
    resolved = resolve_mask_overlaps(
        {"target": target, "goal": goal}, {"target": 0.9, "goal": 0.8}
    )
    assert resolved["target"][0, 0]
    assert not resolved["goal"][0, 0]
    assert not np.any(resolved["target"] & resolved["goal"])


def test_composite_obstacle_and_optional_support():
    provider, factory = make_provider(obstacle=True)
    result = provider.start_episode(rgbd())
    assert result.valid
    assert result.diagnostics["support_detected"]
    np.testing.assert_array_equal(
        result.masks["obstacle"], result.masks["milk"] | result.masks["support"]
    )
    assert len(result.obstacle_points) == int(result.masks["obstacle"].sum())
    assert set(np.unique(factory.created.initial)) == {0, 1, 2, 3, 4}

    missing_support, _ = make_provider(obstacle=True, missing={"stand"})
    milk_only = missing_support.start_episode(rgbd())
    assert milk_only.valid
    assert not milk_only.diagnostics["support_detected"]
    np.testing.assert_array_equal(milk_only.masks["obstacle"], milk_only.masks["milk"])


def test_missing_milk_is_nonfatal_to_task_cloud_and_reported_separately():
    missing_milk, _ = make_provider(obstacle=True, missing={"milk"})
    initial = missing_milk.start_episode(rgbd())
    assert initial.valid
    assert not initial.diagnostics["obstacle_valid"]
    assert initial.diagnostics["obstacle_failure_reason"] == "invalid_mask_or_depth:milk"
    assert len(initial.obstacle_points) == int(initial.masks["support"].sum())

    masks = role_masks()
    labels = np.zeros((8, 9), dtype=np.uint8)
    labels[masks["target"]] = 1
    labels[masks["goal"]] = 2
    provider, _ = make_provider(obstacle=True, tracker_outputs=[labels])
    assert provider.start_episode(rgbd()).valid
    tracked = provider.track(rgbd())
    assert tracked.valid
    assert not tracked.diagnostics["obstacle_valid"]
    assert tracked.diagnostics["failure_reason"] is None


def test_two_frame_history_and_observation_replacement():
    history = PerceptionHistory()
    first = np.zeros((256, 3), dtype=np.float32)
    second = np.ones((256, 3), dtype=np.float32)
    history.reset(first)
    np.testing.assert_array_equal(history.stack(), np.stack([first, first]))
    history.append(second)
    np.testing.assert_array_equal(history.stack(), np.stack([first, second]))

    provider, _ = make_provider(obstacle=False)
    provider.start_episode(rgbd())
    observation = {
        "agent_pos": np.zeros((2, 7), dtype=np.float32),
        "task_pointcloud": np.full((2, 256, 3), 99, dtype=np.float32),
    }
    replaced = replace_policy_pointcloud(observation, provider)
    assert np.array_equal(replaced["agent_pos"], observation["agent_pos"])
    assert not np.array_equal(replaced["task_pointcloud"], observation["task_pointcloud"])
    assert np.array_equal(observation["task_pointcloud"], np.full((2, 256, 3), 99))
    without_oracle = {"agent_pos": observation["agent_pos"]}
    injected = replace_policy_pointcloud(without_oracle, provider)
    assert injected["task_pointcloud"].shape == (2, 256, 3)


def test_invalid_initial_and_tracking_masks_fail_without_detector_retry():
    missing_target, _ = make_provider(obstacle=False, missing={"can"})
    initial = missing_target.start_episode(rgbd())
    assert not initial.valid
    assert initial.diagnostics["failure_reason"] == "initial_detection_missing:target"
    assert [call[0] for call in missing_target.detector.calls] == ["can", "tray"]

    masks = role_masks()
    labels = np.zeros((8, 9), dtype=np.uint8)
    labels[masks["goal"]] = 2
    provider, _ = make_provider(obstacle=False, tracker_outputs=[labels])
    assert provider.start_episode(rgbd()).valid
    tracked = provider.track(rgbd())
    assert not tracked.valid
    assert tracked.diagnostics["failure_reason"] == "invalid_mask_or_depth:target"
    assert [call[0] for call in provider.detector.calls] == ["can", "tray"]


def test_provider_api_cannot_receive_oracle_data(monkeypatch):
    import oracle_audit

    def forbidden(*args, **kwargs):
        raise AssertionError("oracle audit was accessed by perception")

    monkeypatch.setattr(oracle_audit, "render_oracle_masks", forbidden)
    provider, _ = make_provider(obstacle=False)
    frame = rgbd()
    assert provider.start_episode(frame).valid
    with pytest.raises(TypeError):
        provider.start_episode(frame, oracle_masks={})


def test_absolute_eef_contract_uses_world_pose_not_normalized_action_bounds():
    valid = np.array(
        [[0.1, -0.2, 0.9, 2.2, 2.1, 0.1, 1.000015]], dtype=np.float32
    )
    audit = audit_absolute_eef_actions(valid)
    assert audit["valid"]
    assert audit["max_axis_angle_norm_rad"] > 3.0

    invalid = valid.copy()
    invalid[0, 3:6] = [np.pi + 0.01, 0.0, 0.0]
    invalid[0, 6] = 1.1
    audit = audit_absolute_eef_actions(invalid)
    assert not audit["valid"]
    assert audit["violation_count"] == 1
