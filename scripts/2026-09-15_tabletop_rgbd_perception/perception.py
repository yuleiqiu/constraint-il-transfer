"""Oracle-free RGB-D perception primitives for the Tabletop comparison.

The provider in this module deliberately has no simulator dependency.  Its
runtime inputs are an RGB image, metric depth, fixed camera calibration, and
the configured language prompts.  Simulator segmentation belongs in
``oracle_audit.py`` and must never be passed to :class:`PerceptionProvider`.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


ROLE_IDS = {"target": 1, "goal": 2, "milk": 3, "support": 4}
TASK_ROLES = ("target", "goal")
OBSTACLE_ROLES = ("milk", "support")


@dataclass(frozen=True)
class Detection:
    """One detector result in top-left-origin XYXY pixel coordinates."""

    box_xyxy: np.ndarray
    score: float
    phrase: str


@dataclass(frozen=True)
class RgbdFrame:
    """The complete deployment input allowed for one perception frame."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic: np.ndarray
    camera_pose: np.ndarray

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        intrinsic = np.asarray(self.intrinsic)
        camera_pose = np.asarray(self.camera_pose)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8 with shape (H, W, 3)")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth_m must match the RGB height and width")
        if intrinsic.shape != (3, 3) or camera_pose.shape != (4, 4):
            raise ValueError("intrinsic/camera_pose must have shapes (3,3)/(4,4)")
        if not np.isfinite(intrinsic).all() or not np.isfinite(camera_pose).all():
            raise ValueError("camera calibration must be finite")


@dataclass(frozen=True)
class PerceptionFrame:
    """Stable policy-facing result of one RGB-D perception update.

    ``valid`` means the target+goal task cloud is usable by the frozen policy.
    Obstacle reconstruction has its own ``obstacle_valid`` diagnostic because
    it is saved for audit / a later planner but is not a policy input here.
    """

    task_points: np.ndarray
    target_points: np.ndarray
    goal_points: np.ndarray
    obstacle_points: np.ndarray
    masks: Mapping[str, np.ndarray]
    valid: bool
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderConfig:
    """Locked prompts and validity rules for one environment."""

    prompts: Mapping[str, str]
    scene_has_obstacle: bool
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    num_task_points: int = 256
    padding_mode: str = "repeat"
    min_mask_pixels: int = 4
    min_depth_points: int = 4

    def __post_init__(self) -> None:
        required = set(TASK_ROLES)
        if self.scene_has_obstacle:
            required.add("milk")
        missing = required.difference(self.prompts)
        if missing:
            raise ValueError(f"missing required prompts: {sorted(missing)}")
        for role, prompt in self.prompts.items():
            if role not in ROLE_IDS:
                raise ValueError(f"unsupported role: {role}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"prompt for {role} must be non-empty")
        if self.num_task_points != 256:
            raise ValueError("the frozen checkpoint requires exactly 256 task points")
        if self.padding_mode not in ("repeat", "zero"):
            raise ValueError("padding_mode must be repeat or zero")
        if self.min_mask_pixels <= 0 or self.min_depth_points <= 0:
            raise ValueError("minimum point and pixel counts must be positive")
        if self.box_threshold != 0.35 or self.text_threshold != 0.25:
            raise ValueError("this stage locks DINO thresholds to 0.35/0.25")


class Detector(Protocol):
    def detect(
        self,
        rgb: np.ndarray,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> Sequence[Detection]: ...


class Segmenter(Protocol):
    def segment(
        self,
        rgb: np.ndarray,
        boxes_by_role: Mapping[str, np.ndarray],
    ) -> Mapping[str, np.ndarray]: ...


class Tracker(Protocol):
    def initialize(self, rgb: np.ndarray, label_map: np.ndarray) -> None: ...

    def track(self, rgb: np.ndarray) -> np.ndarray: ...


class TrackerFactory(Protocol):
    def create(self) -> Tracker: ...


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str((value.shape, value.dtype)).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def unproject_mask_to_world(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    """Unproject a top-left-origin mask and metric depth into world XYZ."""

    depth_m = np.asarray(depth_m)
    mask = np.asarray(mask, dtype=bool)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    camera_pose = np.asarray(camera_pose, dtype=np.float64)
    if depth_m.ndim != 2 or mask.shape != depth_m.shape:
        raise ValueError("mask and depth_m must have the same 2-D shape")
    if intrinsic.shape != (3, 3) or camera_pose.shape != (4, 4):
        raise ValueError("invalid calibration shapes")
    pixels = np.argwhere(mask)
    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.float32)
    rows = pixels[:, 0]
    cols = pixels[:, 1]
    z = depth_m[rows, cols].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.0)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float32)
    x = (cols.astype(np.float64) - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (rows.astype(np.float64) - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
    world = camera_points @ camera_pose.T
    points = world[:, :3].astype(np.float32)
    return points[np.isfinite(points).all(axis=1)]


def deterministic_farthest_point_sample(
    points: np.ndarray,
    num_points: int,
    *,
    padding_mode: str = "repeat",
) -> np.ndarray:
    """Use the checkpoint's deterministic centroid-seeded FPS contract."""

    points = np.asarray(points, dtype=np.float32)
    num_points = int(num_points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if len(points) == 0:
        raise ValueError("cannot sample an empty cloud")
    if padding_mode not in ("repeat", "zero"):
        raise ValueError("padding_mode must be repeat or zero")
    if len(points) < num_points:
        if padding_mode == "zero":
            padding = np.zeros((num_points - len(points), 3), dtype=np.float32)
            return np.concatenate((points, padding), axis=0)
        repeats = (num_points + len(points) - 1) // len(points)
        return np.tile(points, (repeats, 1))[:num_points].copy()
    if len(points) == num_points:
        return points.copy()

    centroid = points.astype(np.float64).mean(axis=0)
    selected = np.empty(num_points, dtype=np.int64)
    selected[0] = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    min_sq_distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for index in range(1, num_points):
        selected[index] = int(np.argmax(min_sq_distance))
        candidate = np.sum((points - points[selected[index]]) ** 2, axis=1)
        np.minimum(min_sq_distance, candidate, out=min_sq_distance)
    return points[selected].copy()


def resolve_mask_overlaps(
    masks: Mapping[str, np.ndarray],
    scores: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Assign overlapping pixels to the highest-confidence role deterministically."""

    if not masks:
        return {}
    shapes = {np.asarray(mask).shape for mask in masks.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("all role masks must share one 2-D shape")
    if set(masks).difference(ROLE_IDS):
        raise ValueError("mask contains an unknown role")
    shape = next(iter(shapes))
    occupied = np.zeros(shape, dtype=bool)
    resolved: dict[str, np.ndarray] = {}
    ordered = sorted(
        masks,
        key=lambda role: (-float(scores.get(role, 0.0)), ROLE_IDS[role]),
    )
    for role in ordered:
        current = np.asarray(masks[role], dtype=bool) & ~occupied
        resolved[role] = current
        occupied |= current
    return {role: resolved[role] for role in sorted(resolved, key=ROLE_IDS.get)}


def masks_to_label_map(masks: Mapping[str, np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("cannot create a label map without masks")
    shapes = {np.asarray(mask).shape for mask in masks.values()}
    if len(shapes) != 1:
        raise ValueError("all masks must share one shape")
    labels = np.zeros(next(iter(shapes)), dtype=np.uint8)
    for role in sorted(masks, key=ROLE_IDS.get):
        mask = np.asarray(masks[role], dtype=bool)
        if np.any(mask & (labels != 0)):
            raise ValueError("role masks must be disjoint before tracking")
        labels[mask] = ROLE_IDS[role]
    return labels


@dataclass
class PerceptionHistory:
    """Two-frame history matching the frozen checkpoint's frame stack."""

    length: int = 2
    _frames: list[np.ndarray] = field(default_factory=list)

    def reset(self, task_points: np.ndarray) -> None:
        value = self._validate(task_points)
        self._frames = [value.copy() for _ in range(self.length)]

    def append(self, task_points: np.ndarray) -> None:
        value = self._validate(task_points)
        if not self._frames:
            self.reset(value)
            return
        self._frames.append(value.copy())
        self._frames = self._frames[-self.length :]

    def stack(self) -> np.ndarray:
        if len(self._frames) != self.length:
            raise RuntimeError("perception history has not been initialized")
        return np.stack(self._frames).astype(np.float32, copy=False)

    @staticmethod
    def _validate(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (256, 3) or not np.isfinite(value).all():
            raise ValueError("task_points must be finite float32[256,3]")
        return value


class PerceptionProvider:
    """Ground once with DINO+SAM and then propagate fixed IDs with Cutie."""

    def __init__(
        self,
        *,
        detector: Detector,
        segmenter: Segmenter,
        tracker_factory: TrackerFactory,
        config: ProviderConfig,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.tracker_factory = tracker_factory
        self.config = config
        self.history = PerceptionHistory(length=2)
        self.tracker: Tracker | None = None
        self.frame_index = -1
        self.active_roles: tuple[str, ...] = ()
        self.support_detected = False

    def start_episode(self, frame: RgbdFrame) -> PerceptionFrame:
        """Initialize every instance once; no detector retry is permitted."""

        self.tracker = None
        self.frame_index = -1
        self.active_roles = ()
        self.support_detected = False
        started = time.perf_counter()
        detections: dict[str, Detection] = {}
        detector_latency: dict[str, float] = {}
        roles = list(TASK_ROLES)
        if self.config.scene_has_obstacle:
            roles.extend(OBSTACLE_ROLES)
        for role in roles:
            if role not in self.config.prompts:
                continue
            before = time.perf_counter()
            candidates = list(
                self.detector.detect(
                    frame.rgb,
                    self.config.prompts[role],
                    box_threshold=self.config.box_threshold,
                    text_threshold=self.config.text_threshold,
                )
            )
            detector_latency[role] = time.perf_counter() - before
            if candidates:
                detections[role] = max(
                    candidates,
                    key=lambda item: (float(item.score), -float(np.asarray(item.box_xyxy)[0])),
                )

        missing = sorted(set(TASK_ROLES).difference(detections))
        self.support_detected = "support" in detections
        if missing:
            result = self._invalid(
                frame,
                reason="initial_detection_missing:" + ",".join(missing),
                extra={
                    "phase": "initialization",
                "detected_roles": sorted(detections, key=ROLE_IDS.get),
                    "milk_detected": "milk" in detections,
                    "support_detected": self.support_detected,
                    "detector_latency_s": detector_latency,
                    "total_latency_s": time.perf_counter() - started,
                },
            )
            self.history.reset(result.task_points)
            self.frame_index = 0
            return result

        boxes = {role: detection.box_xyxy for role, detection in detections.items()}
        before_segment = time.perf_counter()
        candidate_masks = self.segmenter.segment(frame.rgb, boxes)
        segment_latency = time.perf_counter() - before_segment
        scores = {role: detection.score for role, detection in detections.items()}
        masks = resolve_mask_overlaps(
            {role: mask for role, mask in candidate_masks.items() if role in boxes},
            scores,
        )
        self.active_roles = tuple(sorted(masks, key=ROLE_IDS.get))
        result = self._build_result(
            frame,
            masks,
            phase="initialization",
            extra={
                "prompts": {role: self.config.prompts[role] for role in roles},
                "detections": {
                    role: {
                        "box_xyxy": np.asarray(detection.box_xyxy).tolist(),
                        "score": float(detection.score),
                        "phrase": detection.phrase,
                    }
                    for role, detection in detections.items()
                },
                "detector_latency_s": detector_latency,
                "segmenter_latency_s": segment_latency,
                "tracker_latency_s": None,
                "support_detected": self.support_detected,
                "milk_detected": "milk" in detections,
            },
            started=started,
        )
        if result.valid:
            self.tracker = self.tracker_factory.create()
            before_tracker = time.perf_counter()
            self.tracker.initialize(frame.rgb, masks_to_label_map(masks))
            tracker_latency = time.perf_counter() - before_tracker
            diagnostics = dict(result.diagnostics)
            diagnostics["tracker_latency_s"] = tracker_latency
            diagnostics["total_latency_s"] = time.perf_counter() - started
            result = PerceptionFrame(
                task_points=result.task_points,
                target_points=result.target_points,
                goal_points=result.goal_points,
                obstacle_points=result.obstacle_points,
                masks=result.masks,
                valid=result.valid,
                diagnostics=diagnostics,
            )
        self.history.reset(result.task_points)
        self.frame_index = 0
        return result

    def track(self, frame: RgbdFrame) -> PerceptionFrame:
        if self.tracker is None or self.frame_index < 0:
            raise RuntimeError("start_episode must be called before track")
        started = time.perf_counter()
        before_tracker = time.perf_counter()
        label_map = np.asarray(self.tracker.track(frame.rgb))
        tracker_latency = time.perf_counter() - before_tracker
        if label_map.shape != frame.depth_m.shape:
            result = self._invalid(
                frame,
                reason="tracker_shape_mismatch",
                extra={"phase": "tracking", "tracker_latency_s": tracker_latency},
            )
        else:
            masks = {
                role: label_map == ROLE_IDS[role]
                for role in self.active_roles
            }
            result = self._build_result(
                frame,
                masks,
                phase="tracking",
                extra={
                    "tracker_latency_s": tracker_latency,
                    "support_detected": self.support_detected,
                },
                started=started,
            )
        self.frame_index += 1
        self.history.append(result.task_points)
        return result

    def policy_task_pointcloud(self) -> np.ndarray:
        return self.history.stack()

    def _build_result(
        self,
        frame: RgbdFrame,
        masks: Mapping[str, np.ndarray],
        *,
        phase: str,
        extra: Mapping[str, Any],
        started: float,
    ) -> PerceptionFrame:
        shape = frame.depth_m.shape
        normalized_masks: dict[str, np.ndarray] = {}
        for role in self.active_roles or tuple(sorted(masks, key=ROLE_IDS.get)):
            mask = np.asarray(masks.get(role, np.zeros(shape, dtype=bool)), dtype=bool)
            if mask.shape != shape:
                return self._invalid(
                    frame,
                    reason=f"mask_shape_mismatch:{role}",
                    extra={"phase": phase, **extra},
                )
            normalized_masks[role] = mask
        normalized_masks["obstacle"] = np.logical_or.reduce(
            [
                normalized_masks.get(role, np.zeros(shape, dtype=bool))
                for role in OBSTACLE_ROLES
            ]
        )

        points = {
            role: unproject_mask_to_world(
                frame.depth_m,
                mask,
                frame.intrinsic,
                frame.camera_pose,
            )
            for role, mask in normalized_masks.items()
        }
        invalid_roles = [
            role
            for role in TASK_ROLES
            if int(normalized_masks.get(role, np.zeros(shape, dtype=bool)).sum())
            < self.config.min_mask_pixels
            or len(points.get(role, ())) < self.config.min_depth_points
        ]
        obstacle_invalid_roles = [
            role
            for role in (("milk",) if self.config.scene_has_obstacle else ())
            if int(normalized_masks.get(role, np.zeros(shape, dtype=bool)).sum())
            < self.config.min_mask_pixels
            or len(points.get(role, ())) < self.config.min_depth_points
        ]
        task_raw = np.concatenate(
            [points.get(role, np.empty((0, 3), dtype=np.float32)) for role in TASK_ROLES],
            axis=0,
        )
        if len(task_raw):
            task_points = deterministic_farthest_point_sample(
                task_raw,
                self.config.num_task_points,
                padding_mode=self.config.padding_mode,
            )
        else:
            task_points = np.zeros((self.config.num_task_points, 3), dtype=np.float32)
        valid = not invalid_roles and task_points.shape == (256, 3)
        diagnostics = {
            "frame_index": self.frame_index + 1,
            "phase": phase,
            "failure_reason": (
                None if valid else "invalid_mask_or_depth:" + ",".join(invalid_roles)
            ),
            "obstacle_valid": not obstacle_invalid_roles,
            "obstacle_failure_reason": (
                None
                if not obstacle_invalid_roles
                else "invalid_mask_or_depth:" + ",".join(obstacle_invalid_roles)
            ),
            "mask_pixels": {
                role: int(mask.sum()) for role, mask in normalized_masks.items()
            },
            "valid_depth_points": {role: len(value) for role, value in points.items()},
            "rgb_sha256": array_sha256(frame.rgb),
            "depth_sha256": array_sha256(frame.depth_m),
            **extra,
            "total_latency_s": time.perf_counter() - started,
        }
        return PerceptionFrame(
            task_points=task_points.astype(np.float32, copy=False),
            target_points=points.get("target", np.empty((0, 3), dtype=np.float32)),
            goal_points=points.get("goal", np.empty((0, 3), dtype=np.float32)),
            obstacle_points=points["obstacle"],
            masks=normalized_masks,
            valid=bool(valid),
            diagnostics=diagnostics,
        )

    def _invalid(
        self,
        frame: RgbdFrame,
        *,
        reason: str,
        extra: Mapping[str, Any],
    ) -> PerceptionFrame:
        empty = np.empty((0, 3), dtype=np.float32)
        return PerceptionFrame(
            task_points=np.zeros((256, 3), dtype=np.float32),
            target_points=empty.copy(),
            goal_points=empty.copy(),
            obstacle_points=empty.copy(),
            masks={},
            valid=False,
            diagnostics={
                "frame_index": max(self.frame_index + 1, 0),
                "failure_reason": reason,
                "rgb_sha256": array_sha256(frame.rgb),
                "depth_sha256": array_sha256(frame.depth_m),
                **extra,
            },
        )
