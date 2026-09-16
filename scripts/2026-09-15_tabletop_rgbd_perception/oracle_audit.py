"""Evaluation-only oracle masks and perception metrics.

This is the only experiment module permitted to request MuJoCo segmentation.
Callers must invoke the perception provider first, then pass its immutable
result here.  Nothing returned by this module is accepted by the provider or
inserted into the policy observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from perception import OBSTACLE_ROLES, PerceptionFrame, RgbdFrame, unproject_mask_to_world


ROLE_PREFIXES = {
    "target": ("Can_",),
    "goal": ("GoalTray_",),
    "milk": ("Milk_",),
    "support": ("MilkSupport_",),
}


@dataclass(frozen=True)
class OracleAudit:
    masks: Mapping[str, np.ndarray]
    points: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, Any]


def geom_ids_by_role(raw_env) -> dict[str, tuple[int, ...]]:
    """Resolve evaluation labels without exposing them to perception code."""

    result: dict[str, list[int]] = {role: [] for role in ROLE_PREFIXES}
    for geom_id in range(raw_env.sim.model.ngeom):
        name = raw_env.sim.model.geom_id2name(geom_id) or ""
        for role, prefixes in ROLE_PREFIXES.items():
            if any(name.startswith(prefix) for prefix in prefixes):
                result[role].append(geom_id)
    if not result["target"] or not result["goal"]:
        raise RuntimeError("oracle audit could not find target and goal geoms")
    return {role: tuple(ids) for role, ids in result.items()}


def render_oracle_masks(
    raw_env,
    *,
    camera_name: str,
    height: int,
    width: int,
    role_geom_ids: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, np.ndarray]:
    """Render top-left-origin oracle masks through the isolated audit path."""

    if role_geom_ids is None:
        role_geom_ids = geom_ids_by_role(raw_env)
    segmentation = raw_env.sim.render(
        camera_name=camera_name,
        height=height,
        width=width,
        segmentation=True,
    )[::-1]
    geom_ids = np.asarray(segmentation)[..., 1]
    masks = {
        role: np.isin(geom_ids, np.asarray(ids, dtype=np.int64))
        for role, ids in role_geom_ids.items()
        if ids
    }
    masks["obstacle"] = np.logical_or.reduce(
        [masks.get(role, np.zeros((height, width), dtype=bool)) for role in OBSTACLE_ROLES]
    )
    return masks


def binary_mask_metrics(predicted: np.ndarray, oracle: np.ndarray) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=bool)
    oracle = np.asarray(oracle, dtype=bool)
    if predicted.shape != oracle.shape:
        raise ValueError("predicted and oracle mask shapes differ")
    intersection = int(np.count_nonzero(predicted & oracle))
    predicted_count = int(predicted.sum())
    oracle_count = int(oracle.sum())
    union = predicted_count + oracle_count - intersection
    return {
        "intersection": intersection,
        "predicted_pixels": predicted_count,
        "oracle_pixels": oracle_count,
        "iou": float(intersection / union) if union else 1.0,
        "precision": float(intersection / predicted_count) if predicted_count else 0.0,
        "recall": float(intersection / oracle_count) if oracle_count else 0.0,
    }


def pointcloud_metrics(predicted: np.ndarray, oracle: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float32)
    oracle = np.asarray(oracle, dtype=np.float32)
    result: dict[str, Any] = {
        "predicted_points": len(predicted),
        "oracle_points": len(oracle),
    }
    if not len(predicted) or not len(oracle):
        result.update(
            {
                "nn_5mm_precision": 0.0,
                "nn_5mm_recall": 0.0,
                "nn_10mm_precision": 0.0,
                "nn_10mm_recall": 0.0,
                "chamfer_m": None,
                "centroid_error_m": None,
                "extents_error_m": None,
            }
        )
        return result
    oracle_tree = cKDTree(oracle.astype(np.float64))
    predicted_tree = cKDTree(predicted.astype(np.float64))
    predicted_to_oracle = oracle_tree.query(predicted, workers=1)[0]
    oracle_to_predicted = predicted_tree.query(oracle, workers=1)[0]
    for millimeters in (5, 10):
        threshold = millimeters / 1000.0
        result[f"nn_{millimeters}mm_precision"] = float(
            np.mean(predicted_to_oracle <= threshold)
        )
        result[f"nn_{millimeters}mm_recall"] = float(
            np.mean(oracle_to_predicted <= threshold)
        )
    result["chamfer_m"] = float(
        np.mean(predicted_to_oracle) + np.mean(oracle_to_predicted)
    )
    result["centroid_error_m"] = float(
        np.linalg.norm(predicted.mean(axis=0) - oracle.mean(axis=0))
    )
    predicted_extents = predicted.max(axis=0) - predicted.min(axis=0)
    oracle_extents = oracle.max(axis=0) - oracle.min(axis=0)
    result["extents_error_m"] = float(np.linalg.norm(predicted_extents - oracle_extents))
    return result


def audit_perception_frame(
    perceived: PerceptionFrame,
    rgbd: RgbdFrame,
    oracle_masks: Mapping[str, np.ndarray],
) -> OracleAudit:
    """Score masks/clouds after perception output has already been finalized."""

    shape = rgbd.depth_m.shape
    roles = tuple(role for role in ("target", "goal", "milk", "support", "obstacle") if role in oracle_masks)
    oracle_points = {
        role: unproject_mask_to_world(
            rgbd.depth_m,
            oracle_masks[role],
            rgbd.intrinsic,
            rgbd.camera_pose,
        )
        for role in roles
    }
    perceived_points = {
        "target": perceived.target_points,
        "goal": perceived.goal_points,
        "obstacle": perceived.obstacle_points,
    }
    for role in ("milk", "support"):
        if role in perceived.masks:
            perceived_points[role] = unproject_mask_to_world(
                rgbd.depth_m,
                perceived.masks[role],
                rgbd.intrinsic,
                rgbd.camera_pose,
            )
    mask_scores = {}
    cloud_scores = {}
    identity_swaps = {}
    for role in roles:
        predicted_mask = np.asarray(
            perceived.masks.get(role, np.zeros(shape, dtype=bool)), dtype=bool
        )
        mask_scores[role] = binary_mask_metrics(predicted_mask, oracle_masks[role])
        if role in perceived_points:
            cloud_scores[role] = pointcloud_metrics(
                perceived_points[role], oracle_points[role]
            )
        if role != "obstacle" and predicted_mask.any():
            own_iou = mask_scores[role]["iou"]
            other_ious = {
                other: binary_mask_metrics(predicted_mask, oracle_masks[other])["iou"]
                for other in roles
                if other not in (role, "obstacle")
            }
            best_other = max(other_ious, key=other_ious.get) if other_ious else None
            identity_swaps[role] = bool(
                best_other is not None and other_ious[best_other] > own_iou
            )
        else:
            identity_swaps[role] = False
    diagnostics = {
        "mask_metrics": mask_scores,
        "pointcloud_metrics": cloud_scores,
        "identity_swap": identity_swaps,
        "identity_swap_any": any(identity_swaps.values()),
        "support_oracle_visible": bool(
            np.asarray(oracle_masks.get("support", np.zeros(shape, dtype=bool))).any()
        ),
        "support_detected": bool(perceived.diagnostics.get("support_detected", False)),
    }
    return OracleAudit(masks=oracle_masks, points=oracle_points, diagnostics=diagnostics)
