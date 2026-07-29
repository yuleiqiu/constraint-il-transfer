"""Pure decision logic for gated corrective-policy action selection."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SelectorConfig:
    trigger_clearance_m: float = 0.01
    minimum_start_clearance_m: float = 0.01
    candidate_clearance_m: float = 0.01
    progress_tolerance_m: float = 5e-4
    progress_rule: str = "nonretreat"

    def validate(self):
        if self.trigger_clearance_m < 0.0:
            raise ValueError("trigger_clearance_m must be non-negative")
        if self.minimum_start_clearance_m < 0.0:
            raise ValueError("minimum_start_clearance_m must be non-negative")
        if self.candidate_clearance_m < 0.0:
            raise ValueError("candidate_clearance_m must be non-negative")
        if self.progress_tolerance_m < 0.0:
            raise ValueError("progress_tolerance_m must be non-negative")
        if self.progress_rule not in ("nonretreat", "source_relative"):
            raise ValueError(
                "progress_rule must be 'nonretreat' or 'source_relative'"
            )

    def to_dict(self):
        return asdict(self)


def reconstruct_positions(start_position, actions):
    start = np.asarray(start_position, dtype=np.float64)
    chunk = np.asarray(actions, dtype=np.float64)
    if start.shape != (3,) or chunk.ndim != 2 or chunk.shape[1] < 3:
        raise ValueError("Expected start [3] and action chunk [H, D>=3]")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(chunk)):
        raise ValueError("Start and actions must be finite")
    return start[None] + np.cumsum(chunk[:, :3], axis=0)


def closest_target_progress(start_position, positions, target_position):
    start = np.asarray(start_position, dtype=np.float64)
    points = np.asarray(positions, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if start.shape != (3,) or target.shape != (3,) or points.ndim != 2:
        raise ValueError("Target progress expects start [3], points [H,3], target [3]")
    initial_distance = np.linalg.norm(start - target)
    closest_distance = np.min(np.linalg.norm(points - target[None], axis=1))
    return float(initial_distance - closest_distance)


def minimum_segment_clearance_xy(start_position, positions, centers, radii):
    start = np.asarray(start_position, dtype=np.float64)
    points = np.asarray(positions, dtype=np.float64)
    obstacle_centers = np.asarray(centers, dtype=np.float64)
    obstacle_radii = np.asarray(radii, dtype=np.float64)
    if obstacle_centers.size == 0:
        return float("inf")
    if obstacle_centers.ndim != 2 or obstacle_centers.shape[1] not in (2, 3):
        raise ValueError("Obstacle centers must be [N,2] or [N,3]")
    if obstacle_radii.shape != (len(obstacle_centers),):
        raise ValueError("Obstacle radii must match centers")

    polyline = np.concatenate((start[None, :2], points[:, :2]), axis=0)
    segment_starts = polyline[:-1, None, :]
    segment_delta = (polyline[1:] - polyline[:-1])[:, None, :]
    relative = obstacle_centers[None, :, :2] - segment_starts
    denominator = np.sum(segment_delta * segment_delta, axis=-1)
    numerator = np.sum(relative * segment_delta, axis=-1)
    fraction = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-16,
    )
    closest = segment_starts + np.clip(fraction, 0.0, 1.0)[..., None] * segment_delta
    clearances = (
        np.linalg.norm(closest - obstacle_centers[None, :, :2], axis=-1)
        - obstacle_radii[None, :]
    )
    return float(np.min(clearances))


def point_clearance_xy(position, centers, radii):
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if len(radii) == 0:
        return float("inf")
    return float(
        np.min(np.linalg.norm(centers[:, :2] - np.asarray(position)[None, :2], axis=1) - radii)
    )


def chunk_geometry(start_position, actions, target_position, centers, radii):
    positions = reconstruct_positions(start_position, actions)
    return {
        "positions": positions,
        "progress_m": closest_target_progress(start_position, positions, target_position),
        "clearance_m": minimum_segment_clearance_xy(
            start_position, positions, centers, radii
        ),
    }


def should_trigger(
    *,
    nominal_progress_m,
    nominal_clearance_m,
    start_clearance_m,
    target_contact_seen,
    obstacle_contact_seen,
    obstacle_count,
    config=SelectorConfig(),
):
    config.validate()
    reasons = []
    if obstacle_count <= 0:
        reasons.append("no_obstacles")
    if target_contact_seen:
        reasons.append("target_contact_seen")
    if obstacle_contact_seen:
        reasons.append("obstacle_contact_seen")
    if nominal_progress_m <= 0.0:
        reasons.append("nonpositive_nominal_progress")
    if nominal_clearance_m > config.trigger_clearance_m:
        reasons.append("nominal_clear")
    if start_clearance_m < config.minimum_start_clearance_m:
        reasons.append("start_too_close")
    return len(reasons) == 0, reasons


def normalized_chunk_deviation(candidate, nominal):
    candidate = np.asarray(candidate, dtype=np.float64)
    nominal = np.asarray(nominal, dtype=np.float64)
    if candidate.shape != nominal.shape:
        raise ValueError("Candidate and nominal normalized chunks must match")
    return float(np.sqrt(np.mean(np.square(candidate - nominal))))


def select_candidate(
    *,
    nominal_progress_m,
    nominal_normalized_chunk,
    candidate_raw_chunks,
    candidate_normalized_chunks,
    start_position,
    target_position,
    centers,
    radii,
    config=SelectorConfig(),
):
    """Select the least-deviating safe, progress-preserving candidate."""

    config.validate()
    raw = np.asarray(candidate_raw_chunks, dtype=np.float64)
    normalized = np.asarray(candidate_normalized_chunks, dtype=np.float64)
    if raw.ndim != 3 or normalized.shape != raw.shape:
        raise ValueError("Candidate chunks must have matching [K,H,D] shapes")
    if raw.shape[0] == 0:
        raise ValueError("At least one candidate is required")

    required_progress = (
        nominal_progress_m - config.progress_tolerance_m
        if config.progress_rule == "source_relative"
        else -config.progress_tolerance_m
    )
    rows = []
    for index, (raw_chunk, normalized_chunk) in enumerate(zip(raw, normalized)):
        geometry = chunk_geometry(
            start_position, raw_chunk, target_position, centers, radii
        )
        progress_ok = bool(geometry["progress_m"] >= required_progress)
        clearance_ok = bool(
            geometry["clearance_m"] >= config.candidate_clearance_m
        )
        rows.append(
            {
                "index": index,
                "progress_m": geometry["progress_m"],
                "required_progress_m": required_progress,
                "clearance_m": geometry["clearance_m"],
                "progress_ok": progress_ok,
                "clearance_ok": clearance_ok,
                "eligible": progress_ok and clearance_ok,
                "normalized_rms_deviation": normalized_chunk_deviation(
                    normalized_chunk, nominal_normalized_chunk
                ),
            }
        )

    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None, rows
    selected = min(
        eligible,
        key=lambda row: (
            row["normalized_rms_deviation"],
            -row["clearance_m"],
            -row["progress_m"],
            row["index"],
        ),
    )
    return int(selected["index"]), rows


def chunk_noise_seeds(policy_seed, chunk_index, candidate_count):
    if policy_seed < 0 or chunk_index < 0 or candidate_count <= 0:
        raise ValueError("Seeds and chunk index must be non-negative; count must be positive")
    nominal_seed = int(policy_seed) * 10_000 + int(chunk_index)
    candidate_seeds = [nominal_seed]
    candidate_seeds.extend(
        nominal_seed + 100_000 + index for index in range(candidate_count - 1)
    )
    return nominal_seed, candidate_seeds
