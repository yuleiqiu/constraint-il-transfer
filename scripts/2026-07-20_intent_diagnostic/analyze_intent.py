"""Offline clean-policy intent diagnostic for the delta-EEF rollout matrix.

This analysis reads the existing unguided HDF5 rollouts. It does not create an
environment, sample a policy, or modify guidance.

Usage:
    uv run python scripts/2026-07-20_intent_diagnostic/analyze_intent.py \
      --eval-dir outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50 \
      --out-dir outputs/guided_denoising/intent_diagnostic
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from robosuite.models.objects import BreadObject, CanObject, CerealObject, MilkObject


DEFAULT_EVAL_DIR = "outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50"
DEFAULT_OUT_DIR = "outputs/guided_denoising/intent_diagnostic"
TARGET_OBJECT = "Can"
HARD_DISTRACTOR_COUNTS = (2, 3)
EXPECTED_HDF5_FILES = 12
EXPECTED_EPISODES = 600
EXPECTED_MAIN_COUNTS = {1: 32, 2: 93, 3: 96}
PROGRESS_THRESHOLDS_M = (0.0, 0.0005, 0.001)
CLEARANCE_MARGINS_M = (0.0, 0.01, 0.02)
CATEGORIES = (
    "target_directed_obstructed",
    "non_target_directed",
    "target_directed_clear",
)
CATEGORY_COLORS = {
    "target_directed_obstructed": "#d95f02",
    "non_target_directed": "#7570b3",
    "target_directed_clear": "#1b9e77",
}
OBJECT_COLORS = {"Can": "#2ca02c", "Bread": "#d8a04c", "Cereal": "#9467bd", "Milk": "#1f77b4"}


def decode_strings(values: Iterable[object]) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def first_positive_step(values: Sequence[float]) -> int | None:
    indices = np.flatnonzero(np.asarray(values) > 0)
    return int(indices[0]) if len(indices) else None


def point_segment_distance_xy(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    """Minimum XY distance from a point to a closed line segment."""
    point_xy = np.asarray(point, dtype=np.float64)[:2]
    start_xy = np.asarray(start, dtype=np.float64)[:2]
    end_xy = np.asarray(end, dtype=np.float64)[:2]
    delta = end_xy - start_xy
    denom = float(np.dot(delta, delta))
    if denom == 0.0:
        return float(np.linalg.norm(point_xy - start_xy))
    fraction = float(np.dot(point_xy - start_xy, delta) / denom)
    closest = start_xy + np.clip(fraction, 0.0, 1.0) * delta
    return float(np.linalg.norm(point_xy - closest))


def segment_circle_clearance_xy(
    start: Sequence[float], end: Sequence[float], center: Sequence[float], radius: float
) -> float:
    """Signed segment-to-circle clearance; crossing is negative and tangent is zero."""
    return point_segment_distance_xy(center, start, end) - float(radius)


def polyline_circle_clearance_xy(path: np.ndarray, center: Sequence[float], radius: float) -> float:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
        raise ValueError(f"Expected non-empty [N,D>=2] path, got {path.shape}")
    if path.shape[0] == 1:
        return float(np.linalg.norm(path[0, :2] - np.asarray(center)[:2]) - radius)
    return min(segment_circle_clearance_xy(a, b, center, radius) for a, b in zip(path[:-1], path[1:]))


def closest_progress(path: np.ndarray, target: Sequence[float], xy_only: bool = False) -> float:
    """Start distance minus closest predicted-waypoint distance.

    ``path[0]`` is p0 and is intentionally excluded from the minimum over
    predicted waypoints p_h. This preserves negative progress when every
    predicted waypoint moves farther from the target.
    """
    path = np.asarray(path, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2:
        raise ValueError("closest progress requires p0 and at least one predicted waypoint")
    dims = 2 if xy_only else 3
    distances = np.linalg.norm(path[:, :dims] - target[:dims], axis=1)
    return float(distances[0] - np.min(distances[1:]))


def endpoint_progress(path: np.ndarray, target: Sequence[float], xy_only: bool = False) -> float:
    path = np.asarray(path, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dims = 2 if xy_only else 3
    return float(np.linalg.norm(path[0, :dims] - target[:dims]) - np.linalg.norm(path[-1, :dims] - target[:dims]))


def classify_chunk(progress_m: float, clearance_m: float, progress_threshold_m: float = 0.0, margin_m: float = 0.0) -> str:
    """Return exactly one of the diagnostic's three fixed categories."""
    if not math.isfinite(progress_m):
        raise ValueError("progress must be finite")
    target_directed = progress_m > progress_threshold_m
    obstructed = math.isfinite(clearance_m) and clearance_m <= margin_m
    if not target_directed:
        return "non_target_directed"
    if obstructed:
        return "target_directed_obstructed"
    return "target_directed_clear"


def chunk_containing_step(chunk_starts: Sequence[int], executed_lengths: Sequence[int], step: int) -> int:
    matches = [i for i, (start, length) in enumerate(zip(chunk_starts, executed_lengths)) if start <= step < start + length]
    if len(matches) != 1:
        raise ValueError(f"Step {step} belongs to {len(matches)} chunks")
    return matches[0]


def select_critical_chunk(
    chunk_starts: Sequence[int],
    executed_lengths: Sequence[int],
    cohort: str,
    first_target_contact_step: int | None,
    first_obstacle_contact_step: int | None,
) -> tuple[int, str]:
    """Select the fixed critical chunk and return (chunk index, reason)."""
    if not chunk_starts:
        raise ValueError("Episode has no action chunks")
    if cohort == "main_analysis":
        if first_target_contact_step is not None:
            raise ValueError("main_analysis episode must never contact the target")
        if first_obstacle_contact_step is not None:
            return (
                chunk_containing_step(chunk_starts, executed_lengths, first_obstacle_contact_step),
                "first_obstacle_contact",
            )
        return len(chunk_starts) - 1, "last_chunk_no_contact"
    if cohort in {"success_calibration", "downstream_failure"}:
        if first_target_contact_step is None:
            raise ValueError(f"{cohort} episode has no target contact")
        return chunk_containing_step(chunk_starts, executed_lengths, first_target_contact_step), "first_target_contact"
    raise ValueError(f"Unknown cohort: {cohort}")


def object_horizontal_radii() -> dict[str, float]:
    """Load radii through the same robosuite object properties used by eval."""
    classes = {"Bread": BreadObject, "Cereal": CerealObject, "Milk": MilkObject, "Can": CanObject}
    return {name: float(cls(name=name).horizontal_radius) for name, cls in classes.items()}


def cohort_for_episode(success: bool, first_target_contact_step: int | None) -> str:
    if success:
        return "success_calibration"
    if first_target_contact_step is not None:
        return "downstream_failure"
    return "main_analysis"


def sorted_chunk_names(chunks: h5py.Group) -> list[str]:
    return sorted(chunks.keys(), key=lambda name: int(name.removeprefix("chunk_")))


def environment_distractor_count(object_names: Sequence[str]) -> int:
    if TARGET_OBJECT not in object_names:
        raise ValueError(f"Target {TARGET_OBJECT} missing from {object_names}")
    return len(object_names) - 1


def calculate_chunk_metrics(
    env: str,
    seed: int,
    episode: int,
    success: bool,
    cohort: str,
    ep: h5py.Group,
    chunk_name: str,
    object_names: Sequence[str],
    radii: dict[str, float],
    first_target_contact_step: int | None,
    first_obstacle_contact_step: int | None,
) -> dict[str, object]:
    chunk = ep["chunks"][chunk_name]
    chunk_index = int(chunk_name.removeprefix("chunk_"))
    start_step = int(chunk.attrs["start_step"])
    executed_len = int(chunk.attrs["executed_len"])
    predicted = np.asarray(chunk["raw_pred_eef_pos"][()], dtype=np.float64)
    start_pos = np.asarray(chunk["start_pos"][()], dtype=np.float64)
    path = np.concatenate([start_pos[None], predicted], axis=0)
    positions = np.asarray(ep["object_pos"][start_step], dtype=np.float64)
    target_index = object_names.index(TARGET_OBJECT)
    target_pos = positions[target_index]

    obstacle_clearances: list[tuple[float, str]] = []
    for object_index, object_name in enumerate(object_names):
        if object_name == TARGET_OBJECT:
            continue
        clearance = polyline_circle_clearance_xy(path, positions[object_index], radii[object_name])
        obstacle_clearances.append((clearance, object_name))
    if obstacle_clearances:
        clearance_m, closest_obstacle = min(obstacle_clearances, key=lambda item: (item[0], item[1]))
    else:
        clearance_m, closest_obstacle = math.inf, ""

    progress_closest_m = closest_progress(path, target_pos)
    progress_endpoint_m = endpoint_progress(path, target_pos)
    progress_closest_xy_m = closest_progress(path, target_pos, xy_only=True)
    progress_endpoint_xy_m = endpoint_progress(path, target_pos, xy_only=True)
    category = classify_chunk(progress_closest_m, clearance_m)

    return {
        "env": env,
        "distractors": environment_distractor_count(object_names),
        "seed": seed,
        "episode": episode,
        "success": success,
        "cohort": cohort,
        "first_target_contact_step": first_target_contact_step,
        "first_obstacle_contact_step": first_obstacle_contact_step,
        "chunk_index": chunk_index,
        "chunk_start_step": start_step,
        "chunk_executed_len": executed_len,
        "chunk_action_horizon": int(chunk.attrs["action_horizon"]),
        "target_x_m": float(target_pos[0]),
        "target_y_m": float(target_pos[1]),
        "target_z_m": float(target_pos[2]),
        "eef_start_distance_m": float(np.linalg.norm(path[0] - target_pos)),
        "eef_start_distance_xy_m": float(np.linalg.norm(path[0, :2] - target_pos[:2])),
        "progress_closest_m": progress_closest_m,
        "progress_endpoint_m": progress_endpoint_m,
        "progress_closest_xy_m": progress_closest_xy_m,
        "progress_endpoint_xy_m": progress_endpoint_xy_m,
        "clearance_m": clearance_m,
        "closest_obstacle": closest_obstacle,
        "target_directed_gt_0mm": progress_closest_m > 0.0,
        "target_directed_gt_0_5mm": progress_closest_m > 0.0005,
        "target_directed_gt_1mm": progress_closest_m > 0.001,
        "obstructed_margin_0cm": math.isfinite(clearance_m) and clearance_m <= 0.0,
        "obstructed_margin_1cm": math.isfinite(clearance_m) and clearance_m <= 0.01,
        "obstructed_margin_2cm": math.isfinite(clearance_m) and clearance_m <= 0.02,
        "category": category,
    }


def collect_metrics(eval_dir: Path, radii: dict[str, float]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    h5_paths = sorted(eval_dir.glob("*/seed_*/episodes.hdf5"))
    if len(h5_paths) != EXPECTED_HDF5_FILES:
        raise RuntimeError(f"Expected {EXPECTED_HDF5_FILES} episodes.hdf5 files, found {len(h5_paths)}")

    chunk_rows: list[dict[str, object]] = []
    critical_rows: list[dict[str, object]] = []
    episode_count = 0
    for h5_path in h5_paths:
        env_from_path = h5_path.parent.parent.name
        seed_from_path = int(h5_path.parent.name.removeprefix("seed_"))
        with h5py.File(h5_path, "r") as h5:
            if str(h5.attrs["env"]) != env_from_path or int(h5.attrs["seed"]) != seed_from_path:
                raise RuntimeError(f"Path / HDF5 metadata mismatch: {h5_path}")
            for episode_name in sorted(h5["episodes"].keys(), key=lambda name: int(name.removeprefix("episode_"))):
                ep = h5["episodes"][episode_name]
                episode = int(episode_name.removeprefix("episode_"))
                success = bool(ep.attrs["success"])
                object_names = decode_strings(ep["object_names"][()])
                first_target = first_positive_step(ep["target_contact_count"][()])
                first_obstacle = first_positive_step(ep["non_target_contact_count"][()])
                cohort = cohort_for_episode(success, first_target)
                chunk_names = sorted_chunk_names(ep["chunks"])
                starts = [int(ep["chunks"][name].attrs["start_step"]) for name in chunk_names]
                lengths = [int(ep["chunks"][name].attrs["executed_len"]) for name in chunk_names]
                critical_index, critical_reason = select_critical_chunk(starts, lengths, cohort, first_target, first_obstacle)

                episode_rows: list[dict[str, object]] = []
                for index, chunk_name in enumerate(chunk_names):
                    start_step = starts[index]
                    if first_target is not None and start_step > first_target:
                        continue
                    row = calculate_chunk_metrics(
                        env_from_path,
                        seed_from_path,
                        episode,
                        success,
                        cohort,
                        ep,
                        chunk_name,
                        object_names,
                        radii,
                        first_target,
                        first_obstacle,
                    )
                    row["is_critical"] = index == critical_index
                    row["critical_reason"] = critical_reason if index == critical_index else ""
                    episode_rows.append(row)
                critical_matches = [row for row in episode_rows if row["is_critical"]]
                if len(critical_matches) != 1:
                    raise RuntimeError(f"Expected one critical pre-target chunk for {env_from_path}/{seed_from_path}/{episode}")
                chunk_rows.extend(episode_rows)
                critical_rows.append(dict(critical_matches[0]))
                episode_count += 1

    if episode_count != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected {EXPECTED_EPISODES} episodes, found {episode_count}")
    observed_main = Counter(int(row["distractors"]) for row in critical_rows if row["cohort"] == "main_analysis")
    for distractors, expected in EXPECTED_MAIN_COUNTS.items():
        if observed_main[distractors] != expected:
            raise RuntimeError(
                f"Expected {expected} main-analysis episodes with {distractors} distractors, found {observed_main[distractors]}"
            )
    return chunk_rows, critical_rows, {"hdf5_files": len(h5_paths), "episodes": episode_count}


def csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isinf(float(value)):
            return "inf" if float(value) > 0 else "-inf"
        return format(float(value), ".17g")
    return value


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return center - spread, center + spread


def finite_summary(values: Iterable[object]) -> dict[str, float | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": finite[0],
        "max": finite[-1],
    }


def group_summary(rows: Sequence[dict[str, object]], group: dict[str, object]) -> dict[str, object]:
    counts = Counter(str(row["category"]) for row in rows)
    total = len(rows)
    categories = {}
    for category in CATEGORIES:
        count = counts[category]
        low, high = wilson_interval(count, total)
        categories[category] = {
            "count": count,
            "rate": count / total if total else None,
            "wilson_95": [low, high],
        }
    return {
        **group,
        "n": total,
        "categories": categories,
        "progress_closest_m": finite_summary(row["progress_closest_m"] for row in rows),
        "progress_endpoint_m": finite_summary(row["progress_endpoint_m"] for row in rows),
        "progress_closest_xy_m": finite_summary(row["progress_closest_xy_m"] for row in rows),
        "clearance_m": finite_summary(row["clearance_m"] for row in rows),
    }


def grouped_summaries(critical_rows: Sequence[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    env_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    seed_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in critical_rows:
        env_key = (row["env"], row["distractors"], row["cohort"])
        seed_key = (row["env"], row["distractors"], row["seed"], row["cohort"])
        env_groups[env_key].append(row)
        seed_groups[seed_key].append(row)
    env_result = [
        group_summary(rows, {"env": key[0], "distractors": key[1], "cohort": key[2]})
        for key, rows in sorted(env_groups.items())
    ]
    seed_result = [
        group_summary(rows, {"env": key[0], "distractors": key[1], "seed": key[2], "cohort": key[3]})
        for key, rows in sorted(seed_groups.items())
    ]
    return env_result, seed_result


def sensitivity_summaries(critical_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for distractors in (1, 2, 3):
        rows = [
            row for row in critical_rows if row["cohort"] == "main_analysis" and int(row["distractors"]) == distractors
        ]
        env = str(rows[0]["env"]) if rows else ""
        for threshold in PROGRESS_THRESHOLDS_M:
            for margin in CLEARANCE_MARGINS_M:
                classified = [
                    classify_chunk(float(row["progress_closest_m"]), float(row["clearance_m"]), threshold, margin)
                    for row in rows
                ]
                counts = Counter(classified)
                result.append(
                    {
                        "env": env,
                        "distractors": distractors,
                        "n": len(rows),
                        "progress_threshold_m": threshold,
                        "clearance_margin_m": margin,
                        "categories": {
                            category: {"count": counts[category], "rate": counts[category] / len(rows) if rows else None}
                            for category in CATEGORIES
                        },
                    }
                )
    return result


def diagnostic_rate_summaries(critical_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Report alternate intent definitions and clearance margins by cohort."""
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in critical_rows:
        groups[(row["env"], row["distractors"], row["cohort"])].append(row)
    result = []
    for (env, distractors, cohort), rows in sorted(groups.items()):
        total = len(rows)

        def rate(predicate) -> float:
            return sum(bool(predicate(row)) for row in rows) / total

        result.append(
            {
                "env": env,
                "distractors": distractors,
                "cohort": cohort,
                "n": total,
                "intent_rates": {
                    "closest_3d_gt_0mm": rate(lambda row: float(row["progress_closest_m"]) > 0.0),
                    "closest_3d_gt_0_5mm": rate(lambda row: float(row["progress_closest_m"]) > 0.0005),
                    "closest_3d_gt_1mm": rate(lambda row: float(row["progress_closest_m"]) > 0.001),
                    "endpoint_3d_gt_0mm": rate(lambda row: float(row["progress_endpoint_m"]) > 0.0),
                    "closest_xy_gt_0mm": rate(lambda row: float(row["progress_closest_xy_m"]) > 0.0),
                    "endpoint_xy_gt_0mm": rate(lambda row: float(row["progress_endpoint_xy_m"]) > 0.0),
                },
                "obstruction_rates": {
                    "margin_0cm": rate(
                        lambda row: math.isfinite(float(row["clearance_m"])) and float(row["clearance_m"]) <= 0.0
                    ),
                    "margin_1cm": rate(
                        lambda row: math.isfinite(float(row["clearance_m"])) and float(row["clearance_m"]) <= 0.01
                    ),
                    "margin_2cm": rate(
                        lambda row: math.isfinite(float(row["clearance_m"])) and float(row["clearance_m"]) <= 0.02
                    ),
                },
            }
        )
    return result


def decide_direction(env_summaries: Sequence[dict[str, object]]) -> dict[str, object]:
    hard = [
        summary
        for summary in env_summaries
        if summary["cohort"] == "main_analysis" and int(summary["distractors"]) in HARD_DISTRACTOR_COUNTS
    ]
    if len(hard) != 2:
        raise RuntimeError(f"Expected two hard-environment main-analysis summaries, found {len(hard)}")

    def qualifies(category: str) -> bool:
        return all(
            summary["categories"][category]["rate"] > 0.5
            and summary["categories"][category]["wilson_95"][0] > 0.5
            for summary in hard
        )

    if qualifies("target_directed_obstructed"):
        outcome = "A"
        label = "target-directed but obstructed dominates"
    elif qualifies("non_target_directed"):
        outcome = "B"
        label = "non-target-directed intent dominates"
    else:
        outcome = "C"
        label = "mixed / no single dominant mechanism"
    hard_records = []
    for summary in hard:
        largest = max(CATEGORIES, key=lambda category: summary["categories"][category]["count"])
        hard_records.append(
            {
                "env": summary["env"],
                "distractors": summary["distractors"],
                "n": summary["n"],
                "largest_category": largest,
                "categories": summary["categories"],
                "target_directed_clear_is_largest": largest == "target_directed_clear",
            }
        )
    return {
        "outcome": outcome,
        "label": label,
        "rule": "A or B requires the same category to exceed 0.5 with Wilson 95% CI lower bound > 0.5 in both 2- and 3-distractor main-analysis cohorts; otherwise C.",
        "hard_environments": hard_records,
        "target_directed_clear_largest_in_any_hard_environment": any(
            record["target_directed_clear_is_largest"] for record in hard_records
        ),
    }


def select_representatives(critical_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for distractors in HARD_DISTRACTOR_COUNTS:
        for category in CATEGORIES:
            rows = [
                row
                for row in critical_rows
                if row["cohort"] == "main_analysis"
                and int(row["distractors"]) == distractors
                and row["category"] == category
            ]
            if not rows:
                continue
            progress_median = float(np.median([float(row["progress_closest_m"]) for row in rows]))
            clearance_median = float(np.median([float(row["clearance_m"]) for row in rows]))
            representative = min(
                rows,
                key=lambda row: (
                    math.hypot(
                        float(row["progress_closest_m"]) - progress_median,
                        float(row["clearance_m"]) - clearance_median,
                    ),
                    int(row["seed"]),
                    int(row["episode"]),
                ),
            )
            selected.append(
                {
                    "env": representative["env"],
                    "distractors": distractors,
                    "seed": representative["seed"],
                    "episode": representative["episode"],
                    "chunk_index": representative["chunk_index"],
                    "category": category,
                    "progress_closest_m": representative["progress_closest_m"],
                    "clearance_m": representative["clearance_m"],
                    "category_progress_median": progress_median,
                    "category_clearance_median": clearance_median,
                }
            )
    return selected


def plot_category_proportions(critical_rows: Sequence[dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bottoms = np.zeros(3)
    x = np.arange(3)
    for category in CATEGORIES:
        rates = []
        for distractors in (1, 2, 3):
            rows = [
                row for row in critical_rows if row["cohort"] == "main_analysis" and row["distractors"] == distractors
            ]
            rates.append(sum(row["category"] == category for row in rows) / len(rows))
        ax.bar(x, rates, bottom=bottoms, color=CATEGORY_COLORS[category], label=category.replace("_", " "))
        bottoms += np.asarray(rates)
    ax.set_xticks(x, ["1 distractor\n(n=32)", "2 distractors\n(n=93)", "3 distractors\n(n=96)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of main-analysis episodes")
    ax.set_title("Clean-policy intent diagnostic at the critical chunk")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_progress_clearance(critical_rows: Sequence[dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    for ax, distractors in zip(axes, (1, 2, 3)):
        rows = [
            row for row in critical_rows if row["cohort"] == "main_analysis" and row["distractors"] == distractors
        ]
        for category in CATEGORIES:
            category_rows = [row for row in rows if row["category"] == category]
            ax.scatter(
                [100 * float(row["progress_closest_m"]) for row in category_rows],
                [100 * float(row["clearance_m"]) for row in category_rows],
                s=22,
                alpha=0.75,
                color=CATEGORY_COLORS[category],
                edgecolors="none",
                label=category.replace("_", " "),
            )
        ax.axvline(0, color="black", lw=0.9)
        ax.axhline(0, color="black", lw=0.9)
        ax.set_title(f"{distractors} distractor{'s' if distractors > 1 else ''} (n={len(rows)})")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Minimum segment clearance (cm)")
    axes[1].set_xlabel("Closest target progress (cm)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.suptitle("Critical-chunk target progress versus obstacle clearance")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_representative_geometry(eval_dir: Path, representative: dict[str, object], radii: dict[str, float]) -> dict[str, object]:
    h5_path = eval_dir / str(representative["env"]) / f"seed_{representative['seed']}" / "episodes.hdf5"
    with h5py.File(h5_path, "r") as h5:
        ep = h5["episodes"][f"episode_{representative['episode']}"]
        chunk = ep["chunks"][f"chunk_{representative['chunk_index']}"]
        start_step = int(chunk.attrs["start_step"])
        start_pos = np.asarray(chunk["start_pos"][()], dtype=np.float64)
        predicted = np.asarray(chunk["raw_pred_eef_pos"][()], dtype=np.float64)
        return {
            "path": np.concatenate([start_pos[None], predicted], axis=0),
            "object_names": decode_strings(ep["object_names"][()]),
            "object_pos": np.asarray(ep["object_pos"][start_step], dtype=np.float64),
            "radii": radii,
        }


def plot_representatives(
    eval_dir: Path, out_dir: Path, representatives: Sequence[dict[str, object]], radii: dict[str, float]
) -> None:
    case_dir = out_dir / "representative_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for representative in representatives:
        geometry = load_representative_geometry(eval_dir, representative, radii)
        path = geometry["path"]
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        ax.plot(path[:, 0], path[:, 1], "o-", color="#222222", lw=1.8, ms=3.5, label="predicted EEF path")
        ax.scatter(path[0, 0], path[0, 1], marker="s", s=55, color="#222222", zorder=4, label="chunk start")
        for name, position in zip(geometry["object_names"], geometry["object_pos"]):
            color = OBJECT_COLORS.get(name, "#777777")
            if name == TARGET_OBJECT:
                ax.scatter(position[0], position[1], marker="*", s=180, color=color, zorder=5, label="Can target")
            else:
                ax.add_patch(Circle(position[:2], geometry["radii"][name], facecolor=color, edgecolor=color, alpha=0.25))
                ax.scatter(position[0], position[1], marker="x", s=55, color=color, zorder=4, label=name)
        all_x = np.concatenate([path[:, 0], geometry["object_pos"][:, 0]])
        all_y = np.concatenate([path[:, 1], geometry["object_pos"][:, 1]])
        padding = 0.055
        ax.set_xlim(float(np.min(all_x) - padding), float(np.max(all_x) + padding))
        ax.set_ylim(float(np.min(all_y) - padding), float(np.max(all_y) + padding))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.grid(alpha=0.2)
        ax.set_title(
            f"{representative['category'].replace('_', ' ')}\n"
            f"seed {representative['seed']}, episode {representative['episode']}, chunk {representative['chunk_index']}"
        )
        ax.legend(loc="best", fontsize=8, frameon=False)
        fig.tight_layout()
        filename = (
            f"d{representative['distractors']}_{representative['category']}_"
            f"seed{representative['seed']}_ep{representative['episode']}.png"
        )
        fig.savefig(case_dir / filename, dpi=180)
        plt.close(fig)


def fmt_rate(category_record: dict[str, object]) -> str:
    low, high = category_record["wilson_95"]
    return f"{category_record['count']} ({100 * category_record['rate']:.1f}%; 95% CI {100 * low:.1f}--{100 * high:.1f}%)"


def write_report(path: Path, summary: dict[str, object]) -> None:
    main_env = [item for item in summary["environment_cohort_summaries"] if item["cohort"] == "main_analysis"]
    main_env.sort(key=lambda item: int(item["distractors"]))
    outcome = summary["decision"]["outcome"]
    if outcome == "A":
        decision_explanation = (
            "Both hard environments have a majority of target-directed obstructed episodes and both Wilson 95% "
            "lower bounds exceed 0.5. This supports behavior-preserving feasibility adaptation as the next method direction."
        )
    elif outcome == "B":
        decision_explanation = (
            "Both hard environments have a majority of non-target-directed episodes and both Wilson 95% lower "
            "bounds exceed 0.5. Action-side avoidance is therefore not the right primary method direction."
        )
    else:
        decision_explanation = (
            "In each of the two hardest environments, neither candidate dominant class satisfies the fixed "
            "requirement that its Wilson 95% lower bound exceed 0.5. The third class is stated explicitly in the "
            "table and decision record rather than folded into obstruction or loss of intent."
        )
    rate_lookup = {
        (int(item["distractors"]), str(item["cohort"])): item for item in summary["diagnostic_rate_summaries"]
    }
    lines = [
        "# P0 Clean-Policy Intent Diagnostic",
        "",
        "Status: complete, 2026-07-20.",
        "",
        "## Result",
        "",
        f"**Decision {summary['decision']['outcome']}: {summary['decision']['label']}.**",
        "",
        "The diagnostic reuses the 600 existing unguided rollouts. The main analysis is restricted to task failures that never contacted Can; successful episodes calibrate the intent metric and failures after Can contact are a downstream comparison cohort.",
        "",
        "| distractors | main n | target-directed obstructed | non-target-directed | target-directed clear |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in main_env:
        if int(item["distractors"]) == 0:
            continue
        cats = item["categories"]
        lines.append(
            f"| {item['distractors']} | {item['n']} | {fmt_rate(cats['target_directed_obstructed'])} | "
            f"{fmt_rate(cats['non_target_directed'])} | {fmt_rate(cats['target_directed_clear'])} |"
        )
    lines.extend(
        [
            "",
            f"The one-distractor environment is a trend reference only. {decision_explanation}",
            "",
            "## Metric sensitivity",
            "",
            "| distractors | closest 3D >0 | >0.5 mm | >1 mm | endpoint 3D >0 | closest XY >0 | endpoint XY >0 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for distractors in (1, 2, 3):
        rates = rate_lookup[(distractors, "main_analysis")]["intent_rates"]
        lines.append(
            f"| {distractors} | {100 * rates['closest_3d_gt_0mm']:.1f}% | "
            f"{100 * rates['closest_3d_gt_0_5mm']:.1f}% | {100 * rates['closest_3d_gt_1mm']:.1f}% | "
            f"{100 * rates['endpoint_3d_gt_0mm']:.1f}% | {100 * rates['closest_xy_gt_0mm']:.1f}% | "
            f"{100 * rates['endpoint_xy_gt_0mm']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "| distractors | clearance <=0 | clearance <=+1 cm | clearance <=+2 cm |",
            "|---:|---:|---:|---:|",
        ]
    )
    for distractors in (1, 2, 3):
        rates = rate_lookup[(distractors, "main_analysis")]["obstruction_rates"]
        lines.append(
            f"| {distractors} | {100 * rates['margin_0cm']:.1f}% | {100 * rates['margin_1cm']:.1f}% | "
            f"{100 * rates['margin_2cm']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "These are marginal intent and obstruction rates; the full 3 x 3 threshold/margin category partition is in `summary.json`.",
            "",
            "For calibration, primary closest-3D target-directed rates in successful episodes are "
            + ", ".join(
                f"{d} distractor{'s' if d > 1 else ''}: {100 * rate_lookup[(d, 'success_calibration')]['intent_rates']['closest_3d_gt_0mm']:.1f}%"
                for d in (1, 2, 3)
            )
            + ".",
            "",
            "## Definitions",
            "",
            "For every pre-target chunk, the predicted EEF polyline is the chunk start followed by all raw predicted executable delta-EEF waypoints. Target intent is",
            "",
            "```text",
            "closest progress = distance(start, Can) - min distance(waypoint, Can)",
            "```",
            "",
            "and is positive only when the chunk gets strictly closer in 3D. Endpoint and XY variants, plus 0.5 mm and 1 mm intent thresholds, are retained in the CSV and JSON sensitivity results.",
            "",
            "Obstruction is the minimum signed XY segment clearance from that polyline to any distractor circle at the chunk start. Circle radii come from robosuite `horizontal_radius`; zero or negative clearance is obstructed. The JSON also reports +1 cm and +2 cm margin sensitivity.",
            "",
            "Critical chunks are fixed before computing categories: the chunk containing first obstacle contact for main episodes with contact, the last chunk for main episodes with no contact, and the chunk containing first Can contact for calibration and downstream cohorts.",
            "",
            "## Integrity and scope",
            "",
            f"- Read {summary['input']['hdf5_files']} HDF5 files and {summary['input']['episodes']} episodes; emitted {summary['input']['critical_rows']} critical rows and {summary['input']['pre_target_chunk_rows']} pre-target chunk rows.",
            f"- Main-analysis counts are {summary['input']['main_analysis_counts']['1']} / {summary['input']['main_analysis_counts']['2']} / {summary['input']['main_analysis_counts']['3']} for 1 / 2 / 3 distractors.",
            "- This is an EEF-point / object-footprint diagnostic, not a collision model for the gripper, arm, or whole robot.",
            "- No counterfactual observation, extra rollout, policy training, collision geom ID, or new guidance cost is used.",
            "",
            "## Artifacts",
            "",
            "- `chunk_metrics.csv`: continuous metrics for every pre-target chunk.",
            "- `critical_episode_metrics.csv`: exactly one selected chunk per episode.",
            "- `summary.json`: environment, seed, cohort, threshold sensitivity, decision, and representative selection records.",
            "- `category_proportions.png` and `progress_clearance_scatter.png`: aggregate views.",
            "- `representative_cases/`: deterministic median-nearest top-view examples in the two hardest environments.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def build_summary(
    eval_dir: Path,
    chunk_rows: Sequence[dict[str, object]],
    critical_rows: Sequence[dict[str, object]],
    integrity: dict[str, int],
    radii: dict[str, float],
) -> dict[str, object]:
    environment_summaries, seed_summaries = grouped_summaries(critical_rows)
    main_counts = Counter(str(row["distractors"]) for row in critical_rows if row["cohort"] == "main_analysis")
    representatives = select_representatives(critical_rows)
    return {
        "analysis": "P0 clean-policy intent diagnostic",
        "analysis_date": "2026-07-20",
        "input": {
            "eval_dir": str(eval_dir),
            "hdf5_files": integrity["hdf5_files"],
            "episodes": integrity["episodes"],
            "pre_target_chunk_rows": len(chunk_rows),
            "critical_rows": len(critical_rows),
            "main_analysis_counts": dict(sorted(main_counts.items())),
        },
        "definitions": {
            "main_analysis": "task failure and never contacted Can",
            "success_calibration": "task success; critical chunk contains first Can contact",
            "downstream_failure": "task failure after contacting Can; critical chunk contains first Can contact",
            "path": "chunk start plus raw_pred_eef_pos waypoints",
            "target_intent_primary": "3D start distance to Can minus minimum 3D waypoint distance; target-directed iff > 0 m",
            "obstruction_primary": "minimum XY polyline-to-distractor-circle clearance using chunk-start positions and robosuite horizontal_radius; obstructed iff <= 0 m",
            "categories": list(CATEGORIES),
            "progress_thresholds_m": list(PROGRESS_THRESHOLDS_M),
            "clearance_margins_m": list(CLEARANCE_MARGINS_M),
        },
        "object_horizontal_radii_m": dict(sorted(radii.items())),
        "environment_cohort_summaries": environment_summaries,
        "seed_cohort_summaries": seed_summaries,
        "diagnostic_rate_summaries": diagnostic_rate_summaries(critical_rows),
        "sensitivity": sensitivity_summaries(critical_rows),
        "decision": decide_direction(environment_summaries),
        "representatives": representatives,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path(DEFAULT_EVAL_DIR))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    radii = object_horizontal_radii()
    chunk_rows, critical_rows, integrity = collect_metrics(eval_dir, radii)
    summary = build_summary(eval_dir, chunk_rows, critical_rows, integrity, radii)

    write_csv(out_dir / "chunk_metrics.csv", chunk_rows)
    write_csv(out_dir / "critical_episode_metrics.csv", critical_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    plot_category_proportions(critical_rows, out_dir / "category_proportions.png")
    plot_progress_clearance(critical_rows, out_dir / "progress_clearance_scatter.png")
    plot_representatives(eval_dir, out_dir, summary["representatives"], radii)
    write_report(out_dir / "report.md", summary)

    print(f"Read {integrity['hdf5_files']} HDF5 files and {integrity['episodes']} episodes")
    print(f"Wrote {len(chunk_rows)} pre-target chunk rows and {len(critical_rows)} critical rows")
    print(f"Decision {summary['decision']['outcome']}: {summary['decision']['label']}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
