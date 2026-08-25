"""Select metric-defined typical cases and render recorded H400 comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from empty2d_guidance.evaluation import load_guidable_policy, rollout
from empty2d_guidance.video import render_pair


CONDITION_NAMES = (
    "centered_hinge",
    "centered_huber_hinge",
    "offset_hinge",
    "offset_huber_hinge",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def by_layout(rows: list[dict]) -> dict[str, dict]:
    return {row["layout_id"]: row for row in rows}


def paired_rows(first: list[dict], second: list[dict]) -> list[tuple[dict, dict]]:
    first_by_layout = by_layout(first)
    second_by_layout = by_layout(second)
    if first_by_layout.keys() != second_by_layout.keys():
        raise ValueError("Comparison layout IDs do not match")
    return [(first_by_layout[key], second_by_layout[key]) for key in first_by_layout]


def select_cases(rows_by_condition: dict[str, list[dict]]) -> list[dict]:
    centered_cost_pairs = paired_rows(
        rows_by_condition["centered_hinge"],
        rows_by_condition["centered_huber_hinge"],
    )
    comparable_centered = [
        pair for pair in centered_cost_pairs if pair[0]["safe_success"] and pair[1]["safe_success"]
    ] or centered_cost_pairs
    centered_hinge, centered_huber = max(
        comparable_centered,
        key=lambda pair: (
            pair[0]["x_direction_reversals"] - pair[1]["x_direction_reversals"],
            pair[0]["backward_distance"] - pair[1]["backward_distance"],
            pair[0]["steps"] - pair[1]["steps"],
        ),
    )

    geometry_pairs = paired_rows(
        rows_by_condition["centered_huber_hinge"],
        rows_by_condition["offset_huber_hinge"],
    )
    comparable_geometry = [
        pair for pair in geometry_pairs if pair[0]["safe_success"] and pair[1]["safe_success"]
    ] or geometry_pairs
    centered_geometry, offset_geometry = max(
        comparable_geometry,
        key=lambda pair: (
            pair[0]["steps"] - pair[1]["steps"],
            pair[0]["x_direction_reversals"] - pair[1]["x_direction_reversals"],
            pair[0]["backward_distance"] - pair[1]["backward_distance"],
        ),
    )

    cases = [
        {
            "name": "smooth_cost_largest_reversal_reduction",
            "selection_rule": (
                "Among centered layouts where both variants safely succeed, maximize "
                "hinge minus Huber x-direction reversals; break ties by backward "
                "distance and completion steps."
            ),
            "first": centered_hinge,
            "second": centered_huber,
        },
        {
            "name": "offset_largest_huber_step_reduction",
            "selection_rule": (
                "Among layouts where centered and offset Huber variants safely succeed, "
                "maximize centered minus offset completion steps; break ties by "
                "x-direction reversals and backward distance."
            ),
            "first": centered_geometry,
            "second": offset_geometry,
        },
    ]

    failures = [
        row
        for condition in CONDITION_NAMES
        for row in rows_by_condition[condition]
        if not row["safe_success"]
    ]
    if failures:
        failure = max(
            failures,
            key=lambda row: (
                not row["success"],
                row["collision_any"],
                row["steps"],
                row["x_direction_reversals"],
            ),
        )
        alternate_name = (
            failure["center_mode"] + "_huber_hinge"
            if failure["obstacle_cost_type"] == "hinge"
            else failure["center_mode"] + "_hinge"
        )
        alternate = by_layout(rows_by_condition[alternate_name])[failure["layout_id"]]
        cases.append(
            {
                "name": "guided_failure_boundary",
                "selection_rule": (
                    "Prefer task non-completion, then collision, longer execution, and "
                    "more x-direction reversals among all guided variants that do not "
                    "safely succeed; compare against the other cost on the same layout."
                ),
                "first": failure,
                "second": alternate,
            }
        )
    return cases


def rerun(policy, row: dict) -> dict:
    recorded = rollout(
        policy,
        environment_name=row["environment"],
        policy_seed=row["policy_seed"],
        guidance_scale=row["guidance_scale"],
        safety_distance=row["safety_distance"],
        record_chunks=True,
        horizon=row["horizon"],
        obstacle_center=row["obstacle_center"],
        obstacle_cost_type=row["obstacle_cost_type"],
        obstacle_cost_smoothing=row["obstacle_cost_smoothing"],
        milestone_horizons=tuple(
            int(value) for value in row.get("horizon_outcomes", {})
        ),
    )
    compared_keys = (
        "success",
        "collision_any",
        "safe_success",
        "steps",
        "x_direction_reversals",
    )
    recorded["selected_rollout_metrics"] = {key: row[key] for key in compared_keys}
    recorded["rerun_matches_selected_metrics"] = all(
        recorded[key] == row[key] for key in compared_keys
    )
    recorded.update(
        {
            "condition": row["condition"],
            "center_mode": row["center_mode"],
            "layout_id": row["layout_id"],
            "visualization_label": (
                f"{row['condition']} | center=({row['obstacle_center'][0]:+.3f},"
                f" {row['obstacle_center'][1]:+.3f})"
            ),
        }
    )
    return recorded


def compact(row: dict) -> dict:
    result = {
        key: row[key]
        for key in (
            "condition",
            "layout_id",
            "policy_seed",
            "obstacle_center",
            "success",
            "collision_any",
            "safe_success",
            "steps",
            "backward_distance",
            "x_direction_reversals",
            "path_total_variation",
        )
    }
    result["selected_rollout_metrics"] = row["selected_rollout_metrics"]
    result["rerun_matches_selected_metrics"] = row[
        "rerun_matches_selected_metrics"
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    rows_by_condition = {
        condition: read_jsonl(args.eval_dir / f"{condition}.jsonl")
        for condition in CONDITION_NAMES
    }
    cases = select_cases(rows_by_condition)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, _ = load_guidable_policy(args.checkpoint, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for case in cases:
        first = rerun(policy, case["first"])
        second = rerun(policy, case["second"])
        video_path = args.out_dir / f"{case['name']}_{first['layout_id']}.mp4"
        render_pair(video_path, first, second)
        manifest_cases.append(
            {
                "name": case["name"],
                "selection_rule": case["selection_rule"],
                "video": str(video_path.resolve()),
                "first": compact(first),
                "second": compact(second),
            }
        )
        print(f"rendered {video_path}", flush=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps({"cases": manifest_cases}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
