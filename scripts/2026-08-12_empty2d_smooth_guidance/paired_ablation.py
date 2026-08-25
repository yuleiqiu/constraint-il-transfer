"""Run the frozen H400 Empty2D smooth-cost / obstacle-offset ablation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from empty2d_guidance.environment import EnvEmpty2DExtraSphere, EnvEmpty2DExtraSquare
from empty2d_guidance.evaluation import load_guidable_policy, rollout
from empty2d_guidance.metrics import exact_mcnemar_p, summarize


HORIZON = 600
HORIZON_MILESTONES = (400, 500, 600)
GUIDANCE_SCALE = 1.0
SAFETY_DISTANCE = 0.12
HUBER_SMOOTHING = 0.03
LAYOUT_SEED = 20_260_812
POLICY_SEED_START = 424_000
OFFSET_X_RANGE = (-0.25, 0.25)
OFFSET_ABS_Y_RANGE = (0.08, 0.18)


@dataclass(frozen=True)
class Layout:
    layout_id: str
    policy_seed: int
    obstacle_center: tuple[float, float]


@dataclass(frozen=True)
class Condition:
    name: str
    center_mode: str
    guidance_scale: float
    obstacle_cost_type: str
    obstacle_cost_smoothing: float


CONDITIONS = (
    Condition("centered_baseline", "centered", 0.0, "hinge", 0.0),
    Condition("centered_hinge", "centered", GUIDANCE_SCALE, "hinge", 0.0),
    Condition(
        "centered_huber_hinge",
        "centered",
        GUIDANCE_SCALE,
        "huber_hinge",
        HUBER_SMOOTHING,
    ),
    Condition("offset_baseline", "offset", 0.0, "hinge", 0.0),
    Condition("offset_hinge", "offset", GUIDANCE_SCALE, "hinge", 0.0),
    Condition(
        "offset_huber_hinge",
        "offset",
        GUIDANCE_SCALE,
        "huber_hinge",
        HUBER_SMOOTHING,
    ),
)


def progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] {message}", flush=True)


def frozen_offset_layouts(pair_count: int) -> list[Layout]:
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    rng = np.random.default_rng(LAYOUT_SEED)
    layouts = []
    for pair_index in range(pair_count):
        center_x = float(rng.uniform(*OFFSET_X_RANGE))
        center_abs_y = float(rng.uniform(*OFFSET_ABS_Y_RANGE))
        for mirror_index, sign in enumerate((1.0, -1.0)):
            layouts.append(
                Layout(
                    layout_id=f"pair_{pair_index:02d}_{'upper' if sign > 0 else 'lower'}",
                    policy_seed=POLICY_SEED_START + 2 * pair_index + mirror_index,
                    obstacle_center=(center_x, sign * center_abs_y),
                )
            )
    return layouts


def layouts_for_mode(offset_layouts: list[Layout], center_mode: str) -> list[Layout]:
    if center_mode == "offset":
        return offset_layouts
    if center_mode == "centered":
        return [
            Layout(layout.layout_id, layout.policy_seed, (0.0, 0.0))
            for layout in offset_layouts
        ]
    raise ValueError(f"Unknown center mode: {center_mode}")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=True, sort_keys=True) + "\n")
        stream.flush()


def condition_summary(rows: list[dict]) -> dict:
    result = summarize(rows)
    result.update(
        {
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_backward_distance": float(
                np.mean([row["backward_distance"] for row in rows])
            ),
            "mean_x_direction_reversals": float(
                np.mean([row["x_direction_reversals"] for row in rows])
            ),
            "mean_path_total_variation": float(
                np.mean([row["path_total_variation"] for row in rows])
            ),
            "mean_guidance_activation_switches": float(
                np.mean([row["guidance_activation_switches"] for row in rows])
            ),
            "mean_guidance_update_norm_total_variation": float(
                np.mean(
                    [row["guidance_update_norm_total_variation"] for row in rows]
                )
            ),
        }
    )
    if all("horizon_outcomes" in row for row in rows):
        result["horizon_outcomes"] = {}
        for horizon in HORIZON_MILESTONES:
            if str(horizon) not in rows[0]["horizon_outcomes"]:
                continue
            outcomes = [row["horizon_outcomes"][str(horizon)] for row in rows]
            count = len(outcomes)
            result["horizon_outcomes"][str(horizon)] = {
                "task_sr": sum(item["success"] for item in outcomes) / count,
                "safe_sr": sum(item["safe_success"] for item in outcomes) / count,
                "cr": sum(item["collision_any"] for item in outcomes) / count,
                "ncr": sum(item["ncr"] for item in outcomes) / count,
            }
    return result


def paired_comparison(reference: list[dict], candidate: list[dict]) -> dict:
    reference_by_layout = {row["layout_id"]: row for row in reference}
    candidate_by_layout = {row["layout_id"]: row for row in candidate}
    if reference_by_layout.keys() != candidate_by_layout.keys():
        raise ValueError("Paired layout IDs do not match")
    task_failure_to_success = 0
    task_success_to_failure = 0
    safe_failure_to_success = 0
    safe_success_to_failure = 0
    for layout_id, before in reference_by_layout.items():
        after = candidate_by_layout[layout_id]
        task_failure_to_success += int(not before["success"] and after["success"])
        task_success_to_failure += int(before["success"] and not after["success"])
        safe_before = before["safe_success"]
        safe_after = after["safe_success"]
        safe_failure_to_success += int(not safe_before and safe_after)
        safe_success_to_failure += int(safe_before and not safe_after)
    return {
        "task_failure_to_success": task_failure_to_success,
        "task_success_to_failure": task_success_to_failure,
        "task_exact_mcnemar_p": exact_mcnemar_p(
            task_failure_to_success, task_success_to_failure
        ),
        "safe_failure_to_success": safe_failure_to_success,
        "safe_success_to_failure": safe_success_to_failure,
        "safe_exact_mcnemar_p": exact_mcnemar_p(
            safe_failure_to_success, safe_success_to_failure
        ),
        "mean_backward_distance_change": float(
            np.mean(
                [
                    candidate_by_layout[key]["backward_distance"]
                    - reference_by_layout[key]["backward_distance"]
                    for key in reference_by_layout
                ]
            )
        ),
        "mean_x_direction_reversal_change": float(
            np.mean(
                [
                    candidate_by_layout[key]["x_direction_reversals"]
                    - reference_by_layout[key]["x_direction_reversals"]
                    for key in reference_by_layout
                ]
            )
        ),
    }


def build_contract(
    checkpoint: Path,
    environment: str,
    pair_count: int,
    layouts: list[Layout],
    horizon: int = HORIZON,
) -> dict:
    contract = {
        "checkpoint": str(checkpoint.resolve()),
        "environment": environment,
        "horizon": horizon,
        "guidance_scale": GUIDANCE_SCALE,
        "safety_distance": SAFETY_DISTANCE,
        "huber_smoothing": HUBER_SMOOTHING,
        "layout_seed": LAYOUT_SEED,
        "offset_x_range": list(OFFSET_X_RANGE),
        "offset_abs_y_range": list(OFFSET_ABS_Y_RANGE),
        "offset_pair_count": pair_count,
        "offset_layouts": [asdict(layout) for layout in layouts],
        "conditions": [asdict(condition) for condition in CONDITIONS],
    }
    if horizon != 400:
        contract["horizon_milestones"] = [
            value for value in HORIZON_MILESTONES if value <= horizon
        ]
    # Normalize tuples exactly as they will appear after the JSON contract is
    # reloaded, so resumability comparisons are stable across processes.
    return json.loads(json.dumps(contract))


def write_report(path: Path, contract: dict, summaries: dict, comparisons: dict) -> None:
    lines = [
        "# Empty2D smooth-guidance horizon ablation",
        "",
        (
            f"Circle/square geometry: `{contract['environment']}`; H={contract['horizon']}; "
            f"rho={GUIDANCE_SCALE:g}; Q={SAFETY_DISTANCE:g}; "
            f"Huber epsilon={HUBER_SMOOTHING:g}."
        ),
        "",
        "| Condition | Task SR | Safe SR | CR | NCR | Steps | Backward distance | X reversals | Path TV | Activation switches | Clip rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summaries[condition.name]
        lines.append(
            "| {name} | {task:.2f} | {safe:.2f} | {cr:.2f} | {ncr:.2f} | "
            "{steps:.1f} | {back:.3f} | {reversals:.1f} | {tv:.3f} | "
            "{switches:.1f} | {clip:.3f} |".format(
                name=condition.name,
                task=row["task_sr"],
                safe=row["safe_sr"],
                cr=row["cr"],
                ncr=row["ncr"],
                steps=row["mean_steps"],
                back=row["mean_backward_distance"],
                reversals=row["mean_x_direction_reversals"],
                tv=row["mean_path_total_variation"],
                switches=row["mean_guidance_activation_switches"],
                clip=row["mean_clip_rate"],
            )
        )
    if "horizon_milestones" in contract:
        lines.extend(
            [
                "",
                "## Horizon first-passage outcomes",
                "",
                "| Condition | Horizon | Task SR | Safe SR | CR | NCR |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in CONDITIONS:
            for horizon, outcome in summaries[condition.name][
                "horizon_outcomes"
            ].items():
                lines.append(
                    "| {name} | {horizon} | {task:.2f} | {safe:.2f} | {cr:.2f} | {ncr:.2f} |".format(
                        name=condition.name,
                        horizon=horizon,
                        task=outcome["task_sr"],
                        safe=outcome["safe_sr"],
                        cr=outcome["cr"],
                        ncr=outcome["ncr"],
                    )
                )
    lines.extend(["", "## Paired comparisons", ""])
    for name, comparison in comparisons.items():
        lines.append(
            "- `{}`: task F→S/S→F = {}/{}, safe F→S/S→F = {}/{}, "
            "mean Δ backward distance = {:.3f}, mean Δ x reversals = {:.2f}.".format(
                name,
                comparison["task_failure_to_success"],
                comparison["task_success_to_failure"],
                comparison["safe_failure_to_success"],
                comparison["safe_success_to_failure"],
                comparison["mean_backward_distance_change"],
                comparison["mean_x_direction_reversal_change"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--environment", choices=("circle", "square"), default="circle")
    parser.add_argument("--offset-pairs", type=int, default=10)
    parser.add_argument(
        "--horizon",
        type=int,
        choices=HORIZON_MILESTONES,
        default=HORIZON,
        help="Maximum rollout horizon; H600 records first-passage outcomes at H400/H500/H600.",
    )
    parser.add_argument(
        "--condition-group",
        choices=("all", "centered", "offset"),
        default="all",
        help="Run all conditions, or shard centered and offset conditions across GPUs.",
    )
    parser.add_argument(
        "--source-screen",
        type=Path,
        default=Path("outputs/empty2d_guidance/source_screen.json"),
    )
    parser.add_argument(
        "--device",
        help="Explicit torch device such as cuda:0 (default: cuda if available).",
    )
    args = parser.parse_args()

    source_screen = json.loads(args.source_screen.read_text())
    if not source_screen["passed"]:
        raise RuntimeError("Source gate has not passed")
    environment_class = {
        "circle": EnvEmpty2DExtraSphere,
        "square": EnvEmpty2DExtraSquare,
    }[args.environment]
    offset_layouts = frozen_offset_layouts(args.offset_pairs)
    for layout in offset_layouts:
        environment = environment_class(
            horizon=args.horizon, obstacle_center=layout.obstacle_center
        )
        if not environment.segment_collides(
            np.asarray([-0.875, 0.0]), np.asarray([0.875, 0.0])
        ):
            raise RuntimeError(
                f"Frozen layout {layout.layout_id} does not obstruct the direct path"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    contract = build_contract(
        args.checkpoint,
        args.environment,
        args.offset_pairs,
        offset_layouts,
        horizon=args.horizon,
    )
    contract_path = args.out_dir / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError(f"Existing contract does not match requested run: {contract_path}")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    progress(f"loading epoch-100 policy on {device}")
    policy, _ = load_guidable_policy(args.checkpoint, device)
    rows_by_condition = {}
    selected_conditions = tuple(
        condition
        for condition in CONDITIONS
        if args.condition_group == "all"
        or condition.center_mode == args.condition_group
    )
    for condition in selected_conditions:
        raw_path = args.out_dir / f"{condition.name}.jsonl"
        rows = read_jsonl(raw_path)
        completed = {row["layout_id"] for row in rows}
        layouts = layouts_for_mode(offset_layouts, condition.center_mode)
        for index, layout in enumerate(layouts, start=1):
            if layout.layout_id in completed:
                progress(
                    f"{condition.name} {index}/{len(layouts)} {layout.layout_id} resumed"
                )
                continue
            row = rollout(
                policy,
                environment_name=args.environment,
                policy_seed=layout.policy_seed,
                guidance_scale=condition.guidance_scale,
                safety_distance=SAFETY_DISTANCE,
                horizon=args.horizon,
                obstacle_center=layout.obstacle_center,
                obstacle_cost_type=condition.obstacle_cost_type,
                obstacle_cost_smoothing=condition.obstacle_cost_smoothing,
                milestone_horizons=tuple(
                    value for value in HORIZON_MILESTONES if value <= args.horizon
                ) if args.horizon > 400 else (),
            )
            row.update(
                {
                    "condition": condition.name,
                    "center_mode": condition.center_mode,
                    "layout_id": layout.layout_id,
                }
            )
            append_jsonl(raw_path, row)
            rows.append(row)
            progress(
                f"{condition.name} {index}/{len(layouts)} {layout.layout_id} "
                f"success={int(row['success'])} collision={int(row['collision_any'])} "
                f"steps={row['steps']} reversals={row['x_direction_reversals']}"
            )
        rows_by_condition[condition.name] = sorted(
            rows, key=lambda row: row["layout_id"]
        )

    if args.condition_group != "all":
        progress(
            f"{args.condition_group} shard complete; rerun with --condition-group all "
            "after both shards finish to aggregate the frozen result"
        )
        return

    summaries = {
        name: condition_summary(rows) for name, rows in rows_by_condition.items()
    }
    comparisons = {}
    for mode in ("centered", "offset"):
        baseline = rows_by_condition[f"{mode}_baseline"]
        hinge = rows_by_condition[f"{mode}_hinge"]
        huber = rows_by_condition[f"{mode}_huber_hinge"]
        comparisons[f"{mode}_hinge_vs_baseline"] = paired_comparison(
            baseline, hinge
        )
        comparisons[f"{mode}_huber_vs_baseline"] = paired_comparison(
            baseline, huber
        )
        comparisons[f"{mode}_huber_vs_hinge"] = paired_comparison(hinge, huber)

    result = {
        "contract": contract,
        "summaries": summaries,
        "comparisons": comparisons,
    }
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n"
    )
    write_report(args.out_dir / "report.md", contract, summaries, comparisons)
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
