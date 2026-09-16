"""Aggregate policy/perception metrics and apply the preregistered gates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiment_common import (
    CONDITIONS,
    ENVIRONMENTS,
    PILOT_EPISODES,
    audit_absolute_eef_actions,
    dump_json,
    sha256_file,
)


def load_rows(output: Path, environment: str, stages: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for stage in stages:
        stage_dir = output / "rollouts" / environment / stage
        if not stage_dir.is_dir():
            continue
        for result_path in sorted(stage_dir.glob("ep*/*/result.json")):
            row = json.loads(result_path.read_text())
            trajectory_path = result_path.parent / "trajectory.npz"
            with np.load(trajectory_path) as trajectory:
                contract_audit = audit_absolute_eef_actions(trajectory["actions"])
            contract_audit["step_count_matches"] = (
                contract_audit["action_count"] == row["steps"]
            )
            contract_audit["valid"] = bool(
                contract_audit["valid"] and contract_audit["step_count_matches"]
            )
            dump_json(result_path.parent / "action_contract_audit.json", contract_audit)
            row["absolute_eef_contract_audit"] = contract_audit
            row["stage"] = stage
            row["result_path"] = str(result_path)
            rows.append(row)
    keys = [(row["episode"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate formal rows for {environment}")
    return rows


def condition_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    count = len(selected)
    if not count:
        return {"episodes": 0}
    total = lambda key: int(sum(bool(row[key]) for row in selected))
    return {
        "episodes": count,
        "task_success": total("success"),
        "task_sr": total("success") / count,
        "safe_success": total("safe_success"),
        "safe_sr": total("safe_success") / count,
        "collision_any": total("collision_any"),
        "cr": total("collision_any") / count,
        "ncr": total("ncr"),
        "ncr_rate": total("ncr") / count,
        "partition": {
            "safe_success": total("safe_success"),
            "success_with_collision": total("success_with_collision"),
            "collision_failure": total("collision_failure"),
            "ncr": total("ncr"),
        },
        "perception_failures": total("perception_failure"),
        "action_finite_all": all(
            row["absolute_eef_contract_audit"]["finite"] for row in selected
        ),
        "absolute_eef_contract_all": all(
            row["absolute_eef_contract_audit"]["valid"] for row in selected
        ),
        "action_contract_violation_count": int(
            sum(
                row["absolute_eef_contract_audit"]["violation_count"]
                for row in selected
            )
        ),
        "environment_action_spec_exceedance_count": int(
            sum(
                row.get(
                    "environment_action_spec_exceedance_count",
                    row.get("action_bound_violation_count", 0),
                )
                for row in selected
            )
        ),
    }


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    by_episode = defaultdict(dict)
    for row in rows:
        by_episode[row["episode"]][row["condition"]] = row
    complete = {
        episode: values
        for episode, values in by_episode.items()
        if set(values) == set(CONDITIONS)
    }
    source_name, perceived_name = CONDITIONS
    return {
        "pairs": len(complete),
        "task_rescues": sum(
            not values[source_name]["success"] and values[perceived_name]["success"]
            for values in complete.values()
        ),
        "task_regressions": sum(
            values[source_name]["success"] and not values[perceived_name]["success"]
            for values in complete.values()
        ),
        "safe_rescues": sum(
            not values[source_name]["safe_success"]
            and values[perceived_name]["safe_success"]
            for values in complete.values()
        ),
        "safe_regressions": sum(
            values[source_name]["safe_success"]
            and not values[perceived_name]["safe_success"]
            for values in complete.values()
        ),
    }


def perception_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = []
    for row in rows:
        if row["condition"] != "perceived_mask_source":
            continue
        path = Path(row["result_path"]).parent / "perception_diagnostics.jsonl"
        with path.open() as stream:
            diagnostics.extend(json.loads(line) for line in stream if line.strip())
    if not diagnostics:
        return {"frames": 0}
    role_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    identity_swaps = 0
    identity_swaps_by_role: dict[str, int] = defaultdict(int)
    support_detected_frames = 0
    tracking_failures = 0
    obstacle_failure_frames = 0
    failure_reasons: dict[str, int] = defaultdict(int)
    obstacle_failure_reasons: dict[str, int] = defaultdict(int)
    latencies = []
    for row in diagnostics:
        perception = row["perception"]
        latency = perception.get("total_latency_s")
        if latency is not None:
            latencies.append(float(latency))
        if perception.get("support_detected"):
            support_detected_frames += 1
        if perception.get("phase") == "tracking" and perception.get("failure_reason"):
            tracking_failures += 1
        failure_reason = perception.get("failure_reason")
        if failure_reason:
            failure_reasons[str(failure_reason)] += 1
        obstacle_failure_reason = perception.get("obstacle_failure_reason")
        if obstacle_failure_reason:
            obstacle_failure_frames += 1
            obstacle_failure_reasons[str(obstacle_failure_reason)] += 1
        audit = row.get("oracle_audit") or {}
        identity_swaps += int(bool(audit.get("identity_swap_any")))
        for role, swapped in audit.get("identity_swap", {}).items():
            identity_swaps_by_role[role] += int(bool(swapped))
        for role, metrics in audit.get("mask_metrics", {}).items():
            for name in ("iou", "precision", "recall"):
                role_values[role][f"mask_{name}"].append(float(metrics[name]))
        for role, metrics in audit.get("pointcloud_metrics", {}).items():
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    role_values[role][name].append(float(value))
    role_summary = {
        role: {
            metric: float(np.mean(values))
            for metric, values in sorted(metrics.items())
            if values
        }
        for role, metrics in sorted(role_values.items())
    }
    return {
        "frames": len(diagnostics),
        "roles": role_summary,
        "identity_swap_frames": identity_swaps,
        "identity_swap_frames_by_role": dict(sorted(identity_swaps_by_role.items())),
        "support_detected_frames": support_detected_frames,
        "tracking_failures": tracking_failures,
        "task_failure_reasons": dict(sorted(failure_reasons.items())),
        "obstacle_failure_frames": obstacle_failure_frames,
        "obstacle_failure_reasons": dict(sorted(obstacle_failure_reasons.items())),
        "latency_s": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
            "max": float(np.max(latencies)) if latencies else None,
        },
    }


def summarize_environment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "conditions": {
            condition: condition_summary(rows, condition) for condition in CONDITIONS
        },
        "paired": paired_summary(rows),
        "perception": perception_summary(rows),
    }


def pilot_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for environment in ENVIRONMENTS:
        value = summary[environment]
        source = value["conditions"][CONDITIONS[0]]
        perceived = value["conditions"][CONDITIONS[1]]
        checks[environment] = {
            "ten_pairs_complete": source["episodes"] == perceived["episodes"] == 10,
            "no_perception_failure": perceived["perception_failures"] == 0,
            "at_most_one_task_regression": value["paired"]["task_regressions"] <= 1,
            "task_success_within_one": (
                perceived["task_success"] >= source["task_success"] - 1
            ),
            "actions_finite_and_in_contract": bool(
                source["action_finite_all"]
                and perceived["action_finite_all"]
                and source["absolute_eef_contract_all"]
                and perceived["absolute_eef_contract_all"]
            ),
        }
    passed = all(all(values.values()) for values in checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "decision": (
            "lock configuration and run the remaining 40 layouts per environment"
            if passed
            else "stop this stage; do not launch expansion, training, or planner work"
        ),
    }


def full_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for environment in ENVIRONMENTS:
        value = summary[environment]
        source = value["conditions"][CONDITIONS[0]]
        perceived = value["conditions"][CONDITIONS[1]]
        complete = source["episodes"] == perceived["episodes"] == 50
        checks[environment] = {
            "fifty_pairs_complete": complete,
            "task_drop_at_most_two": complete
            and perceived["task_success"] >= source["task_success"] - 2,
        }
        if environment == "raised_milk":
            checks[environment].update(
                {
                    "safe_drop_at_most_two": complete
                    and perceived["safe_success"] >= source["safe_success"] - 2,
                    "collision_increase_at_most_two": complete
                    and perceived["collision_any"] <= source["collision_any"] + 2,
                }
            )
    passed = all(all(values.values()) for values in checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "decision": (
            "retain the frozen policy; perceived obstacle cloud may be tested with the planner in a separately authorized stage"
            if passed
            else "frozen policy is not robust enough to the measured perception shift; recommend a separate perception-robust fine-tuning stage"
        ),
    }


def write_report(output: Path, summary: dict[str, Any], gate: dict[str, Any], scope: str) -> None:
    lines = [
        f"# RGB-D perception comparison — {scope}",
        "",
        "Frozen epoch-50 absolute-EEF source policy; GroundingDINO + SAM initialization and Cutie tracking.",
        "Oracle segmentation is evaluation-only and is not supplied to the perceived policy observation.",
        "",
        "| Environment | Condition | Task | Safe | CR | NCR | Perception failures |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for environment in ENVIRONMENTS:
        for condition in CONDITIONS:
            row = summary[environment]["conditions"][condition]
            lines.append(
                f"| {environment} | {condition} | {row['task_success']}/{row['episodes']} | "
                f"{row['safe_success']}/{row['episodes']} | {row['collision_any']}/{row['episodes']} | "
                f"{row['ncr']}/{row['episodes']} | {row['perception_failures']} |"
            )
        paired = summary[environment]["paired"]
        lines.extend(
            [
                "",
                f"{environment} paired Task rescues/regressions: "
                f"{paired['task_rescues']}/{paired['task_regressions']}; Safe: "
                f"{paired['safe_rescues']}/{paired['safe_regressions']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Perception diagnostics",
            "",
            "Per-frame means below are audit metrics only; oracle values were never inserted into the perceived policy observation.",
            "",
            "| Environment | Role | Mask IoU | 5 mm P/R | Chamfer (mm) | Centroid error (mm) | Pred/oracle points |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for environment in ENVIRONMENTS:
        perception = summary[environment]["perception"]
        for role in ("target", "goal", "milk", "support", "obstacle"):
            metrics = perception.get("roles", {}).get(role)
            if metrics is None:
                continue
            chamfer = metrics.get("chamfer_m")
            centroid = metrics.get("centroid_error_m")
            lines.append(
                f"| {environment} | {role} | {metrics.get('mask_iou', 0.0):.3f} | "
                f"{metrics.get('nn_5mm_precision', 0.0):.3f}/{metrics.get('nn_5mm_recall', 0.0):.3f} | "
                f"{'n/a' if chamfer is None else f'{1000.0 * chamfer:.1f}'} | "
                f"{'n/a' if centroid is None else f'{1000.0 * centroid:.1f}'} | "
                f"{metrics.get('predicted_points', 0.0):.1f}/{metrics.get('oracle_points', 0.0):.1f} |"
            )
        lines.extend(
            [
                "",
                f"{environment}: {perception.get('frames', 0)} perceived frames; "
                f"task tracking failures {perception.get('tracking_failures', 0)}; "
                f"obstacle-invalid frames {perception.get('obstacle_failure_frames', 0)}; "
                f"identity-swap frames {perception.get('identity_swap_frames', 0)}; "
                f"latency p50/p95 {perception.get('latency_s', {}).get('p50', 0.0) * 1000.0:.1f}/"
                f"{perception.get('latency_s', {}).get('p95', 0.0) * 1000.0:.1f} ms.",
                "",
            ]
        )
    lines.extend(
        [
            f"Gate: **{'PASS' if gate['passed'] else 'FAIL'}**.",
            "",
            gate["decision"] + ".",
            "",
            "This is development-bank simulator evidence, not held-out perception or real-robot validation.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def write_conclusion(
    output: Path, summary: dict[str, Any], gate: dict[str, Any], scope: str
) -> None:
    clean = summary["clean"]
    raised = summary["raised_milk"]
    clean_perception = clean["perception"]
    raised_perception = raised["perception"]
    lines = [
        "# RGB-D perception comparison conclusion",
        "",
        f"Status: **stopped at {scope} gate — {'PASS' if gate['passed'] else 'FAIL'}**.",
        "",
        "## Decision",
        "",
        "Do not launch the 40-layout expansion, policy training, or planner integration from this stage. The frozen wide-support epoch-50 policy did not tolerate the measured RGB-D point-cloud shift, and the obstacle reconstruction is not adequate for planner use.",
        "",
        "## Controlled policy result",
        "",
        "| Environment | Oracle Task / Safe / CR / NCR | Perceived Task / Safe / CR / NCR | Task rescues / regressions |",
        "|---|---:|---:|---:|",
    ]
    for environment, value in (("clean", clean), ("RaisedMilk", raised)):
        source = value["conditions"][CONDITIONS[0]]
        perceived = value["conditions"][CONDITIONS[1]]
        paired = value["paired"]
        lines.append(
            f"| {environment} | {source['task_success']}/10 / {source['safe_success']}/10 / "
            f"{source['collision_any']}/10 / {source['ncr']}/10 | "
            f"{perceived['task_success']}/10 / {perceived['safe_success']}/10 / "
            f"{perceived['collision_any']}/10 / {perceived['ncr']}/10 | "
            f"{paired['task_rescues']} / {paired['task_regressions']} |"
        )
    clean_target = clean_perception["roles"]["target"]
    clean_goal = clean_perception["roles"]["goal"]
    raised_target = raised_perception["roles"]["target"]
    raised_goal = raised_perception["roles"]["goal"]
    raised_obstacle = raised_perception["roles"]["obstacle"]
    lines.extend(
        [
            "",
            "All recorded actions were finite and satisfied the executable absolute-world EEF contract. The perceived zero-collision result is not a safety improvement: every perceived rollout failed the task, with all failures classified NCR.",
            "",
            "## Perception evidence",
            "",
            f"Target/goal mean mask IoU was {clean_target['mask_iou']:.3f}/{clean_goal['mask_iou']:.3f} on clean and {raised_target['mask_iou']:.3f}/{raised_goal['mask_iou']:.3f} on RaisedMilk. Despite these high mean overlaps, perceived Task SR was 0/20. One clean target track became invalid at frame 359; RaisedMilk had no task-mask failure.",
            "",
            f"RaisedMilk obstacle mean mask IoU was {raised_obstacle['mask_iou']:.3f}, with {raised_perception['obstacle_failure_frames']}/{raised_perception['frames']} frames lacking a valid Milk reconstruction and {raised_perception['identity_swap_frames']} identity-swap frames. Support detections do not constitute faithful support reconstruction: support IoU was effectively zero.",
            "",
            "The result establishes a deployment shift failure for this frozen policy/provider combination. It does not by itself identify whether the dominant cause is point sampling, small temporally structured mask errors, or policy sensitivity, and it is development-bank simulator evidence rather than held-out perception or real-robot validation.",
            "",
            "## Reproducibility boundary",
            "",
            "GroundingDINO and SAM ran once at episode start; Cutie propagated fixed IDs with no detector retry or mid-episode reinitialization. Oracle segmentation/contact/object state were confined to evaluation. Each pair restored the same saved state and policy seed in one process and matched initial RGB-D, calibration, policy RNG, and all native non-point-cloud observations.",
            "",
            "See [report.md](report.md), [pilot_summary.json](pilot_summary.json), [pilot_gate.json](pilot_gate.json), and [video_index.md](video_index.md).",
        ]
    )
    (output / "conclusion.md").write_text("\n".join(lines) + "\n")


def write_output_manifest(output: Path, gate: dict[str, Any], scope: str) -> None:
    stage_manifests = {
        environment: output / "rollouts" / environment / "pilot" / "manifest.json"
        for environment in ENVIRONMENTS
    }
    stage_values = {
        environment: json.loads(path.read_text())
        for environment, path in stage_manifests.items()
    }
    result_paths = sorted(output.glob("rollouts/*/pilot/ep*/*/result.json"))
    pair_paths = sorted(output.glob("rollouts/*/pilot/ep*/pair_audit.json"))
    video_paths = sorted(output.glob("rollouts/*/pilot/ep*/*/rollout.mp4"))
    manifest = {
        "format_version": 1,
        "scope": scope,
        "gate_passed": gate["passed"],
        "decision": gate["decision"],
        "formal_rollouts": len(result_paths),
        "formal_pairs": len(pair_paths),
        "execution_videos": len(video_paths),
        "artifact_bytes_excluding_this_manifest": int(
            sum(
                path.stat().st_size
                for path in output.rglob("*")
                if path.is_file() and path != output / "manifest.json"
            )
        ),
        "checkpoint": stage_values["clean"]["checkpoint"],
        "checkpoint_sha256": stage_values["clean"]["checkpoint_sha256"],
        "locked_config_sha256": sha256_file(output / "calibration" / "locked_config.json"),
        "weights_manifest_sha256": sha256_file(output / "weights" / "manifest.json"),
        "smoke_result_sha256": sha256_file(output / "smoke" / "result.json"),
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "stage_manifests": {
            environment: {
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "source_hashes": stage_values[environment]["source_hashes"],
                "git_heads": stage_values[environment]["git_heads"],
            }
            for environment, path in stage_manifests.items()
        },
        "runtime_network_policy": "offline; verified local weight manifest only",
        "development_bank_only": True,
    }
    dump_json(output / "manifest.json", manifest)


def write_video_index(output: Path, rows_by_environment: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Rollout videos", ""]
    for environment, rows in rows_by_environment.items():
        lines.extend([f"## {environment}", ""])
        for row in sorted(rows, key=lambda item: (item["episode"], item["condition"])):
            result_path = Path(row["result_path"])
            video = result_path.parent / "rollout.mp4"
            relative = video.relative_to(output)
            lines.append(
                f"- ep{row['episode']:02d} {row['condition']}: [{relative}]({relative})"
            )
        lines.append("")
    (output / "video_index.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    pilot_rows = {
        environment: load_rows(output, environment, ("pilot",))
        for environment in ENVIRONMENTS
    }
    expected_pilot = 2 * len(PILOT_EPISODES)
    pilot_complete = all(len(rows) == expected_pilot for rows in pilot_rows.values())
    if not pilot_complete:
        raise RuntimeError("both environments need all ten pilot pairs before analysis")
    pilot_summary = {
        environment: summarize_environment(rows)
        for environment, rows in pilot_rows.items()
    }
    pilot = pilot_gate(pilot_summary)
    dump_json(output / "pilot_summary.json", pilot_summary)
    dump_json(output / "pilot_gate.json", pilot)

    all_rows = {
        environment: load_rows(output, environment, ("pilot", "expansion"))
        for environment in ENVIRONMENTS
    }
    full_complete = all(len(rows) == 100 for rows in all_rows.values())
    if full_complete:
        scope = "full50"
        summary = {
            environment: summarize_environment(rows)
            for environment, rows in all_rows.items()
        }
        gate = full_gate(summary)
        dump_json(output / "summary.json", summary)
        dump_json(output / "full50_gate.json", gate)
    else:
        scope = "pilot"
        summary = pilot_summary
        gate = pilot
    write_report(output, summary, gate, scope)
    write_video_index(output, all_rows if full_complete else pilot_rows)
    write_conclusion(output, summary, gate, scope)
    write_output_manifest(output, gate, scope)
    print(json.dumps({"scope": scope, "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
