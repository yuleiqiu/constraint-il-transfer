"""Aggregate sample-refine paired rollouts into JSON and Markdown reports."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def outcome(row):
    for key in ("safe_success", "success_with_collision", "collision_failure", "ncr"):
        if row[key]:
            return key
    raise ValueError("Episode row has no four-way outcome")


def paired_transition_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["env"], row["seed"], row["episode"])].append(row)
    result = []
    for env_name in sorted({row["env"] for row in rows}):
        env_groups = [group for key, group in groups.items() if key[0] == env_name]
        for condition in ("sample_only", "refine_only", "sample_refine"):
            counts = Counter()
            for group in env_groups:
                baseline = next(row for row in group if row["condition"] == "baseline")
                compared = next(row for row in group if row["condition"] == condition)
                counts[(outcome(baseline), outcome(compared))] += 1
            for (before, after), count in sorted(counts.items()):
                result.append(
                    {
                        "env": env_name,
                        "condition": condition,
                        "baseline_outcome": before,
                        "condition_outcome": after,
                        "count": count,
                    }
                )
    return result


def gate_decisions(summary):
    decisions = []
    for env_name in sorted({row["env"] for row in summary}):
        baseline = next(
            row for row in summary if row["env"] == env_name and row["condition"] == "baseline"
        )
        method = next(
            row for row in summary if row["env"] == env_name and row["condition"] == "sample_refine"
        )
        task_gain = method["task_sr"] - baseline["task_sr"]
        ncr_delta = method["ncr"] - baseline["ncr"]
        if task_gain <= 0:
            decision = "do_not_advance"
            reason = "sample_refine did not improve paired Task SR"
        elif ncr_delta >= task_gain:
            decision = "do_not_advance"
            reason = "Task SR gain was matched or exceeded by increased NCR"
        else:
            decision = "advance_to_held_out"
            reason = "sample_refine passed the development Task SR and NCR exchange gate"
        decisions.append(
            {
                "env": env_name,
                "decision": decision,
                "reason": reason,
                "task_sr_gain": task_gain,
                "safe_sr_gain": method["safe_sr"] - baseline["safe_sr"],
                "cr_delta": method["cr"] - baseline["cr"],
                "ncr_delta": ncr_delta,
            }
        )
    return decisions


def summarize(rows):
    summaries = []
    for env_name, condition in sorted({(r["env"], r["condition"]) for r in rows}):
        selected = [r for r in rows if r["env"] == env_name and r["condition"] == condition]
        trigger = sum(r["refine_trigger_count"] for r in selected)
        accepted = sum(r["refine_accept_count"] for r in selected)
        summaries.append(
            {
                "env": env_name,
                "condition": condition,
                "episodes": len(selected),
                "task_sr": mean(selected, "success"),
                "safe_sr": mean(selected, "safe_success"),
                "cr": mean(selected, "collision_any"),
                "ncr": mean(selected, "ncr"),
                "safe_success_count": int(sum(r["safe_success"] for r in selected)),
                "success_with_collision_count": int(sum(r["success_with_collision"] for r in selected)),
                "collision_failure_count": int(sum(r["collision_failure"] for r in selected)),
                "ncr_count": int(sum(r["ncr"] for r in selected)),
                "inference_time_ms_mean": mean(selected, "inference_time_ms_mean"),
                "candidate_coverage": mean(selected, "positive_safe_candidate_coverage"),
                "selected_nonzero_rate": mean(selected, "selected_nonzero_rate"),
                "refine_trigger_count": trigger,
                "refine_accept_count": accepted,
                "refine_fallback_count": sum(r["refine_fallback_count"] for r in selected),
                "refine_accept_rate": accepted / trigger if trigger else None,
                "action_clip_count": int(sum(r["action_clip_count"] for r in selected)),
            }
        )
    baseline_times = {
        row["env"]: row["inference_time_ms_mean"]
        for row in summaries
        if row["condition"] == "baseline"
    }
    for row in summaries:
        baseline_time = baseline_times.get(row["env"])
        row["inference_overhead_x"] = (
            row["inference_time_ms_mean"] / baseline_time
            if baseline_time and row["inference_time_ms_mean"] is not None
            else None
        )

    regimes = []
    for env_name, condition, regime in sorted(
        {
            (r["env"], r["condition"], r["baseline_p0_critical_regime"])
            for r in rows
            if r.get("baseline_p0_critical_regime") is not None
        }
    ):
        selected = [
            r for r in rows
            if r["env"] == env_name
            and r["condition"] == condition
            and r.get("baseline_p0_critical_regime") == regime
        ]
        regimes.append(
            {
                "env": env_name,
                "condition": condition,
                "baseline_p0_critical_regime": regime,
                "episodes": len(selected),
                "task_sr": mean(selected, "success"),
                "safe_sr": mean(selected, "safe_success"),
                "cr": mean(selected, "collision_any"),
                "ncr": mean(selected, "ncr"),
            }
        )
    return summaries, regimes


def fmt(value):
    return "—" if value is None else "{:.3f}".format(value)


def markdown(summary, regimes, transitions=None, decisions=None):
    lines = [
        "# Sample-Refine Paired Rollout Report",
        "",
        "The four-way partition is safe success / success with collision / collision failure / collision-free non-completion (NCR).",
        "",
        "| env | condition | n | Task SR | Safe SR | CR | NCR | safe / succ+coll / coll fail / NCR | ms/chunk | overhead | coverage | refine accept/fallback | clips |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} / {} / {} / {} | {} | {}x | {} | {}/{} | {} |".format(
                row["env"], row["condition"], row["episodes"], fmt(row["task_sr"]),
                fmt(row["safe_sr"]), fmt(row["cr"]), fmt(row["ncr"]),
                row["safe_success_count"], row["success_with_collision_count"],
                row["collision_failure_count"], row["ncr_count"],
                fmt(row["inference_time_ms_mean"]), fmt(row["inference_overhead_x"]),
                fmt(row["candidate_coverage"]),
                row["refine_accept_count"], row["refine_fallback_count"], row["action_clip_count"],
            )
        )
    lines.extend([
        "", "## Baseline P0 critical-regime stratification", "",
        "Each matched layout uses the baseline episode's canonical P0 critical chunk as a shared label across all four conditions.", "",
        "| env | condition | baseline critical regime | n | Task SR | Safe SR | CR | NCR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in regimes:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["env"], row["condition"], row["baseline_p0_critical_regime"], row["episodes"],
                fmt(row["task_sr"]), fmt(row["safe_sr"]), fmt(row["cr"]), fmt(row["ncr"]),
            )
        )
    if transitions is not None:
        lines.extend([
            "", "## Paired outcome transitions", "",
            "| env | condition | baseline outcome | condition outcome | count |",
            "|---|---|---|---|---:|",
        ])
        for row in transitions:
            lines.append(
                "| {env} | {condition} | {baseline_outcome} | {condition_outcome} | {count} |".format(**row)
            )
    if decisions is not None:
        lines.extend(["", "## Development gate", ""])
        for row in decisions:
            lines.append(
                "- **{env}: `{decision}`.** {reason}. Task SR Δ={task_sr_gain:+.3f}, Safe SR Δ={safe_sr_gain:+.3f}, CR Δ={cr_delta:+.3f}, NCR Δ={ncr_delta:+.3f}.".format(**row)
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.eval_dir / "episode_metrics.jsonl").read_text().splitlines() if line]
    summary, regimes = summarize(rows)
    transitions = paired_transition_summary(rows)
    decisions = gate_decisions(summary)
    (args.eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "regimes": regimes,
                "paired_transitions": transitions,
                "decisions": decisions,
            },
            indent=2,
        )
        + "\n"
    )
    (args.eval_dir / "report.md").write_text(
        markdown(summary, regimes, transitions, decisions)
    )
    print("wrote summary.json and report.md", flush=True)


if __name__ == "__main__":
    main()
