"""Aggregate sample-refine paired rollouts into JSON and Markdown reports."""

import argparse
import json
from pathlib import Path

import numpy as np


def mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


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
            (r["env"], r["condition"], r["initial_p0_regime"])
            for r in rows
            if r.get("initial_p0_regime") is not None
        }
    ):
        selected = [
            r for r in rows
            if r["env"] == env_name
            and r["condition"] == condition
            and r.get("initial_p0_regime") == regime
        ]
        regimes.append(
            {
                "env": env_name,
                "condition": condition,
                "initial_p0_regime": regime,
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


def markdown(summary, regimes):
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
        "", "## Initial P0-regime stratification", "",
        "This stratification uses candidate 0 at the first chunk, so the paired conditions share the same regime label.", "",
        "| env | condition | initial regime | n | Task SR | Safe SR | CR | NCR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in regimes:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["env"], row["condition"], row["initial_p0_regime"], row["episodes"],
                fmt(row["task_sr"]), fmt(row["safe_sr"]), fmt(row["cr"]), fmt(row["ncr"]),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.eval_dir / "episode_metrics.jsonl").read_text().splitlines() if line]
    summary, regimes = summarize(rows)
    (args.eval_dir / "summary.json").write_text(json.dumps({"summary": summary, "regimes": regimes}, indent=2) + "\n")
    (args.eval_dir / "report.md").write_text(markdown(summary, regimes))
    print("wrote summary.json and report.md", flush=True)


if __name__ == "__main__":
    main()
