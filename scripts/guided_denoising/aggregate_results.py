"""Aggregate a paired guided-denoising pilot into comparison tables."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import paired_rollout_eval  # noqa: E402


def load_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def paired_summary(rows):
    summaries = []
    for env_name in sorted({row["env"] for row in rows}):
        env_rows = [row for row in rows if row["env"] == env_name]
        pairs = {}
        for row in env_rows:
            key = (row["seed"], row["episode"])
            pairs.setdefault(key, {})[row["condition"]] = row
        if any(set(pair) != {"baseline", "guided"} for pair in pairs.values()):
            raise ValueError("Incomplete baseline/guided pair in {}".format(env_name))
        if any(
            pair["baseline"]["initial_state_hash"]
            != pair["guided"]["initial_state_hash"]
            for pair in pairs.values()
        ):
            raise ValueError("Initial-state hash mismatch in {}".format(env_name))

        baseline = [pair["baseline"] for pair in pairs.values()]
        guided = [pair["guided"] for pair in pairs.values()]
        summaries.append(
            {
                "env": env_name,
                "pairs": len(pairs),
                "baseline_task_sr": float(np.mean([row["success"] for row in baseline])),
                "guided_task_sr": float(np.mean([row["success"] for row in guided])),
                "task_sr_delta": float(
                    np.mean(
                        [
                            int(pair["guided"]["success"])
                            - int(pair["baseline"]["success"])
                            for pair in pairs.values()
                        ]
                    )
                ),
                "baseline_cr": float(np.mean([row["collision_any"] for row in baseline])),
                "guided_cr": float(np.mean([row["collision_any"] for row in guided])),
                "cr_delta": float(
                    np.mean(
                        [
                            int(pair["guided"]["collision_any"])
                            - int(pair["baseline"]["collision_any"])
                            for pair in pairs.values()
                        ]
                    )
                ),
                "baseline_ncr": float(np.mean([row["ncr"] for row in baseline])),
                "guided_ncr": float(np.mean([row["ncr"] for row in guided])),
                "ncr_delta": float(
                    np.mean(
                        [
                            int(pair["guided"]["ncr"])
                            - int(pair["baseline"]["ncr"])
                            for pair in pairs.values()
                        ]
                    )
                ),
                "collision_failure_to_ncr": int(
                    sum(
                        pair["baseline"]["collision_failure"]
                        and pair["guided"]["ncr"]
                        for pair in pairs.values()
                    )
                ),
                "baseline_success_to_guided_failure": int(
                    sum(
                        pair["baseline"]["success"]
                        and not pair["guided"]["success"]
                        for pair in pairs.values()
                    )
                ),
                "baseline_failure_to_guided_success": int(
                    sum(
                        not pair["baseline"]["success"]
                        and pair["guided"]["success"]
                        for pair in pairs.values()
                    )
                ),
            }
        )
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, aggregate_rows, paired_rows, manifest):
    aggregate_by_key = {
        (row["env"], row["condition"]): row for row in aggregate_rows
    }
    lines = [
        "# Guided Denoising Paired Pilot",
        "",
        "This is a small matched-rollout study before a full evaluation matrix. Each pair runs baseline and guided policies from the same initial simulator state and policy random seed; only guidance differs.",
        "",
        "- **Baseline:** guidance scale `0` (no guidance).",
        "- **Guided:** guidance scale `{}`. This is the base physical waypoint-push coefficient $\\lambda$; the actual per-step displacement also includes the DDIM timestep factor $1 / \\sqrt{{\\bar{{\\alpha}}_{{t-1}}}}$, so this value is not a fixed displacement at every denoising step.".format(
            manifest["guidance_scale"]
        ),
        "- **Clearance margin:** `{}` m (`{:g}` cm). The deployment cost expands each oracle obstacle's circular XY radius by this amount before testing waypoint penetration. It is a cost boundary, not a claim that the executed robot maintains that clearance.".format(
            manifest["clearance_margin"], manifest["clearance_margin"] * 100.0
        ),
        "",
        "## Metric definitions",
        "",
        "- **Task SR (Task Success Rate):** fraction of episodes that complete the environment task, whether or not a distractor collision occurred.",
        "- **Safe SR (Safe Success Rate):** fraction of episodes that complete the task with no robot contact against any non-target distractor.",
        "- **CR (Collision Rate):** fraction of episodes with at least one robot-distractor contact, including both successful and failed episodes.",
        "- **NCR (collision-free Non-Completion Rate):** fraction of episodes that do not collide but still fail to complete the task.",
        "",
        "The final four columns below are mutually exclusive episode counts and sum to the number of episodes: safe success, success with collision, collision failure, and collision-free non-completion.",
        "",
        "For rates, `Task SR = safe success rate + success-with-collision rate`; a reduction in CR is useful only if it does not merely move collision failures into NCR.",
        "",
        "## Outcome metrics",
        "",
        "| environment | condition | Task SR | Safe SR | CR | NCR | safe success | success + collision | collision failure | NCR count |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for env_name in sorted({row["env"] for row in aggregate_rows}):
        for condition in ("baseline", "guided"):
            row = aggregate_by_key[(env_name, condition)]
            lines.append(
                "| {env} | {condition} | {task:.3f} | {safe:.3f} | {cr:.3f} | {ncr:.3f} | {ss} | {swc} | {cf} | {ncr_count} |".format(
                    env=env_name,
                    condition=condition,
                    task=row["task_sr"],
                    safe=row["safe_sr"],
                    cr=row["cr"],
                    ncr=row["ncr"],
                    ss=row["safe_success"],
                    swc=row["success_with_collision"],
                    cf=row["collision_failure"],
                    ncr_count=row["collision_free_non_completion"],
                )
            )

    lines.extend(
        [
            "",
            "## Paired changes",
            "",
            "All deltas are `guided - baseline`. The transition columns count matched initial states whose task outcome changed.",
            "",
            "| environment | pairs | Task SR delta | CR delta | NCR delta | collision failure -> NCR | baseline success -> failure | baseline failure -> success |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {env} | {pairs} | {task:+.3f} | {cr:+.3f} | {ncr:+.3f} | {to_ncr} | {lost} | {gained} |".format(
                env=row["env"],
                pairs=row["pairs"],
                task=row["task_sr_delta"],
                cr=row["cr_delta"],
                ncr=row["ncr_delta"],
                to_ncr=row["collision_failure_to_ncr"],
                lost=row["baseline_success_to_guided_failure"],
                gained=row["baseline_failure_to_guided_success"],
            )
        )
    lines.extend(
        [
            "",
            "## Guidance diagnostics",
            "",
            "`trigger rate` is the fraction of DDIM reverse steps whose predicted trajectory enters the deployment-cost boundary. The baseline can therefore have a nonzero trigger rate even though scale `0` applies no update. Update norms are measured in normalized action space; reconstruction errors compare predicted and executed EEF trajectories. `action clips` counts policy actions outside environment limits.",
            "",
            "| environment | condition | trigger rate | update norm mean | update norm max | reconstruction mean (cm) | reconstruction max (cm) | action clips |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for env_name in sorted({row["env"] for row in aggregate_rows}):
        for condition in ("baseline", "guided"):
            row = aggregate_by_key[(env_name, condition)]
            lines.append(
                "| {env} | {condition} | {trigger:.3f} | {update_mean:.3f} | {update_max:.3f} | {recon_mean:.3f} | {recon_max:.3f} | {clips} |".format(
                    env=env_name,
                    condition=condition,
                    trigger=row["guidance_trigger_rate"],
                    update_mean=row["guidance_update_norm_mean"],
                    update_max=row["guidance_update_norm_max"],
                    recon_mean=row["trajectory_reconstruction_error_mean_cm"],
                    recon_max=row["trajectory_reconstruction_error_max_cm"],
                    clips=row["action_clip_count"],
                )
            )
    lines.extend(
        [
            "",
            "Interpret CR reduction together with Task SR and NCR. A matching NCR increase is not evidence of improved task completion.",
        ]
    )
    if "decision" in manifest:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "`{}`: {}".format(
                    manifest["decision"], manifest.get("decision_note", "")
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.eval_dir / "episode_metrics.jsonl")
    manifest = json.loads((args.eval_dir / "manifest.json").read_text())
    aggregate_rows = paired_rollout_eval.aggregate(rows)
    paired_rows = paired_summary(rows)
    (args.eval_dir / "summary.json").write_text(
        json.dumps(aggregate_rows, indent=2) + "\n"
    )
    write_csv(args.eval_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(args.eval_dir / "paired_summary.csv", paired_rows)
    write_report(args.eval_dir / "report.md", aggregate_rows, paired_rows, manifest)
    print("wrote paired report in {}".format(args.eval_dir), flush=True)


if __name__ == "__main__":
    main()
