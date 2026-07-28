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
    max_timestep = manifest.get("max_guidance_timestep")
    timestep_description = (
        "at all configured diffusion timesteps"
        if max_timestep is None
        else "at diffusion timesteps `t <= {}`".format(max_timestep)
    )
    displacement_cap = manifest.get("max_waypoint_displacement_m")
    cap_description = (
        "without a per-waypoint reverse-step displacement cap"
        if displacement_cap is None
        else "with a per-waypoint reverse-step displacement cap of `{}` m".format(
            displacement_cap
        )
    )
    lines = [
        "# Guided Denoising Paired Pilot",
        "",
        "This is a small matched-rollout study before a full evaluation matrix. Each pair runs baseline and guided policies from the same initial simulator state and policy random seed; only guidance differs.",
        "",
        "- **Baseline:** guidance scale `0` (no guidance).",
        "- **Guided:** `{}` cost with guidance scale `{}`; guidance is applied {} {}.".format(
            manifest.get("cost_type", "penetration"),
            manifest["guidance_scale"],
            timestep_description,
            cap_description,
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
            "`trigger rate` is the fraction of evaluated reverse steps whose predicted trajectory enters the deployment-cost boundary. The baseline can therefore have a nonzero trigger rate even though scale `0` applies no update. Update norms are measured in normalized action space; reconstruction errors compare predicted and executed EEF trajectories. `action clips` counts policy actions outside environment limits.",
            "",
            "| environment | condition | trigger rate | update norm mean | update norm max | max waypoint displacement (cm) | reconstruction mean (cm) | reconstruction max (cm) | observation failure rate | action clips | max clip excess |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for env_name in sorted({row["env"] for row in aggregate_rows}):
        for condition in ("baseline", "guided"):
            row = aggregate_by_key[(env_name, condition)]
            lines.append(
                "| {env} | {condition} | {trigger:.3f} | {update_mean:.3f} | {update_max:.3f} | {waypoint:.3f} | {recon_mean:.3f} | {recon_max:.3f} | {obs_failure:.3f} | {clips} | {clip_max:.2e} |".format(
                    env=env_name,
                    condition=condition,
                    trigger=row["guidance_trigger_rate"],
                    update_mean=row["guidance_update_norm_mean"],
                    update_max=row["guidance_update_norm_max"],
                    waypoint=row["max_waypoint_displacement_m"] * 100.0,
                    recon_mean=row["trajectory_reconstruction_error_mean_cm"],
                    recon_max=row["trajectory_reconstruction_error_max_cm"],
                    obs_failure=row["observation_failure_rate"],
                    clips=row["action_clip_count"],
                    clip_max=row["action_clip_max_abs"],
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
    parser.add_argument(
        "--eval-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more paired-evaluation directories to aggregate.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Required when combining multiple evaluation directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.eval_dir) > 1 and args.out_dir is None:
        raise ValueError("--out-dir is required with multiple --eval-dir inputs")
    out_dir = args.out_dir or args.eval_dir[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    manifests = []
    for eval_dir in args.eval_dir:
        rows.extend(load_rows(eval_dir / "episode_metrics.jsonl"))
        manifests.append(json.loads((eval_dir / "manifest.json").read_text()))

    duplicate_keys = set()
    seen_keys = set()
    for row in rows:
        key = (row["env"], row["seed"], row["episode"], row["condition"])
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
    if duplicate_keys:
        raise ValueError("Duplicate episode-condition keys: {}".format(duplicate_keys))

    manifest = dict(manifests[0])
    ignored_keys = {"envs", "seeds", "decision", "decision_note"}
    expected = {key: value for key, value in manifest.items() if key not in ignored_keys}
    for other in manifests[1:]:
        observed = {key: value for key, value in other.items() if key not in ignored_keys}
        if observed != expected:
            raise ValueError("Evaluation manifests have incompatible settings")
    manifest["envs"] = sorted({env for item in manifests for env in item["envs"]})
    manifest["seeds"] = sorted({seed for item in manifests for seed in item["seeds"]})

    if len(args.eval_dir) > 1:
        manifest.pop("decision", None)
        manifest.pop("decision_note", None)
        (out_dir / "episode_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    aggregate_rows = paired_rollout_eval.aggregate(rows)
    paired_rows = paired_summary(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(aggregate_rows, indent=2) + "\n"
    )
    write_csv(out_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(out_dir / "paired_summary.csv", paired_rows)
    write_report(out_dir / "report.md", aggregate_rows, paired_rows, manifest)
    print("wrote paired report in {}".format(out_dir), flush=True)


if __name__ == "__main__":
    main()
