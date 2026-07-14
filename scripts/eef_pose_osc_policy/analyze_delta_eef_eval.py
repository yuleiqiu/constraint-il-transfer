"""
Analyze multi-env delta-EEF eval outputs.

This script consumes the directory produced by eval_delta_eef_multienv.py and
writes episode-level features plus a compact markdown report.

Usage:
    uv run python scripts/eef_pose_osc_policy/analyze_delta_eef_eval.py \
      --eval-dir outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


DEFAULT_EVAL_DIR = "outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50"
TARGET_OBJECT = "Can"
LIFT_THRESHOLD_M = 0.05


def load_jsonl(path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decode_strings(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def first_positive_step(values):
    values = np.asarray(values)
    idx = np.flatnonzero(values > 0)
    if len(idx) == 0:
        return None
    return int(idx[0])


def finite_float(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def flatten_chunk_error(row, key, axis, stat):
    try:
        return finite_float(row[key][axis][stat])
    except KeyError:
        return None


def read_episode_hdf5(h5_path, episode_idx):
    with h5py.File(h5_path, "r") as f:
        ep = f["episodes"][f"episode_{episode_idx}"]
        object_names = decode_strings(ep["object_names"][()])
        target_idx = object_names.index(TARGET_OBJECT)
        obstacle_indices = [i for i, name in enumerate(object_names) if name != TARGET_OBJECT]

        target_contact = ep["target_contact_count"][()]
        non_target_contact = ep["non_target_contact_count"][()]
        object_pos = ep["object_pos"][()]
        object_disp = ep["object_displacement_from_initial"][()]
        min_clearance = ep["min_eef_obstacle_clearance"][()]

        target_pos = object_pos[:, target_idx, :]
        target_disp = object_disp[:, target_idx]
        target_z_gain = target_pos[:, 2] - target_pos[0, 2]

        if obstacle_indices:
            non_target_disp = object_disp[:, obstacle_indices]
            per_obstacle_max_disp = np.max(non_target_disp, axis=0)
            non_target_max_disp = float(np.max(per_obstacle_max_disp))
            obstacle_max_disp = {
                object_names[idx]: float(per_obstacle_max_disp[j])
                for j, idx in enumerate(obstacle_indices)
            }
        else:
            non_target_max_disp = 0.0
            obstacle_max_disp = {}

        return {
            "object_names": object_names,
            "first_target_contact_step": first_positive_step(target_contact),
            "first_non_target_contact_step": first_positive_step(non_target_contact),
            "target_contact_steps": int(np.count_nonzero(target_contact > 0)),
            "non_target_contact_steps": int(np.count_nonzero(non_target_contact > 0)),
            "target_final_displacement": float(target_disp[-1]),
            "target_max_z_gain": float(np.max(target_z_gain)),
            "target_final_z_gain": float(target_z_gain[-1]),
            "non_target_max_displacement": non_target_max_disp,
            "obstacle_max_displacement": obstacle_max_disp,
            "min_clearance_step": first_positive_step(np.asarray(min_clearance) == np.nanmin(min_clearance))
            if np.any(np.isfinite(min_clearance))
            else None,
        }


def collect_episode_features(eval_dir):
    rows = []
    for metrics_path in sorted(eval_dir.glob("*/seed_*/episode_metrics.jsonl")):
        h5_path = metrics_path.with_name("episodes.hdf5")
        env = metrics_path.parent.parent.name
        seed = int(metrics_path.parent.name.removeprefix("seed_"))
        for row in load_jsonl(metrics_path):
            h5 = read_episode_hdf5(h5_path, int(row["episode"]))
            first_target = h5["first_target_contact_step"]
            first_non_target = h5["first_non_target_contact_step"]
            success = bool(row["success"])
            collision_any = bool(row["non_target_collision_any"])
            collision_before_target = first_non_target is not None and (
                first_target is None or first_non_target < first_target
            )
            target_lifted = h5["target_max_z_gain"] >= LIFT_THRESHOLD_M
            never_target_contact = first_target is None

            feature = {
                "env": env,
                "seed": seed,
                "episode": int(row["episode"]),
                "success": success,
                "failure": not success,
                "horizon": int(row["horizon"]),
                "return": finite_float(row.get("return")),
                "action_clip_count": int(row["action_clip_count"]),
                "action_clip_max_abs_delta": finite_float(row["action_clip_max_abs_delta"]),
                "target_contact_count": int(row["target_contact_count"]),
                "non_target_contact_count": int(row["non_target_contact_count"]),
                "first_target_contact_step": first_target,
                "first_non_target_contact_step": first_non_target,
                "collision_any": collision_any,
                "collision_before_target": collision_before_target,
                "never_target_contact": never_target_contact,
                "target_lifted_5cm": target_lifted,
                "min_eef_obstacle_clearance": finite_float(row.get("min_eef_obstacle_clearance")),
                "target_max_displacement": finite_float(row.get("target_max_displacement")),
                "target_final_displacement": h5["target_final_displacement"],
                "target_max_z_gain": h5["target_max_z_gain"],
                "target_final_z_gain": h5["target_final_z_gain"],
                "non_target_max_displacement": finite_float(row.get("non_target_max_displacement")),
                "raw_pos_err_mean_cm": flatten_chunk_error(row, "raw_chunk_error", "pos_cm", "mean"),
                "raw_pos_err_max_cm": flatten_chunk_error(row, "raw_chunk_error", "pos_cm", "max"),
                "raw_ori_err_mean_deg": flatten_chunk_error(row, "raw_chunk_error", "ori_deg", "mean"),
                "raw_ori_err_max_deg": flatten_chunk_error(row, "raw_chunk_error", "ori_deg", "max"),
                "non_target_contact_names": ",".join(row.get("non_target_contact_names", [])),
                "object_names": ",".join(h5["object_names"]),
                "obstacle_max_displacement_json": json.dumps(h5["obstacle_max_displacement"], sort_keys=True),
            }

            if not success and collision_any:
                feature["failure_bucket_collision"] = True
            else:
                feature["failure_bucket_collision"] = False
            feature["failure_bucket_never_touched_target"] = (not success) and never_target_contact
            feature["failure_bucket_touched_no_lift"] = (not success) and (not never_target_contact) and (not target_lifted)
            feature["failure_bucket_lifted_not_success"] = (not success) and target_lifted

            rows.append(feature)
    return rows


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def std(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(values) < 2:
        return 0.0 if values else None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def median(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def summarize_by_env(rows):
    by_env = defaultdict(list)
    for row in rows:
        by_env[row["env"]].append(row)

    summaries = []
    for env, env_rows in sorted(by_env.items()):
        n = len(env_rows)
        failures = [r for r in env_rows if r["failure"]]
        successes = [r for r in env_rows if r["success"]]
        collisions = [r for r in env_rows if r["collision_any"]]
        safe_successes = [r for r in successes if not r["collision_any"]]
        successes_with_collision = [r for r in successes if r["collision_any"]]
        failure_collisions = [r for r in failures if r["collision_any"]]
        non_completion = [r for r in failures if not r["collision_any"]]
        collision_before_target = [r for r in failures if r["collision_before_target"]]
        never_target = [r for r in failures if r["never_target_contact"]]
        touched_no_lift = [r for r in failures if (not r["never_target_contact"]) and (not r["target_lifted_5cm"])]
        lifted_not_success = [r for r in failures if r["target_lifted_5cm"]]

        seed_success_rates = []
        seed_safe_success_rates = []
        seed_collision_rates = []
        seed_non_completion_rates = []
        for seed in sorted({r["seed"] for r in env_rows}):
            seed_rows = [r for r in env_rows if r["seed"] == seed]
            seed_success_rates.append(mean([r["success"] for r in seed_rows]))
            seed_safe_success_rates.append(
                mean([r["success"] and not r["collision_any"] for r in seed_rows])
            )
            seed_collision_rates.append(mean([r["collision_any"] for r in seed_rows]))
            seed_non_completion_rates.append(
                mean([r["failure"] and not r["collision_any"] for r in seed_rows])
            )

        contact_name_counts = Counter()
        for row in collisions:
            for name in row["non_target_contact_names"].split(","):
                if name:
                    contact_name_counts[name] += 1

        summaries.append(
            {
                "env": env,
                "episodes": n,
                "success_rate_mean_over_seeds": mean(seed_success_rates),
                "success_rate_std_over_seeds": std(seed_success_rates),
                "safe_success_rate_mean_over_seeds": mean(seed_safe_success_rates),
                "safe_success_rate_std_over_seeds": std(seed_safe_success_rates),
                "collision_rate_mean_over_seeds": mean(seed_collision_rates),
                "collision_rate_std_over_seeds": std(seed_collision_rates),
                "non_completion_rate_mean_over_seeds": mean(seed_non_completion_rates),
                "non_completion_rate_std_over_seeds": std(seed_non_completion_rates),
                "safe_success_count": len(safe_successes),
                "safe_success_rate": len(safe_successes) / n,
                "success_with_collision_count": len(successes_with_collision),
                "success_with_collision_rate": len(successes_with_collision) / n,
                "collision_failure_count": len(failure_collisions),
                "collision_failure_rate": len(failure_collisions) / n,
                "non_completion_count": len(non_completion),
                "non_completion_rate": len(non_completion) / n,
                "failure_count": len(failures),
                "failure_collision_count": len(failure_collisions),
                "failure_collision_pct_of_failures": pct(len(failure_collisions), len(failures)),
                "failure_collision_before_target_count": len(collision_before_target),
                "failure_collision_before_target_pct_of_failures": pct(len(collision_before_target), len(failures)),
                "failure_never_target_contact_count": len(never_target),
                "failure_never_target_contact_pct_of_failures": pct(len(never_target), len(failures)),
                "failure_touched_no_lift_count": len(touched_no_lift),
                "failure_lifted_not_success_count": len(lifted_not_success),
                "median_first_target_contact_step": median([r["first_target_contact_step"] for r in env_rows]),
                "median_first_collision_step": median([r["first_non_target_contact_step"] for r in collisions]),
                "median_min_clearance_success_m": median([r["min_eef_obstacle_clearance"] for r in successes]),
                "median_min_clearance_failure_m": median([r["min_eef_obstacle_clearance"] for r in failures]),
                "mean_target_max_z_gain_success_m": mean([r["target_max_z_gain"] for r in successes]),
                "mean_target_max_z_gain_failure_m": mean([r["target_max_z_gain"] for r in failures]),
                "mean_raw_pos_err_cm": mean([r["raw_pos_err_mean_cm"] for r in env_rows]),
                "mean_raw_ori_err_deg": mean([r["raw_ori_err_mean_deg"] for r in env_rows]),
                "action_clip_count_sum": sum(r["action_clip_count"] for r in env_rows),
                "contact_name_counts": dict(contact_name_counts),
            }
        )
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, rows):
    if not rows:
        return
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["contact_name_counts"] = json.dumps(flat["contact_name_counts"], sort_keys=True)
        flat_rows.append(flat)
    write_csv(path, flat_rows)


def markdown_report(eval_dir, summaries):
    lines = [
        "# Delta EEF Multi-Env Eval Analysis",
        "",
        f"Eval dir: `{eval_dir}`",
        "",
        "## Environment Summary",
        "",
        "| env | success mean±std | collision mean±std | failures | failure+collision | collision before target | never target contact | touched no lift | lifted not success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {env} | {smean}±{sstd} | {cmean}±{cstd} | {fail} | {fc} ({fcpct}%) | {cbt} ({cbtpct}%) | {nt} ({ntpct}%) | {tnl} | {lns} |".format(
                env=row["env"],
                smean=fmt(row["success_rate_mean_over_seeds"], 3),
                sstd=fmt(row["success_rate_std_over_seeds"], 3),
                cmean=fmt(row["collision_rate_mean_over_seeds"], 3),
                cstd=fmt(row["collision_rate_std_over_seeds"], 3),
                fail=row["failure_count"],
                fc=row["failure_collision_count"],
                fcpct=fmt(row["failure_collision_pct_of_failures"], 1),
                cbt=row["failure_collision_before_target_count"],
                cbtpct=fmt(row["failure_collision_before_target_pct_of_failures"], 1),
                nt=row["failure_never_target_contact_count"],
                ntpct=fmt(row["failure_never_target_contact_pct_of_failures"], 1),
                tnl=row["failure_touched_no_lift_count"],
                lns=row["failure_lifted_not_success_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Safety / Completion Partition",
            "",
            "Paper metrics use safe success as SR, any non-target collision as CR, and collision-free failure as NCR. By construction, safe SR + CR + NCR = 1.",
            "",
            "| env | task SR | safe SR mean±std | CR mean±std | NCR mean±std | success+collision | collision failure |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {env} | {task} | {safe_mean}±{safe_std} | {cr_mean}±{cr_std} | {ncr_mean}±{ncr_std} | {swc} | {cf} |".format(
                env=row["env"],
                task=fmt(row["success_rate_mean_over_seeds"], 3),
                safe_mean=fmt(row["safe_success_rate_mean_over_seeds"], 3),
                safe_std=fmt(row["safe_success_rate_std_over_seeds"], 3),
                cr_mean=fmt(row["collision_rate_mean_over_seeds"], 3),
                cr_std=fmt(row["collision_rate_std_over_seeds"], 3),
                ncr_mean=fmt(row["non_completion_rate_mean_over_seeds"], 3),
                ncr_std=fmt(row["non_completion_rate_std_over_seeds"], 3),
                swc=fmt(row["success_with_collision_rate"], 3),
                cf=fmt(row["collision_failure_rate"], 3),
            )
        )

    lines.extend(
        [
            "",
            "## Trajectory / Controller Health",
            "",
            "| env | action clips | mean chunk pos err cm | mean chunk ori err deg | median first target step | median first collision step |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {env} | {clips} | {pos} | {ori} | {target_step} | {collision_step} |".format(
                env=row["env"],
                clips=row["action_clip_count_sum"],
                pos=fmt(row["mean_raw_pos_err_cm"], 3),
                ori=fmt(row["mean_raw_ori_err_deg"], 3),
                target_step=fmt(row["median_first_target_contact_step"], 1),
                collision_step=fmt(row["median_first_collision_step"], 1),
            )
        )

    lines.extend(
        [
            "",
            "## Clearance / Lift",
            "",
            "| env | median min clearance success m | median min clearance failure m | mean target z gain success m | mean target z gain failure m | contact names |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {env} | {cs} | {cf} | {zs} | {zf} | `{names}` |".format(
                env=row["env"],
                cs=fmt(row["median_min_clearance_success_m"], 3),
                cf=fmt(row["median_min_clearance_failure_m"], 3),
                zs=fmt(row["mean_target_max_z_gain_success_m"], 3),
                zf=fmt(row["mean_target_max_z_gain_failure_m"], 3),
                names=json.dumps(row["contact_name_counts"], sort_keys=True),
            )
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `failure+collision` is an overlap metric: it counts failed episodes with any non-target contact.",
            "- The four-way partition is safe success / success with collision / collision failure / NCR.",
            "- `collision before target` is a stronger obstruction signal: the first non-target contact happened before the first target contact, or the target was never contacted.",
            f"- `touched no lift` uses target max z gain < {LIFT_THRESHOLD_M:.2f} m.",
            "- Negative clearance means the coarse EEF-obstacle clearance proxy crossed the obstacle radius.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_episode_features(eval_dir)
    summaries = summarize_by_env(rows)

    write_csv(out_dir / "episode_features.csv", rows)
    write_summary_csv(out_dir / "env_summary.csv", summaries)
    with (out_dir / "analysis_summary.json").open("w") as f:
        json.dump({"eval_dir": str(eval_dir), "envs": summaries}, f, indent=2)
    (out_dir / "analysis.md").write_text(markdown_report(eval_dir, summaries))

    print(f"wrote {out_dir / 'episode_features.csv'}")
    print(f"wrote {out_dir / 'env_summary.csv'}")
    print(f"wrote {out_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
