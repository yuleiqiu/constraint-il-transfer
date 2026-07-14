"""
Plot representative cases from a delta-EEF multi-env eval.

The script uses only saved eval artifacts. It does not rerun robosuite.

Usage:
    uv run python scripts/eef_pose_osc_policy/plot_delta_eef_eval_cases.py \
      --eval-dir outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50
"""

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_EVAL_DIR = "outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50"
TARGET_OBJECT = "Can"


def parse_bool(value):
    return str(value).lower() == "true"


def parse_int_or_none(value):
    if value in ("", "None", "nan"):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_float_or_none(value):
    if value in ("", "None", "nan"):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def read_features(path):
    rows = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            row["seed"] = int(row["seed"])
            row["episode"] = int(row["episode"])
            row["success"] = parse_bool(row["success"])
            row["failure"] = parse_bool(row["failure"])
            row["collision_any"] = parse_bool(row["collision_any"])
            row["collision_before_target"] = parse_bool(row["collision_before_target"])
            row["never_target_contact"] = parse_bool(row["never_target_contact"])
            row["target_lifted_5cm"] = parse_bool(row["target_lifted_5cm"])
            row["horizon"] = int(row["horizon"])
            row["first_target_contact_step"] = parse_int_or_none(row["first_target_contact_step"])
            row["first_non_target_contact_step"] = parse_int_or_none(row["first_non_target_contact_step"])
            row["min_eef_obstacle_clearance"] = parse_float_or_none(row["min_eef_obstacle_clearance"])
            row["target_max_z_gain"] = parse_float_or_none(row["target_max_z_gain"])
            row["non_target_max_displacement"] = parse_float_or_none(row["non_target_max_displacement"])
            rows.append(row)
    return rows


def choose_cases(rows, per_bucket):
    envs = sorted({r["env"] for r in rows})
    buckets = [
        ("failure_collision_before_target", lambda r: r["failure"] and r["collision_before_target"]),
        ("failure_no_collision", lambda r: r["failure"] and not r["collision_any"]),
        ("success_with_collision", lambda r: r["success"] and r["collision_any"]),
        ("success_clean", lambda r: r["success"] and not r["collision_any"]),
    ]
    selected = []
    seen = set()
    for env in envs:
        env_rows = [r for r in rows if r["env"] == env]
        for bucket, pred in buckets:
            matches = [r for r in env_rows if pred(r)]
            matches.sort(
                key=lambda r: (
                    r["seed"],
                    r["episode"],
                )
            )
            for row in matches[:per_bucket]:
                key = (row["env"], row["seed"], row["episode"])
                if key in seen:
                    continue
                out = dict(row)
                out["case_bucket"] = bucket
                selected.append(out)
                seen.add(key)
    return selected


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


def load_episode(eval_dir, case):
    h5_path = eval_dir / case["env"] / f"seed_{case['seed']}" / "episodes.hdf5"
    with h5py.File(h5_path, "r") as f:
        ep = f["episodes"][f"episode_{case['episode']}"]
        chunk_pos_err = []
        chunk_ori_err = []
        chunk_starts = []
        chunk_pred_xy = []
        for name in sorted(ep["chunks"].keys(), key=lambda x: int(x.split("_")[1])):
            chunk = ep["chunks"][name]
            start_step = int(chunk.attrs["start_step"])
            chunk_starts.append(start_step)
            chunk_pos_err.extend(chunk["raw_pos_err_cm"][()].tolist())
            chunk_ori_err.extend(chunk["raw_ori_err_deg"][()].tolist())
            pred = chunk["raw_pred_eef_pos"][()]
            executed = chunk["actual_eef_pos_executed"][()]
            if len(executed) > 0:
                chunk_pred_xy.append((start_step, pred[: len(executed), :2], executed[:, :2]))

        return {
            "eef_pos": ep["eef_pos"][()],
            "eef_quat_xyzw": ep["eef_quat_xyzw"][()],
            "object_names": decode_strings(ep["object_names"][()]),
            "object_pos": ep["object_pos"][()],
            "object_displacement": ep["object_displacement_from_initial"][()],
            "target_contact": ep["target_contact_count"][()],
            "non_target_contact": ep["non_target_contact_count"][()],
            "min_clearance": ep["min_eef_obstacle_clearance"][()],
            "raw_actions": ep["raw_actions"][()],
            "action_clip_delta_abs": ep["action_clip_delta_abs"][()],
            "chunk_pos_err_cm": np.asarray(chunk_pos_err, dtype=np.float64),
            "chunk_ori_err_deg": np.asarray(chunk_ori_err, dtype=np.float64),
            "chunk_pred_xy": chunk_pred_xy,
        }


def mark_step(ax, step, y=None, label=None, color="k"):
    if step is None:
        return
    ax.axvline(step, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
    if label and y is not None:
        ax.text(step, y, label, color=color, fontsize=8, rotation=90, va="bottom")


def safe_min(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.min(values))


def safe_max(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.max(values))


def plot_case(eval_dir, case, out_dir):
    data = load_episode(eval_dir, case)
    object_names = data["object_names"]
    target_idx = object_names.index(TARGET_OBJECT)
    obstacle_indices = [i for i, name in enumerate(object_names) if name != TARGET_OBJECT]
    eef = data["eef_pos"]
    obj = data["object_pos"]
    disp = data["object_displacement"]
    t = np.arange(len(data["target_contact"]))
    t_obs = np.arange(len(eef))

    first_target = first_positive_step(data["target_contact"])
    first_collision = first_positive_step(data["non_target_contact"])
    min_clear_step = None
    finite_clear = np.isfinite(data["min_clearance"])
    if np.any(finite_clear):
        min_clear_step = int(np.nanargmin(data["min_clearance"]))

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(
        f"{case['case_bucket']} | {case['env']} seed={case['seed']} ep={case['episode']} "
        f"success={case['success']} horizon={case['horizon']}",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.plot(eef[:, 0], eef[:, 1], color="black", linewidth=1.5, label="EEF")
    ax.scatter(eef[0, 0], eef[0, 1], color="black", marker="o", s=30, label="EEF start")
    ax.scatter(eef[-1, 0], eef[-1, 1], color="black", marker="x", s=40, label="EEF end")
    for i, name in enumerate(object_names):
        color = "tab:red" if name == TARGET_OBJECT else None
        ax.scatter(obj[0, i, 0], obj[0, i, 1], marker="o", s=70, label=f"{name} start", color=color)
        ax.scatter(obj[-1, i, 0], obj[-1, i, 1], marker="x", s=70, label=f"{name} end", color=color)
        ax.plot(obj[:, i, 0], obj[:, i, 1], linewidth=1.0, alpha=0.7, color=color)
        ax.text(obj[0, i, 0], obj[0, i, 1], name, fontsize=8)
    ax.set_title("EEF and object XY")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    ax = axes[0, 1]
    ax.plot(t_obs, eef[:, 2], label="EEF z", color="black")
    ax.plot(t_obs, obj[:, target_idx, 2], label="Can z", color="tab:red")
    for idx in obstacle_indices:
        ax.plot(t_obs, obj[:, idx, 2], label=f"{object_names[idx]} z", alpha=0.7)
    ymin, ymax = ax.get_ylim()
    mark_step(ax, first_target, ymin, "target", "tab:red")
    mark_step(ax, first_collision, ymin, "collision", "tab:orange")
    ax.set_title("Z position")
    ax.set_xlabel("step")
    ax.set_ylabel("z (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    if np.any(finite_clear):
        ax.plot(t, data["min_clearance"], color="tab:purple", label="min clearance")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        ymin, ymax = ax.get_ylim()
        mark_step(ax, min_clear_step, ymin, "min", "tab:purple")
    ax.set_title("EEF-obstacle clearance proxy")
    ax.set_xlabel("step")
    ax.set_ylabel("clearance (m)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, data["target_contact"], label="target contact", color="tab:red")
    ax.plot(t, data["non_target_contact"], label="non-target contact", color="tab:orange")
    ymax = max(1.0, safe_max(data["target_contact"]) or 1.0, safe_max(data["non_target_contact"]) or 1.0)
    mark_step(ax, first_target, ymax * 0.7, "target", "tab:red")
    mark_step(ax, first_collision, ymax * 0.5, "collision", "tab:orange")
    ax.set_title("Contacts per step")
    ax.set_xlabel("step")
    ax.set_ylabel("contact count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2, 0]
    ax.plot(t_obs, disp[:, target_idx], label="Can", color="tab:red")
    for idx in obstacle_indices:
        ax.plot(t_obs, disp[:, idx], label=object_names[idx], alpha=0.8)
    ax.set_title("Object displacement from initial")
    ax.set_xlabel("step")
    ax.set_ylabel("distance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2, 1]
    if len(data["chunk_pos_err_cm"]):
        ax.plot(data["chunk_pos_err_cm"], label="pos err cm", color="tab:blue")
    if len(data["chunk_ori_err_deg"]):
        ax.plot(data["chunk_ori_err_deg"], label="ori err deg", color="tab:green")
    ax.set_title("Predicted chunk vs executed EEF pose error")
    ax.set_xlabel("executed action index")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    stem = f"{case['case_bucket']}__{case['env']}__seed_{case['seed']}__ep_{case['episode']}"
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    summary = {
        "case_bucket": case["case_bucket"],
        "env": case["env"],
        "seed": case["seed"],
        "episode": case["episode"],
        "success": case["success"],
        "horizon": case["horizon"],
        "first_target_contact_step": first_target,
        "first_non_target_contact_step": first_collision,
        "min_clearance_step": min_clear_step,
        "min_clearance_m": safe_min(data["min_clearance"]),
        "target_max_z_gain_m": float(np.max(obj[:, target_idx, 2] - obj[0, target_idx, 2])),
        "target_final_displacement_m": float(disp[-1, target_idx]),
        "non_target_max_displacement_m": float(np.max(disp[:, obstacle_indices])) if obstacle_indices else 0.0,
        "object_names": object_names,
        "non_target_contact_names": case["non_target_contact_names"],
        "raw_pos_err_mean_cm": parse_float_or_none(case["raw_pos_err_mean_cm"]),
        "raw_pos_err_max_cm": parse_float_or_none(case["raw_pos_err_max_cm"]),
        "raw_ori_err_mean_deg": parse_float_or_none(case["raw_ori_err_mean_deg"]),
        "raw_ori_err_max_deg": parse_float_or_none(case["raw_ori_err_max_deg"]),
        "png": str(png_path),
    }
    json_path.write_text(json.dumps(summary, indent=2))
    return summary


def write_case_index(out_dir, summaries):
    csv_path = out_dir / "case_index.csv"
    if summaries:
        keys = list(summaries[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)

    lines = [
        "# Delta EEF Eval Case Plots",
        "",
        "| bucket | env | seed | episode | success | first target | first collision | min clearance m | target z gain m | plot |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        png_name = Path(row["png"]).name
        lines.append(
            "| {bucket} | {env} | {seed} | {ep} | {success} | {target} | {collision} | {clearance:.3f} | {zgain:.3f} | [{png}]({png}) |".format(
                bucket=row["case_bucket"],
                env=row["env"],
                seed=row["seed"],
                ep=row["episode"],
                success=row["success"],
                target=row["first_target_contact_step"]
                if row["first_target_contact_step"] is not None
                else "",
                collision=row["first_non_target_contact_step"]
                if row["first_non_target_contact_step"] is not None
                else "",
                clearance=row["min_clearance_m"] if row["min_clearance_m"] is not None else float("nan"),
                zgain=row["target_max_z_gain_m"],
                png=png_name,
            )
        )
    (out_dir / "case_index.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--per-bucket", type=int, default=3)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else eval_dir / "analysis"
    features_path = analysis_dir / "episode_features.csv"
    out_dir = analysis_dir / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_features(features_path)
    cases = choose_cases(rows, args.per_bucket)
    summaries = [plot_case(eval_dir, case, out_dir) for case in cases]
    write_case_index(out_dir, summaries)

    print(f"selected {len(summaries)} cases")
    print(f"wrote {out_dir / 'case_index.md'}")


if __name__ == "__main__":
    main()
