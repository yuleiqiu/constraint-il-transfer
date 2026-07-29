"""Run and summarize the paired D0/D2 targeted-fine-tuning checkpoint screen.

The screen is a development-set procedure, not the final result:

* fixed environment seed 600 and episode indices 0--19;
* deterministic torch execution;
* an episode-specific action seed shared across checkpoints;
* D0 checks source-skill retention and D2 checks targeted adaptation.

Run the frozen source and coarse mixed checkpoints:

    MUJOCO_GL=egl uv run python \
      scripts/2026-07-28_targeted_correction/screen_checkpoints.py

After choosing an update count, evaluate the update-matched clean control:

    MUJOCO_GL=egl uv run python \
      scripts/2026-07-28_targeted_correction/screen_checkpoints.py \
      --no-source --mixed-epochs --clean-epochs 8

The final D0--D3 evaluation must use seeds not used by this screen.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "scripts/eef_pose_osc_policy/eval_delta_eef_multienv.py"
SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "outputs/robomimic/train/dp_can_delta_pose_osc/20260707222943/models"
    / "model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth"
)
MIXED_MODEL_DIR = (
    REPO_ROOT
    / "outputs/robomimic/train/dp_can_delta_pose_osc_targeted_mixed_upper_bound"
    / "20260728204504/models"
)
CLEAN_MODEL_DIR = (
    REPO_ROOT
    / "outputs/robomimic/train/dp_can_delta_pose_osc_clean_continuation_upper_bound"
    / "20260728205810/models"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/targeted_finetuning/checkpoint_screen_seed600_n20"
DEFAULT_ENVS = ["PickPlaceCan", "PickPlaceBreadCerealCan"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-epochs", nargs="*", type=int, default=[4, 8, 12, 16, 20])
    parser.add_argument("--mixed-model-dir", type=Path, default=MIXED_MODEL_DIR)
    parser.add_argument("--mixed-label-prefix", default="mixed")
    parser.add_argument("--mixed-variant", default="mixed")
    parser.add_argument("--clean-epochs", nargs="*", type=int, default=[])
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--seed", type=int, default=600)
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def checkpoint_specs(args):
    specs = []
    if args.source:
        specs.append(("source", "source", 0, SOURCE_CHECKPOINT))
    for epoch in args.mixed_epochs:
        specs.append(
            (
                f"{args.mixed_label_prefix}_u{epoch * 500:05d}",
                args.mixed_variant,
                epoch * 500,
                args.mixed_model_dir.resolve() / f"model_epoch_{epoch}.pth",
            )
        )
    for epoch in args.clean_epochs:
        specs.append(
            (
                f"clean_u{epoch * 500:05d}",
                "clean",
                epoch * 500,
                CLEAN_MODEL_DIR / f"model_epoch_{epoch}.pth",
            )
        )
    labels = [spec[0] for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate checkpoint labels: {labels}")
    return specs


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_metrics_path(eval_dir, env_name, seed):
    return eval_dir / env_name / f"seed_{seed}" / "episode_metrics.jsonl"


def output_is_complete(eval_dir, checkpoint, args):
    manifest_path = eval_dir / "manifest.json"
    summary_path = eval_dir / "summary.csv"
    if not manifest_path.exists() or not summary_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if Path(manifest["agent"]).resolve() != checkpoint.resolve():
        raise ValueError(f"{eval_dir} belongs to a different checkpoint")
    if manifest.get("seeds") != [args.seed] or manifest.get("envs") != args.envs:
        raise ValueError(f"{eval_dir} has a different seed or environment contract")
    if manifest.get("n_rollouts") != args.n_rollouts:
        raise ValueError(f"{eval_dir} has a different rollout count")
    if not manifest.get("paired_action_seeds") or not manifest.get("deterministic_torch"):
        raise ValueError(f"{eval_dir} was not produced by the paired deterministic protocol")
    for env_name in args.envs:
        metrics_path = expected_metrics_path(eval_dir, env_name, args.seed)
        if not metrics_path.exists() or len(read_jsonl(metrics_path)) != args.n_rollouts:
            return False
    return True


def run_eval(label, checkpoint, args):
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    eval_dir = args.out_dir / label
    if output_is_complete(eval_dir, checkpoint, args):
        print(f"[skip complete] {label}", flush=True)
        return
    if eval_dir.exists() and any(eval_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete output already exists at {eval_dir}; inspect it and move it "
            "aside before rerunning"
        )

    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--agent",
        str(checkpoint),
        "--envs",
        *args.envs,
        "--seeds",
        str(args.seed),
        "--n-rollouts",
        str(args.n_rollouts),
        "--horizon",
        str(args.horizon),
        "--paired-action-seeds",
        "--deterministic-torch",
        "--out-dir",
        str(eval_dir),
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu_id)
    env["ROBOMIMIC_GPU_ID"] = str(args.gpu_id)
    print(f"[run] {label}: {checkpoint}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    if not output_is_complete(eval_dir, checkpoint, args):
        raise RuntimeError(f"Evaluation did not produce a complete result: {eval_dir}")


def outcome_counts(rows):
    safe_success = sum(bool(r["success"]) and not bool(r["non_target_collision_any"]) for r in rows)
    success_collision = sum(bool(r["success"]) and bool(r["non_target_collision_any"]) for r in rows)
    collision_failure = sum(not bool(r["success"]) and bool(r["non_target_collision_any"]) for r in rows)
    ncr = sum(not bool(r["success"]) and not bool(r["non_target_collision_any"]) for r in rows)
    assert safe_success + success_collision + collision_failure + ncr == len(rows)
    return {
        "n": len(rows),
        "task_success": safe_success + success_collision,
        "safe_success": safe_success,
        "success_collision": success_collision,
        "collision_failure": collision_failure,
        "ncr_count": ncr,
        "collision_count": success_collision + collision_failure,
        "clip_count": sum(int(r["action_clip_count"]) for r in rows),
    }


def paired_transitions(source_rows, candidate_rows):
    source = {(int(r["seed"]), int(r["episode"])): r for r in source_rows}
    candidate = {(int(r["seed"]), int(r["episode"])): r for r in candidate_rows}
    if source.keys() != candidate.keys():
        raise ValueError("Source and candidate episode keys do not match")
    fail_to_success = 0
    success_to_failure = 0
    collision_failure_to_ncr = 0
    for key in sorted(source):
        src = source[key]
        dst = candidate[key]
        src_success = bool(src["success"])
        dst_success = bool(dst["success"])
        src_collision = bool(src["non_target_collision_any"])
        dst_collision = bool(dst["non_target_collision_any"])
        fail_to_success += int(not src_success and dst_success)
        success_to_failure += int(src_success and not dst_success)
        collision_failure_to_ncr += int(
            not src_success and src_collision and not dst_success and not dst_collision
        )
    return {
        "fail_to_success": fail_to_success,
        "success_to_failure": success_to_failure,
        "collision_failure_to_ncr": collision_failure_to_ncr,
    }


def collect_summary(specs, args):
    raw = {}
    for label, variant, updates, checkpoint in specs:
        eval_dir = args.out_dir / label
        if not output_is_complete(eval_dir, checkpoint, args):
            continue
        raw[label] = {}
        for env_name in args.envs:
            raw[label][env_name] = read_jsonl(
                expected_metrics_path(eval_dir, env_name, args.seed)
            )

    if not raw:
        raise RuntimeError("No complete screen outputs found")
    if "source" not in raw:
        source_dir = args.out_dir / "source"
        source_spec = ("source", "source", 0, SOURCE_CHECKPOINT)
        if output_is_complete(source_dir, SOURCE_CHECKPOINT, args):
            raw["source"] = {
                env_name: read_jsonl(expected_metrics_path(source_dir, env_name, args.seed))
                for env_name in args.envs
            }
            specs = [source_spec, *specs]
        else:
            raise RuntimeError("A complete source evaluation is required for paired transitions")

    rows = []
    seen = set()
    for label, variant, updates, checkpoint in specs:
        if label in seen or label not in raw:
            continue
        seen.add(label)
        for env_name in args.envs:
            counts = outcome_counts(raw[label][env_name])
            n = counts["n"]
            transitions = (
                {"fail_to_success": 0, "success_to_failure": 0, "collision_failure_to_ncr": 0}
                if label == "source"
                else paired_transitions(raw["source"][env_name], raw[label][env_name])
            )
            rows.append(
                {
                    "label": label,
                    "variant": variant,
                    "updates": updates,
                    "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                    "env": env_name,
                    "seed": args.seed,
                    "n": n,
                    "task_sr": counts["task_success"] / n,
                    "safe_sr": counts["safe_success"] / n,
                    "cr": counts["collision_count"] / n,
                    "ncr": counts["ncr_count"] / n,
                    "safe_success": counts["safe_success"],
                    "success_collision": counts["success_collision"],
                    "collision_failure": counts["collision_failure"],
                    "ncr_count": counts["ncr_count"],
                    "fail_to_success": transitions["fail_to_success"],
                    "success_to_failure": transitions["success_to_failure"],
                    "collision_failure_to_ncr": transitions["collision_failure_to_ncr"],
                    "clip_count": counts["clip_count"],
                }
            )
    return rows


def write_summary(rows, args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "screen_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_label_env = {(r["label"], r["env"]): r for r in rows}
    source_d2 = by_label_env.get(("source", "PickPlaceBreadCerealCan"))
    mixed_labels = sorted(
        {
            r["label"]
            for r in rows
            if r["variant"] == args.mixed_variant
        },
        key=lambda label: by_label_env[(label, args.envs[0])]["updates"],
    )
    eligible = []
    for label in mixed_labels:
        d0 = by_label_env.get((label, "PickPlaceCan"))
        d2 = by_label_env.get((label, "PickPlaceBreadCerealCan"))
        if d0 is None or d2 is None or source_d2 is None:
            continue
        if (
            d0["task_sr"] >= 0.85
            and d2["fail_to_success"] >= 4
            and d2["success_to_failure"] <= 1
            and d2["ncr"] <= source_d2["ncr"]
            and d0["clip_count"] == 0
            and d2["clip_count"] == 0
        ):
            eligible.append(label)
    provisional = None
    if eligible:
        provisional = max(
            eligible,
            key=lambda label: (
                by_label_env[(label, "PickPlaceBreadCerealCan")]["task_sr"],
                by_label_env[(label, "PickPlaceBreadCerealCan")]["safe_sr"],
                -by_label_env[(label, "PickPlaceBreadCerealCan")]["ncr"],
                -by_label_env[(label, "PickPlaceBreadCerealCan")]["updates"],
            ),
        )

    payload = {
        "status": "development checkpoint screen; not final evaluation",
        "seed": args.seed,
        "n_rollouts": args.n_rollouts,
        "envs": args.envs,
        "paired_action_seeds": True,
        "deterministic_torch": True,
        "eligible_mixed_labels": eligible,
        "provisional_mixed_label": provisional,
        "selection_rule": (
            "D0 Task SR >= 0.85; D2 fail->success >= 4; D2 success->failure <= 1; "
            "D2 NCR <= source; zero clipping; then maximize D2 Task SR, Safe SR, "
            "minimize NCR, and prefer fewer updates"
        ),
        "rows": rows,
    }
    (args.out_dir / "screen_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Targeted Fine-Tuning Checkpoint Screen",
        "",
        "Development bank only: seed 600, episode indices 0--19, with paired "
        "per-episode action seeds and deterministic torch execution. Final claims "
        "must use untouched seeds.",
        "",
        "| checkpoint | updates | env | Task SR | Safe SR | CR | NCR | "
        "safe / succ+coll / coll fail / NCR | fail→success | success→failure | clips |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        partition = (
            f"{row['safe_success']} / {row['success_collision']} / "
            f"{row['collision_failure']} / {row['ncr_count']}"
        )
        lines.append(
            f"| {row['label']} | {row['updates']} | {row['env']} | "
            f"{row['task_sr']:.3f} | {row['safe_sr']:.3f} | {row['cr']:.3f} | "
            f"{row['ncr']:.3f} | {partition} | {row['fail_to_success']} | "
            f"{row['success_to_failure']} | {row['clip_count']} |"
        )
    lines.extend(
        [
            "",
            f"Eligible mixed checkpoints: {', '.join(eligible) if eligible else 'none'}.",
            f"Provisional D2 choice: `{provisional}`." if provisional else "No checkpoint passes the development gate.",
            "",
            "The provisional choice still requires its update-matched clean-only "
            "control, the fixed 20-state support diagnostic, and D0--D3 evaluation "
            "on seeds 601/602.",
        ]
    )
    (args.out_dir / "screen_summary.md").write_text("\n".join(lines) + "\n")
    return provisional


def main():
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    specs = checkpoint_specs(args)
    if not specs and not args.summarize_only:
        raise ValueError("No checkpoints selected")
    if not args.summarize_only:
        for label, _, _, checkpoint in specs:
            run_eval(label, checkpoint, args)
    rows = collect_summary(specs, args)
    provisional = write_summary(rows, args)
    print(f"Wrote screen summary to {args.out_dir}", flush=True)
    print(f"Provisional mixed checkpoint: {provisional}", flush=True)


if __name__ == "__main__":
    main()
