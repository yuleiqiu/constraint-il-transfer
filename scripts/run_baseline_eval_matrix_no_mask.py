#!/usr/bin/env python3
"""Run the no-guidance baseline eval matrix sequentially."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/robomimic/checkpoints/diffusion_policy_can_yq_image/best.pth"
EVAL_SCRIPT = ROOT / "third_party/robomimic/robomimic/scripts/run_trained_agent.py"
OUTPUT_ROOT = (
    ROOT
    / "outputs/robomimic/eval/baseline/no_guidance/diffusion_policy_can_yq_image"
)

ENVS = [
    "PickPlaceCan",
    "PickPlaceBreadCan",
    "PickPlaceBreadCerealCan",
    "PickPlaceBreadCerealMilkCan",
]
SEEDS = [600, 601, 602]
CAMERAS = ["agentview", "robot0_eye_in_hand"]

REQUIRED_ROLLOUT_KEYS = [
    "Return",
    "Horizon",
    "Success_Rate",
    "Obstacle_Guidance_Cost",
    "Obstacle_Guidance_Min_Distance",
    "Obstacle_Guidance_Trigger_Count",
    "Obstacle_Guidance_Trigger_Rate",
    "Obstacle_Guidance_Positive_Cost_Count",
    "Obstacle_Guidance_Positive_Cost_Rate",
    "Non_Target_Collision_Count",
    "Non_Target_Collision_Step_Count",
    "Non_Target_Collision_Rate",
    "Non_Target_Collision_Any",
    "Non_Target_Collision_Object_Counts",
    "Pointcloud_Total_Point_Count",
    "Pointcloud_Point_Count",
]
NA_KEYS = [
    "Obstacle_Guidance_Cost",
    "Obstacle_Guidance_Min_Distance",
    "Obstacle_Guidance_Trigger_Count",
    "Obstacle_Guidance_Trigger_Rate",
    "Obstacle_Guidance_Positive_Cost_Count",
    "Obstacle_Guidance_Positive_Cost_Rate",
    "Pointcloud_Total_Point_Count",
    "Pointcloud_Point_Count",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def find_existing_eval_processes() -> list[str]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("failed to inspect running processes: {}".format(proc.stderr.strip()))
    current_pid = os.getpid()
    matches = []
    needles = ("run_trained_agent.py", "run_obstacle_guided_agent.py")
    for line in proc.stdout.splitlines():
        if not any(needle in line for needle in needles):
            continue
        parts = line.strip().split(None, 4)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == current_pid:
            continue
        matches.append(line.rstrip())
    return matches


def make_command(env_name: str, seed: int, job_dir: Path) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(EVAL_SCRIPT),
        "--agent",
        str(CHECKPOINT),
        "--env",
        env_name,
        "--seed",
        str(seed),
        "--n_rollouts",
        "50",
        "--horizon",
        "400",
        "--video_path",
        str(job_dir / "rollout.mp4"),
        "--stats_path",
        str(job_dir / "stats.json"),
        "--camera_names",
        *CAMERAS,
    ]


def validate_stats(stats: dict) -> list[str]:
    errors = []
    rollouts = stats.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != 50:
        errors.append("stats.json has {} rollouts, expected 50".format(len(rollouts or [])))
        return errors

    average = stats.get("average", {})
    for key in NA_KEYS:
        if average.get(key) is not None:
            errors.append("average.{} expected null, got {}".format(key, average.get(key)))

    totals = stats.get("totals", {})
    for key in ("Obstacle_Guidance_Trigger_Count", "Obstacle_Guidance_Positive_Cost_Count"):
        if totals.get(key) is not None:
            errors.append("totals.{} expected null, got {}".format(key, totals.get(key)))

    for i, rollout in enumerate(rollouts):
        missing = [key for key in REQUIRED_ROLLOUT_KEYS if key not in rollout]
        if missing:
            errors.append("rollout {} missing keys: {}".format(i, ", ".join(missing)))
        for key in NA_KEYS:
            if rollout.get(key) is not None:
                errors.append("rollout {} {} expected null, got {}".format(i, key, rollout.get(key)))
        object_counts = rollout.get("Non_Target_Collision_Object_Counts", {})
        if isinstance(object_counts, dict):
            visual_names = [name for name in object_counts if str(name).startswith("Visual")]
            if visual_names:
                errors.append(
                    "rollout {} collision object names include Visual markers: {}".format(
                        i, ", ".join(visual_names)
                    )
                )
    return errors


def summarize_job(env_name: str, seed: int, job_dir: Path, stats: dict) -> dict:
    avg = stats.get("average", {})
    totals = stats.get("totals", {})
    return {
        "env": env_name,
        "seed": seed,
        "job_dir": str(job_dir.relative_to(ROOT)),
        "rollout_mp4": str((job_dir / "rollout.mp4").relative_to(ROOT)),
        "stats_json": str((job_dir / "stats.json").relative_to(ROOT)),
        "command_txt": str((job_dir / "command.txt").relative_to(ROOT)),
        "run_log": str((job_dir / "run.log").relative_to(ROOT)),
        "num_rollouts": len(stats.get("rollouts", [])),
        "success_rate": avg.get("Success_Rate"),
        "num_success": avg.get("Num_Success"),
        "return": avg.get("Return"),
        "horizon": avg.get("Horizon"),
        "non_target_collision_count": avg.get("Non_Target_Collision_Count"),
        "non_target_collision_rollouts": avg.get("Num_Non_Target_Collision_Rollouts"),
        "obstacle_guidance_trigger_count_total": totals.get("Obstacle_Guidance_Trigger_Count"),
        "obstacle_guidance_trigger_count_mean": avg.get("Obstacle_Guidance_Trigger_Count"),
        "obstacle_guidance_trigger_rate": avg.get("Obstacle_Guidance_Trigger_Rate"),
        "pointcloud_total_point_count_mean": avg.get("Pointcloud_Total_Point_Count"),
        "non_target_collision_object_counts": totals.get("Non_Target_Collision_Object_Counts", {}),
    }


def write_summary(summary: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "summary.json").open("w") as f:
        json.dump(json_safe(summary), f, indent=4)
        f.write("\n")


def main() -> int:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(EVAL_SCRIPT)

    existing = find_existing_eval_processes()
    if existing:
        print("Refusing to start because an eval agent is already running:")
        for line in existing:
            print(line)
        return 2

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "method": "baseline_no_guidance",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "settings": {
            "envs": ENVS,
            "seeds": SEEDS,
            "n_rollouts": 50,
            "horizon": 400,
            "camera_names": CAMERAS,
            "guidance": None,
            "pointcloud": None,
            "MUJOCO_GL": "egl",
        },
        "jobs": [],
    }
    write_summary(summary)

    env_vars = os.environ.copy()
    env_vars["MUJOCO_GL"] = "egl"

    for env_name in ENVS:
        for seed in SEEDS:
            job_dir = OUTPUT_ROOT / env_name / "seed_{}".format(seed)
            job_dir.mkdir(parents=True, exist_ok=True)
            run_log = job_dir / "run.log"
            command = make_command(env_name=env_name, seed=seed, job_dir=job_dir)

            job_summary = {
                "env": env_name,
                "seed": seed,
                "job_dir": str(job_dir.relative_to(ROOT)),
                "status": "running",
                "started_at": utc_now(),
                "completed_at": None,
                "returncode": None,
                "command": "MUJOCO_GL=egl " + " ".join(command),
            }
            summary["jobs"].append(job_summary)
            write_summary(summary)

            with run_log.open("w") as log_file:
                log_file.write("$ {}\n".format(job_summary["command"]))
                log_file.flush()
                proc = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env_vars,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            job_summary["returncode"] = proc.returncode
            job_summary["completed_at"] = utc_now()
            if proc.returncode != 0:
                job_summary["status"] = "failed"
                summary["status"] = "failed"
                summary["completed_at"] = utc_now()
                summary["failure"] = {
                    "env": env_name,
                    "seed": seed,
                    "reason": "process returned {}".format(proc.returncode),
                    "run_log": str(run_log.relative_to(ROOT)),
                }
                write_summary(summary)
                return proc.returncode

            missing_files = [
                name
                for name in ("rollout.mp4", "stats.json", "command.txt", "run.log")
                if not (job_dir / name).exists()
            ]
            if missing_files:
                job_summary["status"] = "failed"
                summary["status"] = "failed"
                summary["completed_at"] = utc_now()
                summary["failure"] = {
                    "env": env_name,
                    "seed": seed,
                    "reason": "missing files: {}".format(", ".join(missing_files)),
                    "run_log": str(run_log.relative_to(ROOT)),
                }
                write_summary(summary)
                return 1

            stats_path = job_dir / "stats.json"
            with stats_path.open() as f:
                stats = json.load(f)
            validation_errors = validate_stats(stats=stats)
            if validation_errors:
                job_summary["status"] = "failed"
                job_summary["validation_errors"] = validation_errors
                summary["status"] = "failed"
                summary["completed_at"] = utc_now()
                summary["failure"] = {
                    "env": env_name,
                    "seed": seed,
                    "reason": "validation failed",
                    "errors": validation_errors,
                    "run_log": str(run_log.relative_to(ROOT)),
                }
                write_summary(summary)
                return 1

            job_summary.update(summarize_job(env_name=env_name, seed=seed, job_dir=job_dir, stats=stats))
            job_summary["status"] = "completed"
            write_summary(summary)

    summary["status"] = "completed"
    summary["completed_at"] = utc_now()
    write_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
