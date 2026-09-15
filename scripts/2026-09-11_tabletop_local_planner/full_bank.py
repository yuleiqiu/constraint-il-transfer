"""Supervise the fixed50-layout,100-rollout comparison on two GPUs.

Runs unchanged run.py in two subprocesses. This file only manages artifacts,
progress and final verification; it does not change policy/planner parameters.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def progress(out, started, workers):
    rows = []
    for path in sorted(out.glob("shard_*/ep*/r0/*/result.json")):
        row = json.loads(path.read_text())
        row["artifact_folder"] = str(path.parent.relative_to(out))
        rows.append(row)
    pairs = {}
    for r in rows:
        pairs.setdefault(r["episode"], {})[r["condition"]] = r
    complete_pairs = {
        e: p for e, p in pairs.items() if set(p) == {"baseline", "planner"}
    }
    status = {
        "completed_rollouts": len(rows),
        "expected_rollouts": 100,
        "completed_pairs": len(complete_pairs),
        "expected_pairs": 50,
        "elapsed_seconds": time.monotonic() - started,
        "workers": {
            name: {"pid": process.pid, "exit_code": process.poll()}
            for name, process in workers
        },
        "paired_only_counts": {},
    }
    for cond in ("baseline", "planner"):
        status["paired_only_counts"][cond] = {
            key: sum(p[cond][key] for p in complete_pairs.values())
            for key in ("success", "safe_success", "collision_any", "ncr")
        }
    write(out / "progress.json", status)
    lines = [
        "# Full50 comparison execution videos",
        "",
        f"Completed {len(rows)}/100 rollouts; {len(complete_pairs)}/50 complete pairs.",
        "Each link is an actual-execution MP4, viewable in VS Code. No HTML.",
        "",
    ]
    for e, p in sorted(pairs.items()):
        lines += [f"## Layout {e:02d}", ""]
        for c, r in sorted(p.items()):
            lines.append(
                f"- [{c}]({r['artifact_folder']}/rollout.mp4): task {int(r['success'])}, safe {int(r['safe_success'])}"
            )
        lines.append("")
    (out / "video_index.md").write_text("\n".join(lines).rstrip() + "\n")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    # The launch directory can contain its README/log but must have no workers.
    if (out / "supervisor.json").exists():
        raise FileExistsError("This run has already been launched")
    out.mkdir(parents=True, exist_ok=True)
    write(
        out / "supervisor.json",
        {
            "pid": os.getpid(),
            "python": sys.executable,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "episodes": list(range(50)),
            "conditions": ["baseline", "planner"],
            "route_style": "compact_bezier",
            "horizon": 600,
            "criterion": "old strict OR released placement held0.5s",
            "scope": "Previously examined fixed bank; not a held-out generalization test",
        },
    )
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for name in (
        "run.py",
        "geometry.py",
        "runtime.py",
        "smoothing.py",
        "analyze.py",
        "full_bank.py",
    ):
        shutil.copy2(HERE / name, snapshot / name)
    started = time.monotonic()
    workers = []
    logs = []
    try:
        for gpu, lo, hi in ((0, 0, 25), (1, 25, 50)):
            name = f"shard_{lo:02d}_{hi - 1:02d}"
            log = (out / f"{name}.log").open("w")
            logs.append(log)
            command = [
                sys.executable,
                str(HERE / "run.py"),
                "--out",
                str(out / name),
                "--episodes",
                *map(str, range(lo, hi)),
                "--conditions",
                "baseline",
                "planner",
                "--repeats",
                "1",
                "--route-style",
                "compact_bezier",
            ]
            env = dict(os.environ, MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES=str(gpu))
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            workers.append((name, process))
        while True:
            status = progress(out, started, workers)
            if all(p.poll() is not None for _, p in workers):
                break
            time.sleep(15)
        if (
            any(p.returncode != 0 for _, p in workers)
            or status["completed_rollouts"] != 100
        ):
            write(
                out / "failed.json",
                {"reason": "Worker failed or incomplete matrix", **status},
            )
            raise RuntimeError("Incomplete matrix; inspect worker logs and failed.json")
        with (out / "analysis.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, str(HERE / "analyze.py"), "--out", str(out)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            write(
                out / "verification_failed.json",
                {
                    "analysis_exit_code": result.returncode,
                    "completed_rollouts": 100,
                    "note": "No unverified aggregate conclusion is substituted",
                },
            )
            raise RuntimeError("Final verification failed; inspect analysis.log")
        summary = json.loads((out / "summary.json").read_text())
        assert (
            summary["episodes"] == list(range(50))
            and summary["primary_rollouts"] == 100
        )
        # Existing analyzer's engineering gate is descriptive, not a paper claim.
        report = (out / "report.md").read_text()
        report = report.replace(
            "# Local planner development results", "# Full50 local planner comparison"
        )
        report = report.replace(
            "Previously examined engineering cases; repeats excluded from primary counts.",
            "Complete previously examined50-layout bank. One rollout per condition/layout; no held-out claim.",
        )
        report = report.replace(
            "No wider validation is implied.",
            "No independent held-out validation is implied.",
        )
        (out / "report.md").write_text(report)
        counts = summary["conditions"]
        lines = [
            "# Full50 comparison conclusion",
            "",
            "Completed100 rollouts; each has a verified actual-execution MP4.",
            "Frozen compact-Bezier planner versus source on the original50-layout bank.",
            "",
            "| Condition | Task | Safe | CR | NCR | Safe / success+contact / collision failure / NCR |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for c in ("baseline", "planner"):
            s = counts[c]
            lines.append(
                f"| {c} | {s['success']}/50 | {s['safe_success']}/50 | {s['collision_any']}/50 | {s['ncr']}/50 | {s['safe_success']} / {s['success_with_collision']} / {s['collision_failure']} / {s['ncr']} |"
            )
        lines += ["", "Paired transitions:"]
        for m, t in summary["transitions"].items():
            lines.append(
                f"- {m}: {len(t['rescues'])} rescues / {len(t['regressions'])} regressions."
            )
        lines += [
            "",
            "See summary.json for episode IDs and video_index.md for manual review.",
            "This evaluates the complete development bank, not universal safety or held-out generalization.",
            "The planner uses oracle geometry and known goal position. No parameter tuning within this matrix.",
            "Initial observations are checked within each pair; archived baseline prefixes and video artifacts are verified.",
            "No repeated-outcome claim is made for all50 layouts. Earlier0/25/28 repeats remain separate.",
        ]
        (out / "conclusion.md").write_text("\n".join(lines) + "\n")
        write(
            out / "complete.json",
            {
                "rollouts": 100,
                "layouts": 50,
                "verification_passed": True,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
