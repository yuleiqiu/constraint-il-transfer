"""Summarize complete local-planner runs, verify videos and replay prefixes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    rows = []
    checks = []
    video_lines = [
        "# Local planner execution videos",
        "",
        "Open MP4 files directly in VS Code. No HTML.",
        "",
    ]
    for path in sorted(out.glob("*/ep*/r*/*/result.json")):
        r = json.loads(path.read_text())
        folder = path.parent
        tr = json.loads((folder / "trace.json").read_text())
        chunks = json.loads((folder / "chunks.json").read_text())
        with np.load(folder / "trajectory.npz") as a:
            assert (
                len(a["actions"]) == r["steps"] and len(a["states"]) == r["steps"] + 1
            )
            assert len(tr) == r["steps"] + 1
            # What is stored as executable output must be what actually ran.
            flat = a["output_chunks"].reshape(-1, 7)[: r["steps"]]
            assert np.array_equal(flat, a["actions"])
            for c, src, executed in zip(chunks, a["source_chunks"], a["output_chunks"]):
                if r["condition"] == "planner" and not c["planner"]["changed"]:
                    assert np.array_equal(src, executed)
            with imageio.get_reader(str(folder / "rollout.mp4")) as reader:
                count = sum(1 for _ in reader)
            assert count == r["video_frames"] == len(a["video_steps"])
            prefix = None
            if r["condition"] == "baseline":
                shard = "shard_00_24" if r["episode"] < 25 else "shard_25_49"
                old = (
                    ROOT
                    / "outputs/tabletop_guidance_reeval_2026-09-09"
                    / shard
                    / "episodes"
                    / f"ep{r['episode']:02d}"
                    / "baseline/trajectory.npz"
                )
                with np.load(old) as b:
                    n = min(len(a["actions"]), len(b["actions"]))
                    prefix = bool(
                        np.array_equal(a["actions"][:n], b["actions"][:n])
                        and np.array_equal(a["states"][: n + 1], b["states"][: n + 1])
                    )
                assert prefix, f"Archived baseline prefix differs: {folder}"
        assert r["success"] == bool(
            tr[-1]["strict_success"] or tr[-1]["hold_seconds"] >= 0.5
        )
        assert r["collision_any"] == any(t["contact"] for t in tr)
        checks.append(
            {
                "folder": str(folder.relative_to(out)),
                "frames": count,
                "saved_actions_match": True,
                "archived_baseline_prefix_exact": prefix,
            }
        )
        r["artifact_folder"] = str(folder.relative_to(out))
        r["repeat"] = int(folder.parent.name[1:])
        if folder.relative_to(out).parts[0] == "rescue_replay":
            r["repeat"] += 1
            original = (
                out / "development" / f"ep{r['episode']:02d}" / "r0" / r["condition"]
            )
            assert json.loads((original / "trace.json").read_text()) == tr
            with (
                np.load(original / "trajectory.npz") as a,
                np.load(folder / "trajectory.npz") as b,
            ):
                assert np.array_equal(a["actions"], b["actions"]) and np.array_equal(
                    a["states"], b["states"]
                )
            checks[-1]["cross_process_repeat_exact"] = True
        if r["condition"] == "planner":
            r["planner_reasons"] = dict(Counter(c["planner"]["reason"] for c in chunks))
            r["handoffs"] = sum(
                c["planner"]["reason"] == "route_complete_source_clear" for c in chunks
            )
        rows.append(r)
        video_lines.append(
            f"- Layout {r['episode']:02d}, {r['condition']}, repeat {r['repeat']}: "
            f"[video]({r['artifact_folder']}/rollout.mp4) — task {int(r['success'])}, safe {int(r['safe_success'])}"
        )
    assert rows, "No completed rows"
    primary = [r for r in rows if r["repeat"] == 0]
    episodes = sorted({r["episode"] for r in primary})
    conditions = sorted({r["condition"] for r in primary})
    summary = {
        "episodes": episodes,
        "primary_rollouts": len(primary),
        "total_rollouts": len(rows),
        "conditions": {},
    }
    for c in conditions:
        rs = [r for r in primary if r["condition"] == c]
        assert len(rs) == len(episodes)
        summary["conditions"][c] = {
            "n": len(rs),
            **{
                k: sum(int(r[k]) for r in rs)
                for k in (
                    "success",
                    "safe_success",
                    "collision_any",
                    "success_with_collision",
                    "collision_failure",
                    "ncr",
                )
            },
        }
    paired = {
        e: {r["condition"]: r for r in primary if r["episode"] == e} for e in episodes
    }
    summary["transitions"] = {}
    for metric in ("success", "safe_success"):
        summary["transitions"][metric] = {
            "rescues": [
                e
                for e, p in paired.items()
                if not p["baseline"][metric] and p["planner"][metric]
            ],
            "regressions": [
                e
                for e, p in paired.items()
                if p["baseline"][metric] and not p["planner"][metric]
            ],
        }
    s = summary["conditions"]
    summary["task_gate_pass"] = (
        s["planner"]["safe_success"] > s["baseline"]["safe_success"]
        and not summary["transitions"]["success"]["regressions"]
    )
    dump(out / "results.json", rows)
    dump(out / "summary.json", summary)
    dump(
        out / "artifact_integrity.json",
        {"videos_verified": len(checks), "checks": checks},
    )
    (out / "video_index.md").write_text("\n".join(video_lines) + "\n")
    lines = [
        "# Local planner development results",
        "",
        "Previously examined engineering cases; repeats excluded from primary counts.",
        "Criterion: old strict success OR released placement with0.5s residence.",
        "",
        "| Layout | Baseline task / safe | Planner task / safe | Planner changed chunks | Handoffs |",
        "|---|---|---|---:|---:|",
    ]
    for e, p in paired.items():
        b, l = p["baseline"], p["planner"]
        lines.append(
            f"| {e} | {int(b['success'])} / {int(b['safe_success'])} | {int(l['success'])} / {int(l['safe_success'])} | {l['planner_changed_chunks']} | {l['handoffs']} |"
        )
    lines += [
        "",
        f"Task-level engineering gate: **{'PASS' if summary['task_gate_pass'] else 'FAIL'}**.",
        "",
        "See summary.json for Task/Safe/CR/NCR, four-way partitions and paired transitions.",
        "See video_index.md for all actual-execution MP4s. No wider validation is implied.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
