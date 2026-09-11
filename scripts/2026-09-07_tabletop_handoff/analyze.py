"""Summarize selected states, repeat stability, paired transitions and paths."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze(directory):
    rows = [json.loads(x) for x in (directory / "results.jsonl").read_text().splitlines()]
    by_key = {(r["episode"], r["condition"], r["repeat"]): r for r in rows}
    conditions = [c for c in ("baseline", "guided", "short", "long", "near_goal", "teacher")
                  if any(r["condition"] == c for r in rows)]
    summary = {"sampling_unit": "selected layout; repeated identical seeds audit replay, not independent trials",
               "conditions": {}, "pairing": {}, "repeat_audit": {}}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        summary["conditions"][condition] = {
            "runs": len(subset), "layouts": len({r["episode"] for r in subset}),
            **{name: sum(r[name] for r in subset) / len(subset)
               for name in ("success", "safe_success", "collision_any", "ncr")},
            "four_way_counts": {name: sum(r[name] for r in subset) for name in
                                ("safe_success", "success_with_collision", "collision_failure", "ncr")}}
        pairs = [(by_key[(r["episode"], "baseline", r["repeat"])], r)
                 for r in subset if (r["episode"], "baseline", r["repeat"]) in by_key]
        summary["pairing"][condition] = {name: {
            "failure_to_success": sum(not a[name] and b[name] for a, b in pairs),
            "success_to_failure": sum(a[name] and not b[name] for a, b in pairs)}
            for name in ("success", "safe_success")}
    for episode in sorted({r["episode"] for r in rows}):
        for condition in conditions:
            subset = [r for r in rows if r["episode"] == episode and r["condition"] == condition]
            if not subset:
                continue
            first_chunks = [np.asarray(r["first_source_chunk"]) for r in subset if r["first_source_chunk"] is not None]
            trace_paths = [directory / f"ep{episode:02d}_{condition}_r{r['repeat']}_trace.json" for r in subset]
            traces = [json.loads(p.read_text()) for p in trace_paths]
            summary["repeat_audit"][f"{episode}:{condition}"] = {
                "runs": len(subset),
                "outcomes": [[r["success"], r["collision_any"]] for r in subset],
                "boundary_state_max_abs": max(r["restore_audit"]["state_max_abs"] for r in subset),
                "boundary_policy_obs_max_abs": {k: max(r["restore_audit"]["obs_max_abs"][k] for r in subset)
                                                 for k in ("agent_pos", "task_pointcloud")},
                "first_chunk_repeat_max_abs": (max(float(np.max(np.abs(c - first_chunks[0]))) for c in first_chunks)
                                                if len(first_chunks) > 1 else None),
                "full_recorded_trace_equal": (all(t == traces[0] for t in traces[1:]) if len(traces) > 1 else None)}
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for condition in conditions:
            path = directory / f"ep{episode:02d}_{condition}_r0_trace.json"
            if not path.exists():
                continue
            trace = json.loads(path.read_text())
            start = by_key[(episode, condition, 0)]["start_step"]
            trace = [t for t in trace if t["step"] >= start]
            can = np.array([t["can"] for t in trace])
            line, = axes[0].plot(can[:, 0], can[:, 1], label=condition)
            contacts = np.array([t["contact"] for t in trace])
            if contacts.any():
                axes[0].scatter(can[contacts, 0], can[contacts, 1], marker="x", s=12, color=line.get_color())
            axes[1].plot([t["step"] for t in trace], [t["goal_distance"] for t in trace], label=condition)
            handoff = by_key[(episode, condition, 0)]["handoff"]
            if handoff is not None and condition in ("short", "long", "near_goal"):
                axes[0].scatter(*handoff["can"][:2], marker="o", color=line.get_color(), s=60)
        route = json.loads((directory / f"ep{episode:02d}_route.json").read_text())
        axes[0].scatter(*route["obstacle"], color="black", marker="s", label="obstacle center")
        axes[0].set(xlabel="world X (m)", ylabel="world Y (m)", title=f"Layout {episode}: carried Can path (x = contact)")
        axes[0].axis("equal")
        axes[0].legend(fontsize=8)
        axes[1].set(xlabel="absolute environment step", ylabel="Can to goal XY distance (m)", title="Task progress")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / f"ep{episode:02d}_paths.png", dpi=160)
        plt.close(fig)
    (directory / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Tabletop oracle-route diagnostic", "", "Selected transport states; repeated seeds are replay audits, not independent trials.", "",
             "| Condition | Runs | Task SR | Safe SR | CR | NCR | Four-way counts: safe / success+contact / collision failure / NCR |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for condition, item in summary["conditions"].items():
        four = " / ".join(str(v) for v in item["four_way_counts"].values())
        lines.append(f"| {condition} | {item['runs']} | {item['success']:.3f} | {item['safe_success']:.3f} | {item['collision_any']:.3f} | {item['ncr']:.3f} | {four} |")
    lines += ["", "See analysis.json for paired transitions, individual-layout repeat outcomes, boundary state / observation differences and first-chunk reproduction.", "",
              "These are oracle-route feasibility diagnostics, not learned-method performance or evidence against every local guidance method."]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
