"""Compare compact Bezier execution with archived rectangle execution."""

import argparse
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/tabletop_local_planner_2026-09-11"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def bypass_metrics(folder):
    chunks = read(folder / "chunks.json")
    trace = read(folder / "trace.json")
    start = next(c for c in chunks if c["planner"]["trigger"])
    end = next(
        (
            c["step"]
            for c in chunks
            if c["planner"]["reason"] == "route_complete_source_clear"
        ),
        None,
    )
    last = trace[-1]["step"] if end is None else end
    actual = np.array([t["eef"] for t in trace[start["step"] : last + 1]])
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(actual, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-9]
    actual_unique = actual[keep]
    distance_unique = distance[keep]
    sample_s = np.arange(0, distance[-1], 0.01)
    sampled = np.column_stack(
        [np.interp(sample_s, distance_unique, actual_unique[:, i]) for i in range(3)]
    )
    vectors = np.diff(sampled, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    unit = vectors / np.maximum(norms[:, None], 1e-12)
    angles = np.rad2deg(np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1)))
    route = start["planner"]["new_route"]
    return {
        "trigger": start["step"],
        "handoff": end,
        "route": route,
        "actual": actual,
        "planned_length_m": route["length"],
        "actual_length_m": float(distance[-1]),
        "turn_p95_deg_10mm": float(np.percentile(angles, 95)) if len(angles) else None,
        "turn_max_deg_10mm": float(max(angles)) if len(angles) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    summary = []
    checks = []
    videos = [
        "# Rectangle versus compact Bezier videos",
        "",
        "Actual execution MP4s; repeats listed separately.",
        "",
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, ep in zip(axs, (0, 25, 28)):
        old = (
            OLD / ("preflight" if ep == 0 else "development") / f"ep{ep:02d}/r0/planner"
        )
        folder = out / ("gpu0" if ep == 0 else "gpu1") / f"ep{ep:02d}/r0/planner"
        before, after = bypass_metrics(old), bypass_metrics(folder)
        a, b = read(old / "result.json"), read(folder / "result.json")
        assert (
            a["reset_audit"]["initial_obs_hashes"]
            == b["reset_audit"]["initial_obs_hashes"]
        )
        assert before["trigger"] == after["trigger"]
        assert np.allclose(
            before["route"]["points"], after["route"]["original_points"], rtol=0, atol=0
        )
        with (
            np.load(old / "trajectory.npz") as x,
            np.load(folder / "trajectory.npz") as y,
        ):
            n = before["trigger"]
            assert np.array_equal(x["actions"][:n], y["actions"][:n])
            assert np.array_equal(x["states"][: n + 1], y["states"][: n + 1])
        for rep in range(2):
            current = folder.parent.parent / f"r{rep}/planner"
            result = read(current / "result.json")
            with np.load(current / "trajectory.npz") as x:
                assert np.array_equal(
                    x["actions"], x["output_chunks"].reshape(-1, 7)[: result["steps"]]
                )
                with imageio.get_reader(str(current / "rollout.mp4")) as reader:
                    count = sum(1 for _ in reader)
                assert count == result["video_frames"] == len(x["video_steps"])
            checks.append(
                {
                    "episode": ep,
                    "repeat": rep,
                    "frames": count,
                    "executed_actions_match": True,
                }
            )
        repeat = read(folder.parent.parent / "r1/planner/trace.json")
        assert repeat == read(folder / "trace.json")
        item = {
            "episode": ep,
            "rectangle_success": a["success"],
            "rectangle_safe": a["safe_success"],
            "bezier_success": b["success"],
            "bezier_safe": b["safe_success"],
            "rectangle": {
                k: v for k, v in before.items() if k not in ("route", "actual")
            },
            "bezier": {k: v for k, v in after.items() if k not in ("route", "actual")},
            "planned_reduction_percent": 100
            * (1 - after["planned_length_m"] / before["planned_length_m"]),
            "actual_reduction_percent": 100
            * (1 - after["actual_length_m"] / before["actual_length_m"]),
            "bezier_arcs": len(after["route"]["bezier_controls"]),
            "exact_prefix": True,
            "repeat_trace_exact": True,
        }
        summary.append(item)
        for info, label, color in (
            (before, "rectangle", "tab:gray"),
            (after, "compact Bezier", "tab:blue"),
        ):
            xy = info["actual"]
            ax.plot(xy[:, 0], xy[:, 1], label=label, color=color)
        for controls in after["route"]["bezier_controls"]:
            cp = np.array(controls)
            ax.plot(cp[:, 0], cp[:, 1], ":", color="tab:orange", alpha=0.6)
        ax.set_title(f"Layout {ep}")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.axis("equal")
        ax.legend(fontsize=8)
        videos.extend(
            [
                f"## Layout {ep}",
                "",
                f"- [Original rectangle]({old}/rollout.mp4)",
                f"- [Compact Bezier]({folder.relative_to(out)}/rollout.mp4)",
                f"- [Compact Bezier repeat]({folder.parent.parent.relative_to(out)}/r1/planner/rollout.mp4)",
                "",
            ]
        )
    fig.tight_layout()
    fig.savefig(out / "bypass_comparison.png", dpi=160)
    plt.close(fig)
    write(out / "summary.json", summary)
    write(
        out / "artifact_integrity.json",
        {"videos_verified": len(checks), "checks": checks},
    )
    (out / "video_index.md").write_text("\n".join(videos).rstrip() + "\n")
    lines = [
        "# Compact Bezier validation",
        "",
        "Same three development layouts, unchanged10mm clearance margin and task criterion.",
        "Six new executions; repeats excluded from primary counts. Archived rectangle videos remain unchanged.",
        "",
        "| Layout | Safe rectangle / Bezier | Planned length (cm) | Actual bypass length (cm) | Actual turn p95 (degrees per10mm) |",
        "|---|---|---|---|---|",
    ]
    for s in summary:
        a, b = s["rectangle"], s["bezier"]
        lines.append(
            f"| {s['episode']} | {int(s['rectangle_safe'])} / {int(s['bezier_safe'])} | {a['planned_length_m'] * 100:.1f} / {b['planned_length_m'] * 100:.1f} | {a['actual_length_m'] * 100:.1f} / {b['actual_length_m'] * 100:.1f} | {a['turn_p95_deg_10mm']:.1f} / {b['turn_p95_deg_10mm']:.1f} |"
        )
    lines += [
        "",
        "The turning statistic uses EEF paths resampled every10mm within the intervention interval.",
        "It is not a jerk metric or proof of continuous acceleration. No held-out generalization claim.",
        "Prefix states/actions and initial observations match the archived rectangle until intervention;",
        "the initial rectangle route is identical. Each new condition repeats its full recorded trace.",
        "All executed output arrays match saved actions; all six videos decode with matching frame counts.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
