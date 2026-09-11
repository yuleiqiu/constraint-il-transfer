"""Apply the user-confirmed union criterion at the earliest recorded success."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/tabletop_guidance_reeval_2026-09-09"


def main():
    rows = []
    for row in json.loads((BASE / "placement_rescore/results.json").read_text()):
        ep, condition = row["episode"], row["condition"]
        shard = "shard_00_24" if ep < 25 else "shard_25_49"
        folder = BASE / shard / "episodes" / f"ep{ep:02d}" / condition
        trace = json.loads((folder / "trace.json").read_text())
        old = next((t["step"] for t in trace if t["success"]), None)
        new = row["success_step"]
        times = [s for s in (old, new) if s is not None]
        success = bool(times)
        stop = min(times) if times else trace[-1]["step"]
        collision = next(t["collision_any"] for t in trace if t["step"] == stop)
        assert success == (row["old_success"] or row["status"] == "success")
        assert success or stop == 600
        rows.append(
            dict(
                episode=ep,
                condition=condition,
                success=success,
                safe_success=success and not collision,
                collision_any=collision,
                success_with_collision=success and collision,
                collision_failure=not success and collision,
                ncr=not success and not collision,
                stop_step=stop,
                old_success_step=old,
                placement_success_step=new,
            )
        )
    summary = {
        "criterion": "old strict OR released placement held0.5s",
        "conditions": {},
    }
    keys = [
        "success",
        "safe_success",
        "collision_any",
        "success_with_collision",
        "collision_failure",
        "ncr",
    ]
    for condition in ("baseline", "guided"):
        rs = [r for r in rows if r["condition"] == condition]
        summary["conditions"][condition] = {
            "n": len(rs),
            **{k: sum(r[k] for r in rs) for k in keys},
        }
    pairs = {r["episode"]: {} for r in rows}
    for r in rows:
        pairs[r["episode"]][r["condition"]] = r
    summary["paired"] = {}
    for metric in ("success", "safe_success"):
        summary["paired"][metric] = {
            "rescues": [
                ep
                for ep, p in pairs.items()
                if not p["baseline"][metric] and p["guided"][metric]
            ],
            "regressions": [
                ep
                for ep, p in pairs.items()
                if p["baseline"][metric] and not p["guided"][metric]
            ],
        }
    out = BASE / "inclusive_rescore"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Inclusive success rescore",
        "",
        "Human-confirmed2026-09-11: old strict success OR released placement held0.5s.",
        "Offline reconstruction only; original traces and scoring reports are preserved.",
        "Stop at the earliest satisfied predicate and count contact only through that boundary.",
        "All50 pairs have resolved outcomes; missing residence after old success is unnecessary.",
        "",
        "| Condition | Task | Safe | CR | NCR | Safe / success+contact / collision failure / NCR |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for c, s in summary["conditions"].items():
        lines.append(
            f"| {c} | {s['success']}/50 | {s['safe_success']}/50 | {s['collision_any']}/50 | {s['ncr']}/50 | {s['safe_success']} / {s['success_with_collision']} / {s['collision_failure']} / {s['ncr']} |"
        )
    lines += ["", "Paired guidance transitions:"]
    for m, p in summary["paired"].items():
        lines.append(
            f"- {m}: {len(p['rescues'])} rescues / {len(p['regressions'])} regressions."
        )
    lines += [
        "",
        "This fixed, previously examined bank supports a bounded comparison of the original",
        "guidance setting. It does not establish that all avoidance interventions fail.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
