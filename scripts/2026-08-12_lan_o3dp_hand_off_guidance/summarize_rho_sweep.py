"""Combine rho trajectory shards and report causal versus late separation."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--extreme-input", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_causal_window(row, baseline, guided):
    common = min(len(baseline), len(guided))
    xy = np.linalg.norm(guided[:common, :2] - baseline[:common, :2], axis=1)
    active_step = row["outcome"]["first_guidance_active_step"]
    if active_step is None:
        early_max = 0.0
    else:
        early_end = min(common, active_step + 65)
        early_max = float(np.max(xy[active_step:early_end]))
    collision_steps = [
        value
        for value in (
            row["baseline_outcome"]["first_collision_step"],
            row["outcome"]["first_collision_step"],
        )
        if value is not None
    ]
    comparison_end = min(collision_steps) if collision_steps else common - 1
    row["causal_window"] = {
        "max_xy_first_64_steps_after_activation_m": early_max,
        "earliest_collision_step_either_trajectory": (
            comparison_end if collision_steps else None
        ),
        "max_xy_through_earliest_collision_m": float(
            np.max(xy[: min(common, comparison_end + 1)])
        ),
    }


def main():
    args = parse_args()
    if args.out_dir.exists() and not args.overwrite:
        raise FileExistsError("Output already exists: {}".format(args.out_dir))
    args.out_dir.mkdir(parents=True, exist_ok=args.overwrite)

    rows = []
    manifests = []
    plot_paths = {}
    for input_dir in args.inputs:
        manifests.append(json.loads((input_dir / "manifest.json").read_text()))
        shard_rows = json.loads((input_dir / "results.json").read_text())
        rows.extend(shard_rows)
        for episode in sorted({row["episode"] for row in shard_rows}):
            trajectories = np.load(
                input_dir / "episode_{:03d}_trajectories.npz".format(episode)
            )
            baseline = trajectories["baseline"]
            for row in [item for item in shard_rows if item["episode"] == episode]:
                guided = trajectories["rho_{}".format(row["rho"])]
                add_causal_window(row, baseline, guided)
            plot_paths[episode] = str(
                (
                    input_dir
                    / "episode_{:03d}_trajectory.png".format(episode)
                ).resolve()
            )

    reference = manifests[0]
    for manifest in manifests[1:]:
        for key in (
            "agent",
            "environment",
            "seed",
            "rhos",
            "horizon",
            "xy_only",
            "hand_collision_enabled",
            "obstacle_cost_type",
            "safety_distance_m",
            "obstacle_cost_smoothing_m",
        ):
            if manifest[key] != reference[key]:
                raise ValueError("Shard manifest mismatch for {}".format(key))
    episodes = sorted({row["episode"] for row in rows})
    rhos = reference["rhos"]
    if len(rows) != len(episodes) * len(rhos):
        raise ValueError("Incomplete episode/rho matrix")

    aggregate = []
    for rho in rhos:
        selected = [row for row in rows if row["rho"] == rho]
        max_separations = np.asarray(
            [row["separation"]["max_xy_separation_m"] for row in selected]
        )
        early_separations = np.asarray(
            [
                row["causal_window"][
                    "max_xy_first_64_steps_after_activation_m"
                ]
                for row in selected
            ]
        )
        aggregate.append(
            {
                "rho": rho,
                "episodes": len(selected),
                "task_successes": sum(row["outcome"]["success"] for row in selected),
                "collisions": sum(
                    row["outcome"]["collision_any"] for row in selected
                ),
                "states_max_xy_ge_1cm": int(np.count_nonzero(max_separations >= 0.01)),
                "states_max_xy_ge_2cm": int(np.count_nonzero(max_separations >= 0.02)),
                "states_sustained_xy_ge_1cm_20_steps": sum(
                    row["separation"]["longest_run_xy_ge_10mm"] >= 20
                    for row in selected
                ),
                "median_max_xy_separation_m": float(np.median(max_separations)),
                "max_early_64_xy_separation_m": float(np.max(early_separations)),
            }
        )

    baseline_by_episode = {
        row["episode"]: row["baseline_outcome"] for row in rows
    }
    baseline_task = sum(item["success"] for item in baseline_by_episode.values())
    baseline_collision = sum(
        item["collision_any"] for item in baseline_by_episode.values()
    )

    extreme_rows = []
    if args.extreme_input is not None:
        extreme_rows = json.loads((args.extreme_input / "results.json").read_text())
        for episode in sorted({row["episode"] for row in extreme_rows}):
            trajectories = np.load(
                args.extreme_input
                / "episode_{:03d}_trajectories.npz".format(episode)
            )
            baseline = trajectories["baseline"]
            for row in [item for item in extreme_rows if item["episode"] == episode]:
                add_causal_window(
                    row,
                    baseline,
                    trajectories["rho_{}".format(row["rho"])],
                )

    (args.out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    (args.out_dir / "extreme_results.json").write_text(
        json.dumps(extreme_rows, indent=2) + "\n"
    )
    output_manifest = {
        "inputs": [str(path.resolve()) for path in args.inputs],
        "source_manifest": reference,
        "episodes": episodes,
        "plots": plot_paths,
        "extreme_input": (
            str(args.extreme_input.resolve()) if args.extreme_input else None
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n"
    )

    lines = [
        "# D2 hand-off XY rho trajectory diagnostic",
        "",
        "## Conclusion",
        "",
        "Increasing `rho` can produce numerically and eventually visually large "
        "executed-trajectory separation, but it does **not** produce a controlled "
        "or useful avoidance displacement in these fixed states. The large "
        "separation is delayed closed-loop divergence, concentrated in H600 "
        "failures, rather than an immediate obstacle-clearing response.",
        "",
        "Across all 20 episode/rho combinations, the largest XY separation "
        "within the first 64 executed steps after guidance first activates is "
        "only `{:.2f} mm`. In episodes 0 and 4, collision occurs before any "
        "1 cm separation at every `rho <= 1`. Episode 0 collides five steps after "
        "activation, while its first 1 cm separation does not occur until "
        "steps 256--416. Episode 4 first collides around steps 92--96; its "
        "1 cm separation, when present, begins only around steps 254--283.".format(
            1000
            * max(
                row["causal_window"][
                    "max_xy_first_64_steps_after_activation_m"
                ]
                for row in rows
            )
        ),
        "",
        "The aggregate task count remains `{}/4` for baseline and `{}/4` at "
        "every tested rho. Collision labels fluctuate non-monotonically. The "
        "formally positive episode 35 never reaches 1 cm separation at any "
        "rho (maximum `0.82 cm`), confirming that its collision clearance is a "
        "contact-boundary change rather than a visible bypass.".format(
            baseline_task,
            aggregate[0]["task_successes"],
        ),
        "",
        "An extreme boundary probe confirms the distinction. At `rho=10`, "
        "episode 4 exceeds 1 cm separation before collision (`1.19 cm` maximum "
        "through collision), so sufficiently large `rho` can force an early, "
        "visible displacement. It still collides and does not change the task "
        "outcome. Episode 0 remains effectively unchanged before its very early "
        "collision even at `rho=10` (`0.01 mm` maximum through collision).",
        "",
        "Decision: increasing `rho` alone does not rescue the current XY point "
        "guidance. It can amplify small interventions into late trajectory "
        "divergence, but no tested scale creates an early, monotonic, "
        "task-preserving avoidance route. Do not select a larger `rho` from "
        "this sweep for a formal rollout matrix.",
        "",
        "## Protocol",
        "",
        "D2 (`PickPlaceBreadCerealCan`) with only the Panda hand shell disabled; "
        "finger physics retained. Four fixed seed-812 states (episodes "
        "0/4/8/35), H600, XY-only Huberized hinge, `Q=0.04 m`, "
        "`epsilon=0.01 m`, and `rho in {0.01, 0.03, 0.1, 0.3, 1.0}`. "
        "Each state uses one baseline trajectory; every guided trajectory is "
        "compared against that baseline at identical environment steps.",
        "A separate boundary probe tests `rho in {3, 10}` on episodes 0 and 4; "
        "it is not included in the four-state aggregate.",
        "",
        "## Aggregate",
        "",
        "Baseline: Task `{}/4`, collision `{}/4` in this same-process replay.".format(
            baseline_task, baseline_collision
        ),
        "",
        "| rho | Task | Collision | Max >=1 cm | Sustained >=1 cm | Median max XY | Max early-64 XY |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate:
        lines.append(
            "| {:g} | {}/4 | {}/4 | {}/4 | {}/4 | {:.2f} cm | {:.2f} mm |".format(
                item["rho"],
                item["task_successes"],
                item["collisions"],
                item["states_max_xy_ge_1cm"],
                item["states_sustained_xy_ge_1cm_20_steps"],
                100 * item["median_max_xy_separation_m"],
                1000 * item["max_early_64_xy_separation_m"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-state timing",
            "",
            "| Episode | rho | Task | Collision | First active | Earliest collision | First >=1 cm | Max XY | Max through collision |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["episode"], item["rho"])):
        lines.append(
            "| {} | {:g} | {} | {} | {} | {} | {} | {:.2f} cm | {:.2f} mm |".format(
                row["episode"],
                row["rho"],
                int(row["outcome"]["success"]),
                int(row["outcome"]["collision_any"]),
                row["outcome"]["first_guidance_active_step"],
                row["causal_window"]["earliest_collision_step_either_trajectory"],
                row["separation"]["first_step_xy_ge_10mm"],
                100 * row["separation"]["max_xy_separation_m"],
                1000
                * row["causal_window"]["max_xy_through_earliest_collision_m"],
            )
        )
    if extreme_rows:
        lines.extend(
            [
                "",
                "## Extreme-rho boundary probe",
                "",
                "| Episode | rho | Task | Collision | First active | Earliest collision | First >=1 cm | Max XY | Max through collision | Pos clips |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(
            extreme_rows, key=lambda item: (item["episode"], item["rho"])
        ):
            lines.append(
                "| {} | {:g} | {} | {} | {} | {} | {} | {:.2f} cm | {:.2f} mm | {} |".format(
                    row["episode"],
                    row["rho"],
                    int(row["outcome"]["success"]),
                    int(row["outcome"]["collision_any"]),
                    row["outcome"]["first_guidance_active_step"],
                    row["causal_window"][
                        "earliest_collision_step_either_trajectory"
                    ],
                    row["separation"]["first_step_xy_ge_10mm"],
                    100 * row["separation"]["max_xy_separation_m"],
                    1000
                    * row["causal_window"][
                        "max_xy_through_earliest_collision_m"
                    ],
                    row["outcome"]["position_action_clip_count"],
                )
            )
    lines.extend(["", "## Trajectory plots", ""])
    for episode in episodes:
        lines.append(
            "- Episode {}: [{}]({})".format(
                episode,
                Path(plot_paths[episode]).name,
                plot_paths[episode],
            )
        )
    lines.extend(
        [
            "",
            "Position action clipping is not sufficient to explain the late "
            "divergence: episode 0 at `rho=0.03` reaches `8.76 cm` maximum "
            "separation with zero position clips, while several successful "
            "baseline/guided trajectories share nearly identical pre-existing "
            "position clipping counts. The diagnostic therefore attributes "
            "the large late difference to closed-loop amplification, not a "
            "clean direct displacement or solely to clipping.",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
