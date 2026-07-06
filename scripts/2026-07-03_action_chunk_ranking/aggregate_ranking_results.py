"""Aggregate action-chunk ranking rollout matrix results."""

import argparse
import json
from pathlib import Path

import numpy as np


KEYS = [
    "Success_Rate",
    "Non_Target_Collision_Any",
    "Non_Target_Collision_Step_Count",
    "Non_Target_Collision_Rate",
    "Obstacle_Guidance_Trigger_Count",
    "Ranking_Safe_Rate",
    "Ranking_Cost_Improvement",
    "Ranking_Distance_Improvement",
    "Ranking_First_Distance",
    "Ranking_Best_Distance",
    "Ranking_Skipped_Rate",
]


def load_stats(path):
    with path.open() as f:
        payload = json.load(f)
    return payload["average"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/robomimic/eval/action_chunk_ranking")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main(args):
    input_dir = Path(args.input_dir)
    rows = []
    for stats_path in sorted(input_dir.glob("*/*/seed_*.json")):
        env_name = stats_path.parents[1].name
        condition = stats_path.parent.name
        seed = stats_path.stem.replace("seed_", "")
        stats = load_stats(stats_path)
        row = dict(env=env_name, condition=condition, seed=seed)
        for key in KEYS:
            row[key] = float(stats.get(key, 0.0))
        rows.append(row)

    groups = {}
    for row in rows:
        groups.setdefault((row["env"], row["condition"]), []).append(row)

    summary = []
    for (env_name, condition), group_rows in sorted(groups.items()):
        out = dict(env=env_name, condition=condition, seeds=len(group_rows))
        for key in KEYS:
            vals = [row[key] for row in group_rows]
            out[key] = float(np.mean(vals))
            out[key + "_std"] = float(np.std(vals))
        summary.append(out)

    print("| env | condition | seeds | success | collision_any | collision_steps | ranking_safe | skipped | cost_improve | dist_improve |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        print(
            "| {env} | {condition} | {seeds} | {success:.3f} | {coll_any:.3f} | {coll_steps:.2f} | {safe:.3f} | {skipped:.3f} | {cost:.6g} | {dist:.4f} |".format(
                env=row["env"],
                condition=row["condition"],
                seeds=row["seeds"],
                success=row["Success_Rate"],
                coll_any=row["Non_Target_Collision_Any"],
                coll_steps=row["Non_Target_Collision_Step_Count"],
                safe=row["Ranking_Safe_Rate"],
                skipped=row["Ranking_Skipped_Rate"],
                cost=row["Ranking_Cost_Improvement"],
                dist=row["Ranking_Distance_Improvement"],
            )
        )

    payload = dict(rows=rows, summary=summary)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(payload, f, indent=2)
        print("Wrote aggregate results to {}".format(output))


if __name__ == "__main__":
    main(parse_args())
