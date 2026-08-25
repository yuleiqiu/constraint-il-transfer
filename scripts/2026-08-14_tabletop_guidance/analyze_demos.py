"""Summarize coverage and continuity of Tabletop demonstration data."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    lengths = []
    starts = []
    goals = []
    max_rotation_step = 0.0
    max_transport_deviation = 0.0
    with h5py.File(args.dataset, "r") as input_file:
        data = input_file["data"]
        names = sorted(data, key=lambda name: int(name.split("_")[-1]))
        for name in names:
            demo = data[name]
            lengths.append(int(demo.attrs["num_samples"]))
            starts.append(np.asarray(demo.attrs["can_start"])[:2])
            goals.append(np.asarray(demo.attrs["goal_xy"]))
            rotation = demo["abs_eef_pose_action"][:, 3:6]
            if len(rotation) > 1:
                max_rotation_step = max(max_rotation_step, float(np.linalg.norm(np.diff(rotation, axis=0), axis=1).max()))
            max_transport_deviation = max(
                max_transport_deviation,
                float(demo.attrs["max_transport_line_deviation_m"]),
            )
    starts = np.asarray(starts)
    goals = np.asarray(goals)
    summary = {
        "demos": len(lengths),
        "transitions": int(sum(lengths)),
        "length_min_mean_max": [int(min(lengths)), float(np.mean(lengths)), int(max(lengths))],
        "start_xy_min": starts.min(axis=0).tolist(),
        "start_xy_max": starts.max(axis=0).tolist(),
        "goal_xy_min": goals.min(axis=0).tolist(),
        "goal_xy_max": goals.max(axis=0).tolist(),
        "max_consecutive_rotation_vector_change_rad": max_rotation_step,
        "max_transport_line_deviation_m": max_transport_deviation,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
