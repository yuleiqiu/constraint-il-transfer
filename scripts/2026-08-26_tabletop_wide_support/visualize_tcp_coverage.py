"""Overlay achieved TCP transport routes and summarize broad-route coverage."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


WIDE_THRESHOLD_M = 0.065
NEAR_DIRECT_THRESHOLD_M = 0.04
PROXY_EXCLUSION_RADIUS_M = 0.06
PROXY_OBSTACLE_OFFSETS_M = (-0.0175, 0.0175)


def decode(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def segment_point_distance(start, end, point):
    delta = end - start
    denom = float(delta @ delta)
    if denom <= 1e-12:
        return float(np.linalg.norm(start - point))
    alpha = float(np.clip(((point - start) @ delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(start + alpha * delta - point))


def route_proxy_clearance(aligned_xy, obstacle):
    return min(
        segment_point_distance(start, end, obstacle)
        for start, end in zip(aligned_xy[:-1], aligned_xy[1:])
    ) - PROXY_EXCLUSION_RADIUS_M


def read_routes(path):
    routes = []
    with h5py.File(path, "r") as source:
        train = set(decode(source["mask/train"][:]))
        valid = set(decode(source["mask/valid"][:]))
        for name in sorted(source["data"], key=lambda item: int(item.split("_")[-1])):
            demo = source[f"data/{name}"]
            phases = np.asarray(decode(demo["phases"][:]))
            indices = np.flatnonzero(phases == "transport")
            if len(indices) < 2:
                raise AssertionError(f"{name} has fewer than two transport samples")
            start = np.asarray(demo["obs/agent_pos"][indices[0], :3], dtype=np.float64)
            achieved = np.asarray(demo["next_obs/agent_pos"][indices, :3], dtype=np.float64)
            goal = achieved[-1]
            delta = goal[:2] - start[:2]
            direct_distance = float(np.linalg.norm(delta))
            tangent = delta / direct_distance
            normal = np.asarray([-tangent[1], tangent[0]])
            relative = achieved[:, :2] - start[:2]
            aligned = np.column_stack((relative @ tangent, relative @ normal))
            signed_peak = float(aligned[np.argmax(np.abs(aligned[:, 1])), 1])
            routes.append(
                {
                    "name": name,
                    "split": "train" if name in train else "valid",
                    "world_xy": achieved[:, :2],
                    "aligned_xy": aligned,
                    "direct_distance_m": direct_distance,
                    "signed_peak_m": signed_peak,
                }
            )
        if {route["name"] for route in routes} != train | valid:
            raise AssertionError("route names do not match train/valid masks")
    return routes


def summarize(routes):
    result = {}
    for split in ("train", "valid", "all"):
        selected = routes if split == "all" else [route for route in routes if route["split"] == split]
        peaks = np.asarray([route["signed_peak_m"] for route in selected])
        proxy = {}
        for offset in PROXY_OBSTACLE_OFFSETS_M:
            clearances = []
            for route in selected:
                obstacle = np.asarray([0.5 * route["direct_distance_m"], offset])
                clearances.append(route_proxy_clearance(route["aligned_xy"], obstacle))
            clearances = np.asarray(clearances)
            proxy[f"obstacle_offset_{offset:+.4f}"] = {
                "safe": int(np.count_nonzero(clearances > 0.0)),
                "collision": int(np.count_nonzero(clearances <= 0.0)),
                "minimum_clearance_m": float(clearances.min()),
                "median_clearance_m": float(np.median(clearances)),
            }
        result[split] = {
            "routes": len(selected),
            "upper_wide": int(np.count_nonzero(peaks > WIDE_THRESHOLD_M)),
            "lower_wide": int(np.count_nonzero(peaks < -WIDE_THRESHOLD_M)),
            "near_direct": int(np.count_nonzero(np.abs(peaks) < NEAR_DIRECT_THRESHOLD_M)),
            "moderate": int(
                np.count_nonzero(
                    (np.abs(peaks) >= NEAR_DIRECT_THRESHOLD_M)
                    & (np.abs(peaks) <= WIDE_THRESHOLD_M)
                )
            ),
            "signed_peak_min_m": float(peaks.min()),
            "signed_peak_max_m": float(peaks.max()),
            "signed_peak_mean_m": float(peaks.mean()),
            "counterfactual_proxy": proxy,
        }
    return result


def route_color(peak):
    if peak > WIDE_THRESHOLD_M:
        return "tab:blue"
    if peak < -WIDE_THRESHOLD_M:
        return "tab:orange"
    if abs(peak) < NEAR_DIRECT_THRESHOLD_M:
        return "0.45"
    return "tab:green"


def plot(routes, summary, output):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    world, aligned, histogram = axes
    for route in routes:
        color = route_color(route["signed_peak_m"])
        linestyle = "-" if route["split"] == "train" else "--"
        alpha = 0.32 if route["split"] == "train" else 0.85
        linewidth = 0.9 if route["split"] == "train" else 1.5
        world.plot(*route["world_xy"].T, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        aligned.plot(*route["aligned_xy"].T, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)

    world.set(
        title="Achieved TCP transport routes — world XY",
        xlabel="world x (m)",
        ylabel="world y (m)",
        xlim=(-0.19, 0.19),
        ylim=(-0.28, 0.29),
        aspect="equal",
    )
    world.grid(alpha=0.2)

    median_distance = float(np.median([route["direct_distance_m"] for route in routes]))
    for offset in PROXY_OBSTACLE_OFFSETS_M:
        aligned.add_patch(
            Circle(
                (0.5 * median_distance, offset),
                PROXY_EXCLUSION_RADIUS_M,
                facecolor="tab:red",
                edgecolor="darkred",
                alpha=0.10,
                linestyle="--",
            )
        )
    aligned.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    aligned.set(
        title="Same routes aligned to each direct transport axis",
        xlabel="direct-axis progress (m)",
        ylabel="signed lateral TCP offset (m)",
        xlim=(-0.01, 0.52),
        ylim=(-0.18, 0.18),
    )
    aligned.grid(alpha=0.2)

    train_peaks = [route["signed_peak_m"] for route in routes if route["split"] == "train"]
    valid_peaks = [route["signed_peak_m"] for route in routes if route["split"] == "valid"]
    bins = np.linspace(-0.17, 0.17, 18)
    histogram.hist(train_peaks, bins=bins, alpha=0.65, label="train (90)", color="tab:purple")
    histogram.hist(valid_peaks, bins=bins, histtype="step", linewidth=2.0, label="valid (10)", color="black")
    for value in (-WIDE_THRESHOLD_M, -NEAR_DIRECT_THRESHOLD_M, NEAR_DIRECT_THRESHOLD_M, WIDE_THRESHOLD_M):
        histogram.axvline(value, color="0.35", linestyle="--", linewidth=0.8)
    train = summary["train"]
    histogram.set_title(
        "Signed lateral peak coverage\n"
        f"train: upper-wide {train['upper_wide']}, lower-wide {train['lower_wide']}, "
        f"near-direct {train['near_direct']}"
    )
    histogram.set(xlabel="signed peak offset (m)", ylabel="route count")
    histogram.legend()
    histogram.grid(axis="y", alpha=0.2)

    figure.suptitle(
        "Obstacle-free Tabletop wide-support demonstrations\n"
        "solid=train, dashed=validation; red disks are post-hoc mirrored RaisedMilk exclusion proxies",
        fontsize=13,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    routes = read_routes(args.dataset)
    summary = summarize(routes)
    plot(routes, summary, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
