"""Evaluate local policy-manifold feasibility on saved pre-collision states."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_DIR = SCRIPT_DIR.parent / "guided_denoising"
for directory in (SCRIPT_DIR, GUIDED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common as guided_common  # noqa: E402
from manifold_feasibility import (  # noqa: E402
    ManifoldConfig,
    closest_target_progress,
    minimum_segment_clearance_xy,
    reconstruct_positions,
    solve_manifold_projection,
)
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


DEFAULT_STATE_BANKS = [
    Path("outputs/task_constrained_guidance/one_chunk_lead_states/states.hdf5"),
    Path(
        "outputs/task_constrained_guidance/"
        "one_chunk_lead_states_seed601_602/states.hdf5"
    ),
    Path(
        "outputs/task_constrained_guidance/"
        "one_chunk_lead_states_seed603_607/states.hdf5"
    ),
]


def _text_value(item, name):
    value = item[name][()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_state(item, bank_name):
    ep_meta = _text_value(item, "ep_meta")
    return {
        "state_id": "{}:{}".format(bank_name, item.name.rsplit("/", 1)[-1]),
        "env": str(item.attrs["env"]),
        "seed": int(item.attrs["seed"]),
        "episode": int(item.attrs["episode"]),
        "chunk": int(item.attrs["chunk"]),
        "start_step": int(item.attrs["start_step"]),
        "noise_seed": int(item.attrs["noise_seed"]),
        "state": {
            "states": item["sim_state"][()],
            "model": _text_value(item, "model_xml"),
            **(
                {"ep_meta": ep_meta}
                if ep_meta
                else {}
            ),
        },
        "obs": {key: dataset[()] for key, dataset in item["obs"].items()},
        "centers": item["obstacle_centers"][()],
        "radii": item["obstacle_radii"][()],
        "obstacle_names": [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in item["obstacle_names"][()]
        ],
        "target_position": item["target_position"][()]
        if "target_position" in item
        else None,
    }


def sample_policy_chunks(policy, saved, centers, radii, sample_count):
    context = guided_common.make_guidance_context(
        policy,
        saved["obs"],
        centers,
        radii,
        scale=0.0,
        clearance_margin=0.0,
    )
    seeds = [saved["noise_seed"]]
    seeds.extend(saved["noise_seed"] + 100_000 + index for index in range(sample_count - 1))
    chunks = []
    for noise_seed in seeds:
        _, raw, _ = guided_common.sample_action_chunk(
            policy,
            saved["obs"],
            context,
            noise_seed=noise_seed,
        )
        chunks.append(raw)
    return seeds, np.asarray(chunks, dtype=np.float64)


def raw_sample_statistics(
    start_position,
    chunks,
    positions,
    target_position,
    centers,
    radii,
    nominal_progress,
    progress_tolerance,
    clearance_margin,
):
    progress = np.asarray(
        [
            closest_target_progress(start_position, points, target_position)
            for points in positions
        ]
    )
    clearance = np.asarray(
        [
            minimum_segment_clearance_xy(start_position, points, centers, radii)
            for points in positions
        ]
    )
    progress_preserving = progress >= nominal_progress - progress_tolerance
    safe = clearance >= clearance_margin
    joint = progress_preserving & safe
    return {
        "sample_count": len(chunks),
        "progress_preserving_count": int(np.count_nonzero(progress_preserving)),
        "safe_count": int(np.count_nonzero(safe)),
        "safe_progress_preserving_count": int(np.count_nonzero(joint)),
        "safe_progress_preserving_rate": float(np.mean(joint)),
        "best_clearance_progress_preserving_m": float(
            np.max(clearance[progress_preserving])
        )
        if np.any(progress_preserving)
        else None,
        "maximum_progress_m": float(np.max(progress)),
        "minimum_progress_m": float(np.min(progress)),
        "maximum_clearance_m": float(np.max(clearance)),
        "minimum_clearance_m": float(np.min(clearance)),
    }


def summarize(rows, margins):
    by_margin = {}
    for margin in margins:
        key = str(float(margin))
        selected = [row["margins"][key] for row in rows]
        projected = [item for item in selected if item["projection"]["feasible"]]
        by_margin[key] = {
            "states": len(selected),
            "states_with_raw_safe_progress_sample": int(
                sum(item["raw_samples"]["safe_progress_preserving_count"] > 0 for item in selected)
            ),
            "raw_safe_progress_state_rate": float(
                np.mean(
                    [
                        item["raw_samples"]["safe_progress_preserving_count"] > 0
                        for item in selected
                    ]
                )
            )
            if selected
            else None,
            "projection_feasible_states": len(projected),
            "projection_feasible_rate": len(projected) / len(selected)
            if selected
            else None,
            "projection_only_rescues": int(
                sum(
                    item["raw_samples"]["safe_progress_preserving_count"] == 0
                    and item["projection"]["feasible"]
                    for item in selected
                )
            ),
            "median_feasible_support_ratio": float(
                np.median(
                    [item["projection"]["support_ratio"] for item in projected]
                )
            )
            if projected
            else None,
            "median_feasible_max_waypoint_displacement_cm": float(
                np.median(
                    [
                        item["projection"]["maximum_waypoint_displacement_m"] * 100.0
                        for item in projected
                    ]
                )
            )
            if projected
            else None,
        }
    return {
        "states": len(rows),
        "margins": by_margin,
        "median_effective_rank": float(
            np.median([row["manifold"]["effective_rank"] for row in rows])
        )
        if rows
        else None,
        "median_condition_number": float(
            np.median([row["manifold"]["condition_number"] for row in rows])
        )
        if rows
        else None,
    }


def write_report(path, summary, args):
    lines = [
        "# Policy-Manifold Fixed-State Feasibility Diagnostic",
        "",
        "This is an upper-bound diagnostic, not a rollout method. It asks whether",
        "a progress-preserving collision-free eight-step XY trajectory exists",
        "inside a local support ellipsoid estimated from frozen-policy samples.",
        "No semantic task phase, simulator branching, policy update, or execution",
        "of the projected actions is used.",
        "",
        "## Setup",
        "",
        "- states: {} pre-target states captured one chunk before a real collision".format(
            summary["states"]
        ),
        "- policy samples per state: {}".format(args.samples),
        "- support: {:.0f}th percentile shrinkage-covariance ellipsoid".format(
            args.support_quantile * 100.0
        ),
        "- support squared-radius multiplier: {:.2f}".format(
            args.support_multiplier
        ),
        "- covariance shrinkage: {}; variance floor: {:.2f} mm".format(
            args.shrinkage,
            args.variance_floor * 1000.0,
        ),
        "- target-progress tolerance: {:.2f} mm".format(
            args.progress_tolerance * 1000.0
        ),
        "",
        "The geometry is the existing oracle XY EEF-point / object-circle proxy.",
        "The clearance-margin sweep tests support feasibility before implementing",
        "a whole-gripper or whole-body SDF.",
        "",
        "## Results",
        "",
        "| expanded boundary | raw samples contain safe+progress path | manifold projection feasible | projection-only rescues | median feasible support ratio | median max waypoint shift |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for margin in args.clearance_margins:
        item = summary["margins"][str(float(margin))]
        lines.append(
            "| {:.1f} cm | {}/{} ({:.1%}) | {}/{} ({:.1%}) | {} | {} | {} |".format(
                margin * 100.0,
                item["states_with_raw_safe_progress_sample"],
                item["states"],
                item["raw_safe_progress_state_rate"],
                item["projection_feasible_states"],
                item["states"],
                item["projection_feasible_rate"],
                item["projection_only_rescues"],
                "{:.3f}".format(item["median_feasible_support_ratio"])
                if item["median_feasible_support_ratio"] is not None
                else "—",
                "{:.2f} cm".format(
                    item["median_feasible_max_waypoint_displacement_cm"]
                )
                if item["median_feasible_max_waypoint_displacement_cm"] is not None
                else "—",
            )
        )
    lines.extend(
        [
            "",
            "Median covariance effective rank: {:.2f}; median condition number: {:.1f}.".format(
                summary["median_effective_rank"],
                summary["median_condition_number"],
            ),
            "",
            "## Interpretation contract",
            "",
            "- A raw-sample success means ordinary policy sampling already contains",
            "  a safe, progress-preserving path for that fixed observation.",
            "- A projection-only rescue means the fitted local ellipsoid connects to",
            "  such a path even though the finite sample set did not contain one.",
            "- Feasibility does not imply that OSC will execute the path accurately",
            "  or that a full rollout will succeed.",
            "- Infeasibility is evidence against this local, phase-free",
            "  policy-support mechanism for the selected collision regime.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def plot_primary_margin(path, rows, primary_margin):
    key = str(float(primary_margin))
    labels = [row["state_id"].split(":", 1)[-1] for row in rows]
    baseline = [row["baseline_min_clearance_m"] * 100.0 for row in rows]
    best_raw = []
    projected = []
    feasible = []
    for row in rows:
        item = row["margins"][key]
        raw_value = item["raw_samples"]["best_clearance_progress_preserving_m"]
        best_raw.append(raw_value * 100.0 if raw_value is not None else np.nan)
        projected.append(item["projection"]["corrected_min_clearance_m"] * 100.0)
        feasible.append(item["projection"]["feasible"])

    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(max(9.0, len(rows) * 0.5), 4.8))
    axis.axhline(
        primary_margin * 100.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="required expanded-boundary clearance",
    )
    axis.scatter(x, baseline, label="nominal", marker="x", s=45)
    axis.scatter(x, best_raw, label="best progress-preserving raw sample", s=35)
    axis.scatter(
        x,
        projected,
        label="manifold projection",
        s=45,
        facecolors=["tab:green" if value else "none" for value in feasible],
        edgecolors=["tab:green" if value else "tab:red" for value in feasible],
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("minimum swept XY clearance (cm)")
    axis.set_title(
        "Policy-manifold feasibility at {:.1f} cm expansion".format(
            primary_margin * 100.0
        )
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=guided_common.DEFAULT_AGENT)
    parser.add_argument(
        "--states",
        nargs="+",
        type=Path,
        default=DEFAULT_STATE_BANKS,
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--clearance-margins", nargs="+", type=float, default=[0.0, 0.01])
    parser.add_argument("--progress-tolerance", type=float, default=5e-4)
    parser.add_argument("--support-quantile", type=float, default=0.95)
    parser.add_argument("--support-multiplier", type=float, default=1.0)
    parser.add_argument("--shrinkage", type=float, default=0.2)
    parser.add_argument("--variance-floor", type=float, default=0.001)
    parser.add_argument("--maximum-multistarts", type=int, default=10)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/policy_manifold_diagnostic/fixed_state_d2"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 3:
        raise ValueError("--samples must be at least three")
    if not args.clearance_margins:
        raise ValueError("--clearance-margins must be non-empty")
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError("{} exists; pass --overwrite".format(args.out_dir))
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    config = ManifoldConfig(
        shrinkage=args.shrinkage,
        variance_floor_m=args.variance_floor,
        support_quantile=args.support_quantile,
        support_multiplier=args.support_multiplier,
        progress_tolerance_m=args.progress_tolerance,
        maximum_multistarts=args.maximum_multistarts,
    )
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt = guided_common.load_guided_policy_and_checkpoint(args.agent, device)
    envs = {}
    rows = []
    try:
        for states_path in args.states:
            with h5py.File(states_path, "r") as input_file:
                bank_name = states_path.parent.name
                for key in sorted(input_file["states"]):
                    saved = read_state(input_file["states"][key], bank_name)
                    if saved["env"] not in envs:
                        envs[saved["env"]], _ = FileUtils.env_from_checkpoint(
                            ckpt_dict=ckpt,
                            env_name=saved["env"],
                            render=False,
                            render_offscreen=False,
                            verbose=False,
                        )
                    env = envs[saved["env"]]
                    policy.start_episode()
                    env.reset_to(saved["state"])
                    raw_env = guided_common.raw_env_from_wrapper(env)
                    records = guided_common.active_object_records(raw_env)
                    centers, radii, names = guided_common.oracle_obstacle_geometry(
                        raw_env,
                        records,
                        args.target_object_name,
                    )
                    target_position = (
                        np.asarray(saved["target_position"], dtype=np.float64)
                        if saved["target_position"] is not None
                        else guided_common.target_object_position(
                            raw_env,
                            records,
                            args.target_object_name,
                        )
                    )
                    start_position = guided_common.current_eef_position(saved["obs"])
                    noise_seeds, chunks = sample_policy_chunks(
                        policy,
                        saved,
                        centers,
                        radii,
                        args.samples,
                    )
                    positions = np.asarray(
                        [
                            reconstruct_positions(start_position, chunk)
                            for chunk in chunks
                        ]
                    )
                    nominal_actions = chunks[0]
                    nominal_positions = positions[0]
                    nominal_progress = closest_target_progress(
                        start_position,
                        nominal_positions,
                        target_position,
                    )
                    nominal_clearance = minimum_segment_clearance_xy(
                        start_position,
                        nominal_positions,
                        centers,
                        radii,
                    )
                    low, high = guided_common.action_limits(
                        env,
                        nominal_actions.shape[1],
                    )

                    margin_results = {}
                    manifold_payload = None
                    for margin in args.clearance_margins:
                        raw_stats = raw_sample_statistics(
                            start_position,
                            chunks,
                            positions,
                            target_position,
                            centers,
                            radii,
                            nominal_progress,
                            args.progress_tolerance,
                            margin,
                        )
                        model, projection = solve_manifold_projection(
                            start_position=start_position,
                            nominal_actions=nominal_actions,
                            sample_positions=positions,
                            target_position=target_position,
                            obstacle_centers=centers,
                            obstacle_radii=radii,
                            action_low=low,
                            action_high=high,
                            clearance_margin_m=margin,
                            config=config,
                        )
                        if manifold_payload is None:
                            manifold_payload = model.to_dict()
                        margin_results[str(float(margin))] = {
                            "raw_samples": raw_stats,
                            "projection": projection.to_dict(),
                        }

                    row = {
                        "state_id": saved["state_id"],
                        "env": saved["env"],
                        "seed": saved["seed"],
                        "episode": saved["episode"],
                        "chunk": saved["chunk"],
                        "start_step": saved["start_step"],
                        "noise_seeds": noise_seeds,
                        "obstacle_names": names,
                        "start_position": start_position,
                        "target_position": target_position,
                        "baseline_target_progress_m": nominal_progress,
                        "baseline_min_clearance_m": nominal_clearance,
                        "manifold": manifold_payload,
                        "margins": margin_results,
                    }
                    rows.append(row)
                    primary = margin_results[str(float(args.clearance_margins[-1]))]
                    print(
                        "{} samples={} primary_raw={} primary_projection={}".format(
                            saved["state_id"],
                            args.samples,
                            primary["raw_samples"]["safe_progress_preserving_count"],
                            primary["projection"]["feasible"],
                        ),
                        flush=True,
                    )
    finally:
        envs.clear()

    with (args.out_dir / "state_metrics.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(guided_common.json_safe(row)) + "\n")
    summary = summarize(rows, args.clearance_margins)
    (args.out_dir / "summary.json").write_text(
        json.dumps(guided_common.json_safe(summary), indent=2) + "\n"
    )
    manifest = {
        "agent": str(args.agent),
        "state_banks": [str(path) for path in args.states],
        "samples_per_state": args.samples,
        "clearance_margins_m": args.clearance_margins,
        "target_object_name": args.target_object_name,
        "frozen_horizons": {"observation": 2, "prediction": 16, "action": 8},
        "manifold_config": config.__dict__,
        "scope": (
            "fixed-state policy-support upper bound using oracle XY EEF-point/"
            "object-circle geometry; projected actions are not executed"
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    write_report(args.out_dir / "report.md", summary, args)
    plot_primary_margin(
        args.out_dir / "clearance_by_state.png",
        rows,
        args.clearance_margins[-1],
    )
    print(json.dumps(guided_common.json_safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
