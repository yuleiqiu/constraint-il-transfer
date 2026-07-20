"""Compare final unguided and guided trajectories after all DDIM steps."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import common  # noqa: E402
import same_state_diagnostic  # noqa: E402
import visualize_guidance_vectors as GuidanceViz  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402


def load_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def paired_rows(rows, baseline_scale, guided_scale):
    pairs = {}
    for row in rows:
        scale = float(row["scale"])
        condition = None
        if np.isclose(scale, baseline_scale, atol=1e-12, rtol=0.0):
            condition = "baseline"
        elif np.isclose(scale, guided_scale, atol=1e-12, rtol=0.0):
            condition = "guided"
        if condition is not None:
            pairs.setdefault(row["state_id"], {})[condition] = row

    incomplete = [
        state_id
        for state_id, pair in pairs.items()
        if set(pair) != {"baseline", "guided"}
    ]
    if incomplete:
        raise ValueError("incomplete final-trajectory pairs: {}".format(incomplete))
    for state_id, pair in pairs.items():
        if pair["baseline"]["noise_seed"] != pair["guided"]["noise_seed"]:
            raise ValueError("noise-seed mismatch for {}".format(state_id))
        if pair["baseline"]["env"] != pair["guided"]["env"]:
            raise ValueError("environment mismatch for {}".format(state_id))
    return pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=common.DEFAULT_AGENT)
    parser.add_argument(
        "--states-file",
        type=Path,
        default=Path("outputs/guided_denoising/same_state/states.hdf5"),
    )
    parser.add_argument(
        "--sweep-results",
        type=Path,
        default=Path("outputs/guided_denoising/same_state/sweep_results.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-ids", nargs="+")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--baseline-scale", type=float, default=0.0)
    parser.add_argument("--guided-scale", type=float, default=0.01)
    parser.add_argument("--clearance-margin", type=float, default=0.02)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--show-arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("render dimensions must be positive")
    if args.max_states is not None and args.max_states <= 0:
        raise ValueError("max-states must be positive")
    if np.isclose(args.baseline_scale, args.guided_scale):
        raise ValueError("baseline and guided scales must differ")

    pairs = paired_rows(
        load_rows(args.sweep_results),
        args.baseline_scale,
        args.guided_scale,
    )
    checkpoint = FileUtils.load_dict_from_checkpoint(str(args.agent))
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    ObsUtils.initialize_obs_utils_with_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []

    with h5py.File(args.states_file, "r") as states_file:
        available = sorted(set(states_file["states"]) & set(pairs))
        if args.state_ids:
            missing = sorted(set(args.state_ids) - set(available))
            if missing:
                raise KeyError("unknown or unpaired state ids: {}".format(missing))
            selected = args.state_ids
        else:
            selected = available
        if args.max_states is not None:
            selected = selected[: args.max_states]

        for state_id in selected:
            state_group = states_file["states"][state_id]
            saved = same_state_diagnostic.read_state(state_group)
            baseline = pairs[state_id]["baseline"]
            guided = pairs[state_id]["guided"]
            before = np.asarray(
                baseline["predicted_eef_positions"], dtype=np.float32
            )
            after = np.asarray(guided["predicted_eef_positions"], dtype=np.float32)
            if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 3:
                raise ValueError("invalid paired trajectory shape for {}".format(state_id))

            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=checkpoint,
                env_name=saved["env"],
                render=False,
                render_offscreen=True,
                verbose=False,
            )
            try:
                env.reset_to(saved["state"])
                image = env.render(
                    mode="rgb_array",
                    height=args.height,
                    width=args.width,
                    camera_name=args.camera_name,
                )
                transform = env.get_camera_transform_matrix(
                    camera_name=args.camera_name,
                    camera_height=args.height,
                    camera_width=args.width,
                )
                start = common.current_eef_position(saved["obs"])
                centers = state_group["obstacle_centers"][()]
                radii = state_group["obstacle_radii"][()]
                boundaries = GuidanceViz.obstacle_boundaries(
                    centers,
                    radii,
                    args.clearance_margin,
                    z=float(start[2]),
                )
                displacement = after - before
                displacement_norms = np.linalg.norm(displacement, axis=-1)
                output_path = args.output_dir / "{}_final.png".format(state_id)
                if output_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        "{} already exists; pass --overwrite".format(output_path)
                    )
                title = (
                    "{}\nFinal trajectories after 10 DDIM steps | same noise={} | "
                    "max waypoint delta={:.2f} cm"
                ).format(
                    state_id,
                    baseline["noise_seed"],
                    100.0 * float(np.max(displacement_norms)),
                )
                metadata = {
                    "state_id": state_id,
                    "env": saved["env"],
                    "noise_seed": baseline["noise_seed"],
                    "baseline_scale": args.baseline_scale,
                    "guided_scale": args.guided_scale,
                    "waypoints": int(before.shape[0]),
                    "mean_waypoint_delta_m": float(np.mean(displacement_norms)),
                    "max_waypoint_delta_m": float(np.max(displacement_norms)),
                    "baseline_min_clearance_m": baseline["predicted_min_clearance_m"],
                    "guided_min_clearance_m": guided["predicted_min_clearance_m"],
                    "image": output_path.name,
                }
                GuidanceViz.draw_overlay(
                    image,
                    start,
                    before,
                    after,
                    boundaries,
                    transform,
                    metadata,
                    output_path,
                    args.show_arrows,
                    before_label="final no guidance (scale={:g})".format(
                        args.baseline_scale
                    ),
                    after_label="final with guidance (scale={:g})".format(
                        args.guided_scale
                    ),
                    vector_label="final net guidance effect",
                    title_text=title,
                    zoom_description=(
                        "final vectors unscaled"
                        if args.show_arrows
                        else "final trajectory geometry unchanged"
                    ),
                )
                output_rows.append(metadata)
                print("wrote {}".format(output_path), flush=True)
            finally:
                raw_env = common.raw_env_from_wrapper(env)
                if hasattr(raw_env, "close"):
                    raw_env.close()

    manifest = {
        "agent": str(args.agent),
        "states_file": str(args.states_file),
        "sweep_results": str(args.sweep_results),
        "comparison": "final trajectories after all 10 DDIM steps",
        "pairing": "same saved state, observation, and initial diffusion noise",
        "baseline_scale": args.baseline_scale,
        "guided_scale": args.guided_scale,
        "show_arrows": args.show_arrows,
        "camera_name": args.camera_name,
        "source_image_size": [args.height, args.width],
        "visualizations": output_rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
