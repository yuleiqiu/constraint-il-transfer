"""Overlay one DDIM guidance step's waypoint push on an agentview image."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import common  # noqa: E402
import same_state_diagnostic  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


BEFORE_COLOR = "#00B7FF"
AFTER_COLOR = "#FF8C00"
VECTOR_COLOR = "#7CFC00"
START_COLOR = "#FF2DAA"
BOUNDARY_COLOR = "#FF3659"


def project(points, transform, height, width):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), axis=-1)
    pixels = np.matmul(np.asarray(transform), homogeneous[..., None])[..., 0]
    pixels = pixels[..., :2] / pixels[..., 2:3]
    pixels[..., 0] = np.clip(pixels[..., 0], 0, width - 1)
    pixels[..., 1] = np.clip(pixels[..., 1], 0, height - 1)
    return pixels


def obstacle_boundaries(centers, radii, clearance_margin, z, samples=96):
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=True)
    boundaries = []
    for center, radius in zip(centers, radii):
        effective_radius = float(radius) + float(clearance_margin)
        ring = np.zeros((samples, 3), dtype=np.float32)
        ring[:, 0] = center[0] + effective_radius * np.cos(angles)
        ring[:, 1] = center[1] + effective_radius * np.sin(angles)
        ring[:, 2] = z
        boundaries.append(ring)
    return boundaries


def select_step(diagnostics, selector):
    if not diagnostics:
        raise ValueError("Guided sample did not produce reverse-step diagnostics")
    if selector == "max-displacement":
        return int(
            np.argmax([item["max_waypoint_displacement_m"] for item in diagnostics])
        )
    index = int(selector)
    if index < 0:
        index += len(diagnostics)
    if index < 0 or index >= len(diagnostics):
        raise IndexError(
            "step index {} is outside the {} recorded DDIM steps".format(
                selector, len(diagnostics)
            )
        )
    return index


def _draw_scene(
    axis,
    image,
    start_px,
    before_px,
    after_px,
    boundary_pixels,
    active,
    show_arrows,
    before_label,
    after_label,
    vector_label,
):
    axis.imshow(image)
    for boundary_index, boundary_px in enumerate(boundary_pixels):
        axis.plot(
            boundary_px[:, 0],
            boundary_px[:, 1],
            linestyle=":",
            linewidth=2.0,
            color=BOUNDARY_COLOR,
            alpha=0.85,
            label="obstacle cost boundary" if boundary_index == 0 else None,
        )

    before_path = np.concatenate((start_px[None], before_px), axis=0)
    after_path = np.concatenate((start_px[None], after_px), axis=0)
    axis.plot(
        before_path[:, 0], before_path[:, 1],
        linestyle="-", linewidth=2.6, marker="o", markersize=4.5,
        color=BEFORE_COLOR, label=before_label, zorder=5,
    )
    axis.plot(
        after_path[:, 0], after_path[:, 1],
        linestyle="-", linewidth=2.6, marker="o", markersize=4.5,
        color=AFTER_COLOR, label=after_label, zorder=6,
    )

    pixel_displacement = after_px - before_px
    if show_arrows and np.any(active):
        axis.quiver(
            before_px[active, 0], before_px[active, 1],
            pixel_displacement[active, 0], pixel_displacement[active, 1],
            angles="xy", scale_units="xy", scale=1.0,
            width=0.006, headwidth=4.5, headlength=5.5, headaxislength=4.5,
            color=VECTOR_COLOR, label=vector_label, zorder=8,
        )
    axis.scatter(
        [start_px[0]], [start_px[1]], s=80, marker="*",
        color=START_COLOR, edgecolors="white", linewidths=0.8,
        label="current EEF", zorder=9,
    )
    for index, point in enumerate(before_px):
        axis.text(
            point[0] + 4, point[1] - 4, str(index + 1),
            color="white", fontsize=8, weight="bold", zorder=10,
        )
    axis.axis("off")


def draw_overlay(
    image,
    start,
    before,
    after,
    boundaries,
    transform,
    metadata,
    output_path,
    show_arrows,
    before_label="before guidance",
    after_label="after guidance",
    vector_label="net guidance displacement",
    title_text=None,
    zoom_description=None,
):
    height, width = image.shape[:2]
    start_px = project(np.asarray(start)[None], transform, height, width)[0]
    before_px = project(before, transform, height, width)
    after_px = project(after, transform, height, width)
    boundary_pixels = [project(item, transform, height, width) for item in boundaries]
    active = np.linalg.norm(after - before, axis=-1) > 1e-8

    trajectory_pixels = np.concatenate((start_px[None], before_px, after_px), axis=0)
    center = 0.5 * (trajectory_pixels.min(axis=0) + trajectory_pixels.max(axis=0))
    span = float(np.max(np.ptp(trajectory_pixels, axis=0)))
    crop_size = min(float(min(height, width)), max(120.0, span * 2.0 + 48.0))
    x0 = float(np.clip(center[0] - crop_size / 2.0, 0, width - crop_size))
    y0 = float(np.clip(center[1] - crop_size / 2.0, 0, height - crop_size))
    x1, y1 = x0 + crop_size, y0 + crop_size

    figure = plt.figure(figsize=(2 * width / 100.0, height / 100.0), dpi=100)
    full_axis = figure.add_axes((0.0, 0.0, 0.5, 1.0))
    zoom_axis = figure.add_axes((0.5, 0.0, 0.5, 1.0))
    _draw_scene(
        full_axis,
        image,
        start_px,
        before_px,
        after_px,
        boundary_pixels,
        active,
        show_arrows,
        before_label,
        after_label,
        vector_label,
    )
    _draw_scene(
        zoom_axis,
        image,
        start_px,
        before_px,
        after_px,
        boundary_pixels,
        active,
        show_arrows,
        before_label,
        after_label,
        vector_label,
    )

    full_axis.add_patch(
        plt.Rectangle(
            (x0, y0), crop_size, crop_size,
            fill=False, edgecolor="white", linewidth=2.0, linestyle="--", zorder=15,
        )
    )
    full_axis.set_xlim(0, width - 1)
    full_axis.set_ylim(height - 1, 0)
    zoom_axis.set_xlim(x0, x1)
    zoom_axis.set_ylim(y1, y0)

    title = title_text
    if title is None:
        title = (
            "{state_id}\nDDIM {step_index}/{step_count_minus_one}, t={timestep} | "
            "scale={scale:g} | active={active_count} | max push={max_push_cm:.2f} cm"
        ).format(**metadata)
    full_axis.text(
        0.012, 0.018, title, transform=full_axis.transAxes,
        color="white", fontsize=9, va="bottom", ha="left",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4, "edgecolor": "none"},
        zorder=20,
    )
    full_axis.legend(
        loc="upper right", fontsize=8, framealpha=0.75,
        facecolor="black", labelcolor="white",
    )
    zoom_axis.text(
        0.018, 0.982,
        "local zoom: {:.1f}x ({})".format(
            width / crop_size,
            zoom_description
            or ("vectors unscaled" if show_arrows else "trajectory geometry unchanged"),
        ),
        transform=zoom_axis.transAxes, color="white", fontsize=11,
        va="top", ha="left",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4, "edgecolor": "none"},
        zorder=20,
    )
    figure.savefig(output_path, dpi=100, facecolor="black")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=common.DEFAULT_AGENT)
    parser.add_argument(
        "--states-file",
        type=Path,
        default=Path("outputs/guided_denoising/same_state/states.hdf5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/guided_denoising/guidance_visualizations"),
    )
    parser.add_argument("--state-ids", nargs="+")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--guidance-scale", type=float, default=0.01)
    parser.add_argument("--clearance-margin", type=float, default=0.02)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--show-arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--step-index",
        default="max-displacement",
        help="zero-based diagnostics index, negative index, or max-displacement",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("render dimensions must be positive")
    if args.max_states is not None and args.max_states <= 0:
        raise ValueError("max-states must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = common.load_guided_policy_and_checkpoint(args.agent, device)
    envs = {}
    output_rows = []
    try:
        with h5py.File(args.states_file, "r") as states_file:
            available = sorted(states_file["states"])
            if args.state_ids:
                missing = sorted(set(args.state_ids) - set(available))
                if missing:
                    raise KeyError("unknown state ids: {}".format(", ".join(missing)))
                selected = args.state_ids
            else:
                selected = available
            if args.max_states is not None:
                selected = selected[: args.max_states]

            for state_id in selected:
                saved = same_state_diagnostic.read_state(states_file["states"][state_id])
                env_name = saved["env"]
                if env_name not in envs:
                    envs[env_name], _ = FileUtils.env_from_checkpoint(
                        ckpt_dict=checkpoint,
                        env_name=env_name,
                        render=False,
                        render_offscreen=True,
                        verbose=False,
                    )
                env = envs[env_name]
                policy.start_episode()
                env.reset_to(saved["state"])
                raw_env = common.raw_env_from_wrapper(env)
                records = common.active_object_records(raw_env)
                centers, radii, names = common.oracle_obstacle_geometry(
                    raw_env, records, args.target_object_name
                )
                context = common.make_guidance_context(
                    policy,
                    saved["obs"],
                    centers,
                    radii,
                    scale=args.guidance_scale,
                    clearance_margin=args.clearance_margin,
                )
                _, _, diagnostics = common.sample_action_chunk(
                    policy,
                    saved["obs"],
                    context,
                    noise_seed=saved["noise_seed"],
                )
                step_index = select_step(diagnostics, args.step_index)
                step = diagnostics[step_index]
                before = np.asarray(step["before_waypoints_m"], dtype=np.float32)[0]
                after = np.asarray(step["after_waypoints_m"], dtype=np.float32)[0]
                start = common.current_eef_position(saved["obs"])

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
                boundaries = obstacle_boundaries(
                    centers,
                    radii,
                    args.clearance_margin,
                    z=float(start[2]),
                )
                output_path = args.output_dir / "{}_step_{:02d}_t_{}.png".format(
                    state_id, step_index, step["timestep"]
                )
                if output_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        "{} already exists; pass --overwrite".format(output_path)
                    )
                metadata = {
                    "state_id": state_id,
                    "env": env_name,
                    "step_index": step_index,
                    "step_count": len(diagnostics),
                    "step_count_minus_one": len(diagnostics) - 1,
                    "timestep": step["timestep"],
                    "guidance_scale": args.guidance_scale,
                    "scale": args.guidance_scale,
                    "active_count": step["active_penetration_count"],
                    "max_push_m": step["max_waypoint_displacement_m"],
                    "max_push_cm": step["max_waypoint_displacement_m"] * 100.0,
                    "obstacle_names": names,
                    "image": output_path.name,
                }
                draw_overlay(
                    image,
                    start,
                    before,
                    after,
                    boundaries,
                    transform,
                    metadata,
                    output_path,
                    args.show_arrows,
                )
                output_rows.append(metadata)
                print("wrote {}".format(output_path), flush=True)
                restored_raw_env = common.raw_env_from_wrapper(env)
                if hasattr(restored_raw_env, "close"):
                    restored_raw_env.close()
                del envs[env_name]
    finally:
        for env in envs.values():
            raw_env = common.raw_env_from_wrapper(env)
            if hasattr(raw_env, "close"):
                raw_env.close()

    manifest = {
        "agent": str(args.agent),
        "states_file": str(args.states_file),
        "camera_name": args.camera_name,
        "image_size": [args.height, args.width],
        "guidance_scale": args.guidance_scale,
        "clearance_margin": args.clearance_margin,
        "step_selector": args.step_index,
        "show_arrows": args.show_arrows,
        "visualizations": output_rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
