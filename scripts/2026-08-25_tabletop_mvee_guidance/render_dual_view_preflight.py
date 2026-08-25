"""Render the Tabletop agent/back views and their fused obstacle point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite
from matplotlib.patches import Patch
from robosuite.utils import camera_utils as CameraUtils

from robomimic.utils.target_pointcloud_utils import render_target_pointcloud
from robosuite.environments.manipulation.pick_place_can_tabletop import (
    BACKVIEW_CAMERA_LOOK_AT,
    BACKVIEW_CAMERA_POS,
    BACKVIEW_CAMERA_QUAT_WXYZ,
    BACKVIEW_REFERENCE_QUAT_WXYZ,
)


CAMERA_NAMES = ("agentview", "backview")
VIEW_COLORS = ("#0072B2", "#D55E00")
SEMANTIC_COLORS = {
    "Can": np.asarray([0.0, 170.0, 255.0]),
    "GoalTray": np.asarray([230.0, 90.0, 200.0]),
    "Obstacle": np.asarray([255.0, 45.0, 45.0]),
}


def geom_ids_with_prefix(env, prefix):
    ids = []
    for geom_id in range(env.sim.model.ngeom):
        name = env.sim.model.geom_id2name(geom_id)
        if name is not None and name.startswith(prefix):
            ids.append(geom_id)
    if not ids:
        raise RuntimeError("No geoms found with prefix {!r}".format(prefix))
    return tuple(ids)


def obstacle_geom_ids(env):
    ids = []
    for obj in (env.obstacle, env.obstacle_support):
        if obj is None:
            continue
        for attribute in ("visual_geoms", "contact_geoms"):
            for geom_name in getattr(obj, attribute, ()):
                try:
                    ids.append(env.sim.model.geom_name2id(geom_name))
                except Exception:
                    pass
    result = tuple(sorted(set(ids)))
    if not result:
        raise RuntimeError("RaisedMilk environment exposes no obstacle geoms")
    return result


def pointcloud_config(camera_names, *, height, width, num_points):
    return {
        "camera_names": list(camera_names),
        "height": height,
        "width": width,
        "num_points": num_points,
        "padding_mode": "repeat",
        "max_geom_distance": None,
        "target_object": "unused",
        "include_visual_goal": False,
    }


def render_rgb_and_masks(env, camera_name, semantic_geom_ids, *, height, width):
    rgb = env.sim.render(
        camera_name=camera_name,
        height=height,
        width=width,
    )[::-1]
    segmentation = env.sim.render(
        camera_name=camera_name,
        height=height,
        width=width,
        segmentation=True,
    )[::-1]
    overlay = rgb.astype(np.float32)
    masks = {}
    for label, geom_ids in semantic_geom_ids.items():
        mask = np.isin(segmentation[..., 1], np.asarray(geom_ids))
        masks[label] = mask
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * SEMANTIC_COLORS[label]
    return np.clip(overlay, 0, 255).astype(np.uint8), masks


def task_corridor_points(env):
    can_xy = np.asarray(env.sim.data.body_xpos[env.can_body_id, :2])
    fractions = np.linspace(0.0, 1.0, 9)
    xy = can_xy[None] + fractions[:, None] * (env.goal_xy - can_xy)[None]
    return np.concatenate(
        [
            np.column_stack([xy, np.full(len(xy), height)])
            for height in (0.82, 0.95, 1.10)
        ],
        axis=0,
    )


def project_world_points(env, camera_name, points, *, height, width):
    transform = CameraUtils.get_camera_transform_matrix(
        env.sim,
        camera_name,
        camera_height=height,
        camera_width=width,
    )
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    projected = homogeneous @ transform.T
    pixels_xy = projected[:, :2] / projected[:, 2:3]
    inside = (
        (projected[:, 2] > 0.0)
        & (pixels_xy[:, 0] >= 0.0)
        & (pixels_xy[:, 0] < width)
        & (pixels_xy[:, 1] >= 0.0)
        & (pixels_xy[:, 1] < height)
    )
    margins = np.minimum.reduce(
        [
            pixels_xy[:, 0],
            width - 1 - pixels_xy[:, 0],
            pixels_xy[:, 1],
            height - 1 - pixels_xy[:, 1],
        ]
    )
    return pixels_xy, inside, margins


def voxel_count(points, resolution=0.005):
    voxels = np.floor(np.asarray(points) / float(resolution)).astype(np.int64)
    return int(len(np.unique(voxels, axis=0)))


def equalize_3d_axes(axis, points):
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 0.04)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/tabletop_mvee_guidance/dual_view_preflight_seed825.png"
        ),
    )
    parser.add_argument("--seed", type=int, default=825)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.output.with_suffix(".json")
    for path in (args.output, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = suite.make(
        "PickPlaceCanTabletopRaisedMilk",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        initialization_noise=None,
        camera_names=list(CAMERA_NAMES),
        camera_heights=args.height,
        camera_widths=args.width,
        seed=args.seed,
    )
    try:
        env.reset()
        obstacle_ids = obstacle_geom_ids(env)
        semantic_geom_ids = {
            "Can": geom_ids_with_prefix(env, "Can_"),
            "GoalTray": geom_ids_with_prefix(env, "GoalTray_"),
            "Obstacle": obstacle_ids,
        }
        corridor_points = task_corridor_points(env)
        views = {}
        overlays = {}
        masks = {}
        corridor_projections = {}
        for camera_name in CAMERA_NAMES:
            overlays[camera_name], masks[camera_name] = render_rgb_and_masks(
                env,
                camera_name,
                semantic_geom_ids,
                height=args.height,
                width=args.width,
            )
            corridor_projections[camera_name] = project_world_points(
                env,
                camera_name,
                corridor_points,
                height=args.height,
                width=args.width,
            )
            views[camera_name] = render_target_pointcloud(
                env,
                config=pointcloud_config(
                    [camera_name],
                    height=args.height,
                    width=args.width,
                    num_points=args.num_points,
                ),
                target_geom_ids=obstacle_ids,
                return_details=True,
            )
        fused = render_target_pointcloud(
            env,
            config=pointcloud_config(
                CAMERA_NAMES,
                height=args.height,
                width=args.width,
                num_points=args.num_points,
            ),
            target_geom_ids=obstacle_ids,
            return_details=True,
        )

        figure = plt.figure(figsize=(13, 10), constrained_layout=True)
        grid = figure.add_gridspec(2, 2)
        for index, camera_name in enumerate(CAMERA_NAMES):
            axis = figure.add_subplot(grid[0, index])
            axis.imshow(overlays[camera_name])
            pixels_xy, inside, _ = corridor_projections[camera_name]
            visible_midheight = inside[9:18]
            axis.plot(
                pixels_xy[9:18, 0][visible_midheight],
                pixels_xy[9:18, 1][visible_midheight],
                "o-",
                color="#F0E442",
                markeredgecolor="#111111",
                markeredgewidth=0.5,
                linewidth=1.5,
                markersize=4,
            )
            axis.set_title(
                "{} — Can {:,} px | goal {:,} px | obstacle {:,} px\n"
                "corridor {}/{} in frame; obstacle points {:,}".format(
                    camera_name,
                    int(masks[camera_name]["Can"].sum()),
                    int(masks[camera_name]["GoalTray"].sum()),
                    int(masks[camera_name]["Obstacle"].sum()),
                    int(inside.sum()),
                    len(inside),
                    len(views[camera_name].valid_points),
                )
            )
            axis.legend(
                handles=[
                    Patch(color=SEMANTIC_COLORS[label] / 255.0, label=label)
                    for label in SEMANTIC_COLORS
                ]
                + [
                    plt.Line2D(
                        [],
                        [],
                        color="#F0E442",
                        marker="o",
                        label="transport centerline (z=0.95 m)",
                    )
                ],
                loc="lower right",
                fontsize=7,
            )
            axis.axis("off")

        axis_3d = figure.add_subplot(grid[1, 0], projection="3d")
        for camera_name, color in zip(CAMERA_NAMES, VIEW_COLORS):
            points = views[camera_name].valid_points
            stride = max(1, len(points) // 2500)
            axis_3d.scatter(
                points[::stride, 0],
                points[::stride, 1],
                points[::stride, 2],
                s=2,
                alpha=0.35,
                color=color,
                label="{} valid".format(camera_name),
            )
        axis_3d.scatter(
            fused.points[:, 0],
            fused.points[:, 1],
            fused.points[:, 2],
            s=7,
            alpha=0.8,
            color="#111111",
            label="fused FPS{}".format(args.num_points),
        )
        equalize_3d_axes(axis_3d, fused.valid_points)
        axis_3d.set_xlabel("world X [m]")
        axis_3d.set_ylabel("world Y [m]")
        axis_3d.set_zlabel("world Z [m]")
        axis_3d.set_title("RaisedMilk + support: fused world-frame cloud")
        axis_3d.legend(loc="upper left", fontsize=8)

        axis_xy = figure.add_subplot(grid[1, 1])
        for camera_name, color in zip(CAMERA_NAMES, VIEW_COLORS):
            points = views[camera_name].valid_points
            stride = max(1, len(points) // 3000)
            axis_xy.scatter(
                points[::stride, 0],
                points[::stride, 1],
                s=3,
                alpha=0.3,
                color=color,
                label=camera_name,
            )
        axis_xy.scatter(
            fused.points[:, 0],
            fused.points[:, 1],
            s=8,
            alpha=0.75,
            color="#111111",
            label="fused FPS",
        )
        can_xy = np.asarray(env.sim.data.body_xpos[env.can_body_id, :2])
        axis_xy.scatter(
            can_xy[0],
            can_xy[1],
            marker="o",
            s=80,
            color="#009E73",
            label="Can start",
        )
        axis_xy.scatter(
            env.goal_xy[0],
            env.goal_xy[1],
            marker="*",
            s=130,
            color="#CC79A7",
            label="Goal",
        )
        axis_xy.annotate(
            "",
            xy=env.goal_xy,
            xytext=can_xy,
            arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.5},
        )
        axis_xy.set_aspect("equal", adjustable="box")
        axis_xy.set_xlim(-0.3, 0.3)
        axis_xy.set_ylim(-0.3, 0.3)
        axis_xy.set_xlabel("world X [m]")
        axis_xy.set_ylabel("world Y [m]")
        axis_xy.set_title("Top-down coverage and transport direction")
        axis_xy.legend(loc="upper right", fontsize=8)
        axis_xy.grid(alpha=0.2)

        figure.suptitle(
            "Tabletop dual-view preflight — reference Backview position, task-retargeted optical axis",
            fontsize=15,
        )
        figure.savefig(args.output, dpi=170)
        plt.close(figure)

        manifest = {
            "environment": "PickPlaceCanTabletopRaisedMilk",
            "seed": args.seed,
            "reference_repository_commit": "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
            "reference_path": (
                "safelibero/libero/libero/assets/scenes/"
                "libero_tabletop_base_style.xml"
            ),
            "reference_backview": {
                "position": list(BACKVIEW_CAMERA_POS),
                "quaternion_wxyz": list(BACKVIEW_REFERENCE_QUAT_WXYZ),
                "note": "Original horizontal optical axis excludes GoalTray in this task",
            },
            "deployed_backview": {
                "position": list(BACKVIEW_CAMERA_POS),
                "look_at_world_xyz": list(BACKVIEW_CAMERA_LOOK_AT),
                "quaternion_wxyz": list(BACKVIEW_CAMERA_QUAT_WXYZ),
            },
            "image_size": [args.height, args.width],
            "obstacle_geom_ids": list(obstacle_ids),
            "views": {
                camera_name: {
                    "can_mask_pixels": int(masks[camera_name]["Can"].sum()),
                    "goal_mask_pixels": int(masks[camera_name]["GoalTray"].sum()),
                    "obstacle_mask_pixels": int(
                        masks[camera_name]["Obstacle"].sum()
                    ),
                    "corridor_points_in_frame": int(
                        corridor_projections[camera_name][1].sum()
                    ),
                    "corridor_points_total": int(len(corridor_points)),
                    "minimum_corridor_margin_pixels": float(
                        corridor_projections[camera_name][2][
                            corridor_projections[camera_name][1]
                        ].min()
                    ),
                    "valid_points": int(len(views[camera_name].valid_points)),
                    "occupied_5mm_voxels": voxel_count(
                        views[camera_name].valid_points
                    ),
                }
                for camera_name in CAMERA_NAMES
            },
            "fused": {
                "valid_points_before_fps": int(len(fused.valid_points)),
                "sampled_points": int(len(fused.points)),
                "occupied_5mm_voxels_before_fps": voxel_count(fused.valid_points),
                "minimum_xyz": fused.valid_points.min(axis=0).tolist(),
                "maximum_xyz": fused.valid_points.max(axis=0).tolist(),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        print("wrote {}".format(args.output))
    finally:
        env.close()


if __name__ == "__main__":
    main()
