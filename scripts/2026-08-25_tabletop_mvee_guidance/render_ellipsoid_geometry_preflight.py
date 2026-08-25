"""Fit and visualize Tabletop obstacle, gripper, and target ellipsoids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite

from robomimic.utils.ellipsoid_guidance_utils import (
    fit_minimum_volume_enclosing_ellipsoid,
    fit_robosuite_geom_ellipsoid,
    robosuite_geom_surface_points,
)
from robomimic.utils.target_pointcloud_utils import render_target_pointcloud

from render_dual_view_preflight import (
    CAMERA_NAMES,
    VIEW_COLORS,
    obstacle_geom_ids,
    pointcloud_config,
)


ELLIPSOID_COLORS = {
    "obstacle": "#D55E00",
    "gripper": "#0072B2",
    "target": "#009E73",
}


def ellipsoid_surface(center, rotation, semi_axes, *, longitude=48, latitude=24):
    u = np.linspace(0.0, 2.0 * np.pi, longitude)
    v = np.linspace(0.0, np.pi, latitude)
    unit = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=-1,
    )
    return center + (unit * semi_axes) @ rotation.T


def set_equal_3d(axis, points, minimum_radius=0.1):
    points = np.asarray(points)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, minimum_radius)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def geom_ids(env, names):
    return tuple(env.sim.model.geom_name2id(name) for name in names)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/tabletop_mvee_guidance/"
            "ellipsoid_geometry_preflight_seed825.png"
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
        seed=args.seed,
    )
    try:
        env.reset()
        obstacle_ids = obstacle_geom_ids(env)
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
        obstacle = fit_minimum_volume_enclosing_ellipsoid(fused.valid_points)

        eef_site_id = env.robots[0].eef_site_id[env.robots[0].arms[0]]
        eef_position = np.asarray(env.sim.data.site_xpos[eef_site_id]).copy()
        eef_rotation = np.asarray(env.sim.data.site_xmat[eef_site_id]).reshape(3, 3).copy()
        gripper_names = tuple(env.robots[0].gripper[env.robots[0].arms[0]].contact_geoms)
        gripper_ids = geom_ids(env, gripper_names)
        gripper_local = fit_robosuite_geom_ellipsoid(
            env,
            gripper_ids,
            reference_position=eef_position,
            reference_rotation=eef_rotation,
        )
        gripper = {
            "center": eef_position + eef_rotation @ gripper_local.center,
            "rotation": eef_rotation @ gripper_local.rotation,
            "semi_axes": gripper_local.semi_axes,
        }
        gripper_points = robosuite_geom_surface_points(env, gripper_ids)

        target_names = tuple(env.can.contact_geoms)
        target_ids = geom_ids(env, target_names)
        target = fit_robosuite_geom_ellipsoid(env, target_ids)
        target_local = fit_robosuite_geom_ellipsoid(
            env,
            target_ids,
            reference_position=eef_position,
            reference_rotation=eef_rotation,
        )
        target_points = robosuite_geom_surface_points(env, target_ids)

        surfaces = {
            "obstacle": ellipsoid_surface(
                obstacle.center,
                obstacle.rotation,
                obstacle.semi_axes,
            ),
            "gripper": ellipsoid_surface(
                gripper["center"],
                gripper["rotation"],
                gripper["semi_axes"],
            ),
            "target": ellipsoid_surface(
                target.center,
                target.rotation,
                target.semi_axes,
            ),
        }

        figure = plt.figure(figsize=(14, 11), constrained_layout=True)
        grid = figure.add_gridspec(2, 2)
        full_axis = figure.add_subplot(grid[0, 0], projection="3d")
        point_sets = {
            "obstacle": fused.valid_points,
            "gripper": gripper_points,
            "target": target_points,
        }
        for label, points in point_sets.items():
            stride = max(1, len(points) // 1800)
            full_axis.scatter(
                points[::stride, 0],
                points[::stride, 1],
                points[::stride, 2],
                s=2,
                alpha=0.25,
                color=ELLIPSOID_COLORS[label],
                label="{} geometry".format(label),
            )
            surface = surfaces[label]
            full_axis.plot_surface(
                surface[..., 0],
                surface[..., 1],
                surface[..., 2],
                color=ELLIPSOID_COLORS[label],
                alpha=0.18,
                linewidth=0,
            )
        all_points = np.concatenate(list(point_sets.values()), axis=0)
        set_equal_3d(full_axis, all_points, minimum_radius=0.18)
        full_axis.set_xlabel("world X [m]")
        full_axis.set_ylabel("world Y [m]")
        full_axis.set_zlabel("world Z [m]")
        full_axis.set_title("Task geometry: conservative rigid actor ellipsoids")
        full_axis.legend(loc="upper left", fontsize=8)

        obstacle_axis = figure.add_subplot(grid[0, 1], projection="3d")
        start = 0
        for camera_name, count, color in zip(
            fused.camera_names,
            fused.valid_point_counts,
            VIEW_COLORS,
        ):
            points = fused.valid_points[start : start + count]
            start += count
            stride = max(1, len(points) // 1800)
            obstacle_axis.scatter(
                points[::stride, 0],
                points[::stride, 1],
                points[::stride, 2],
                s=3,
                alpha=0.35,
                color=color,
                label="{} ({:,})".format(camera_name, count),
            )
        surface = surfaces["obstacle"]
        obstacle_axis.plot_surface(
            surface[..., 0],
            surface[..., 1],
            surface[..., 2],
            color=ELLIPSOID_COLORS["obstacle"],
            alpha=0.25,
            linewidth=0,
        )
        set_equal_3d(obstacle_axis, fused.valid_points, minimum_radius=0.08)
        obstacle_axis.set_xlabel("world X [m]")
        obstacle_axis.set_ylabel("world Y [m]")
        obstacle_axis.set_zlabel("world Z [m]")
        obstacle_axis.set_title(
            "Obstacle MVEE: all {:,} fused points enclosed".format(
                len(fused.valid_points)
            )
        )
        obstacle_axis.legend(loc="upper left", fontsize=8)

        top_axis = figure.add_subplot(grid[1, 0])
        for label, points in point_sets.items():
            stride = max(1, len(points) // 2000)
            top_axis.scatter(
                points[::stride, 0],
                points[::stride, 1],
                s=3,
                alpha=0.2,
                color=ELLIPSOID_COLORS[label],
            )
            surface = surfaces[label]
            top_axis.scatter(
                surface[..., 0],
                surface[..., 1],
                s=1,
                alpha=0.18,
                color=ELLIPSOID_COLORS[label],
                label="{} ellipsoid".format(label),
            )
        can_xy = np.asarray(env.sim.data.body_xpos[env.can_body_id, :2])
        top_axis.annotate(
            "",
            xy=env.goal_xy,
            xytext=can_xy,
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 2},
        )
        top_axis.scatter(
            env.goal_xy[0],
            env.goal_xy[1],
            marker="*",
            s=130,
            color="#CC79A7",
            label="goal",
        )
        top_axis.set_aspect("equal", adjustable="box")
        top_axis.set_xlim(-0.25, 0.25)
        top_axis.set_ylim(-0.30, 0.30)
        top_axis.set_xlabel("world X [m]")
        top_axis.set_ylabel("world Y [m]")
        top_axis.set_title("Top-down volumes and start-to-goal direction")
        top_axis.grid(alpha=0.2)
        top_axis.legend(loc="upper left", fontsize=8)

        axes_axis = figure.add_subplot(grid[1, 1])
        labels = ("obstacle MVEE", "gripper envelope", "Can target")
        all_axes = np.stack(
            [obstacle.semi_axes, gripper["semi_axes"], target.semi_axes]
        )
        x = np.arange(3)
        width = 0.24
        for index in range(3):
            axes_axis.bar(
                x + (index - 1) * width,
                100.0 * all_axes[:, index],
                width,
                label="semi-axis {}".format(index + 1),
            )
        axes_axis.set_xticks(x, labels, rotation=10)
        axes_axis.set_ylabel("semi-axis [cm]")
        axes_axis.set_title("Fitted volume scales (descending axis order)")
        axes_axis.grid(axis="y", alpha=0.2)
        axes_axis.legend(fontsize=8)

        figure.suptitle(
            "Tabletop MVEE geometry preflight — fused obstacle + gripper / target actors",
            fontsize=15,
        )
        figure.savefig(args.output, dpi=170)
        plt.close(figure)

        manifest = {
            "environment": "PickPlaceCanTabletopRaisedMilk",
            "seed": args.seed,
            "obstacle": {
                "source": "agentview + backview oracle geom point cloud",
                "camera_names": list(fused.camera_names),
                "valid_point_counts": list(fused.valid_point_counts),
                "total_valid_points": int(len(fused.valid_points)),
                "mvee": obstacle.to_dict(),
            },
            "gripper": {
                "source": "RethinkGripper collision geoms",
                "geom_names": list(gripper_names),
                "local_eef_mvee": gripper_local.to_dict(),
                "world_center": gripper["center"].tolist(),
                "world_rotation": gripper["rotation"].tolist(),
                "approximation": "mesh/cylinder samples; other geoms use conservative AABB corners",
            },
            "target": {
                "source": "Can collision mesh vertices",
                "geom_names": list(target_names),
                "world_mvee": target.to_dict(),
                "local_eef_mvee": target_local.to_dict(),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        print("wrote {}".format(args.output))
    finally:
        env.close()


if __name__ == "__main__":
    main()
