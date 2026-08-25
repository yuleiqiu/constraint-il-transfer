"""Render one unguided Milk-on-table rollout with visual and collision views."""

import argparse
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.paper_lan_guidance_utils import paper_guided_policy_from_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
SPEC = importlib.util.spec_from_file_location("tabletop_paper_lan_paired_eval", PAIRED_EVAL_PATH)
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)
ENV_NAME = "PickPlaceCanTabletopMilk"


def collision_z_bounds(raw_env, obj):
    minimum = np.inf
    maximum = -np.inf
    for geom_name in obj.contact_geoms:
        geom_id = raw_env.sim.model.geom_name2id(geom_name)
        center = raw_env.sim.model.geom_aabb[geom_id, :3]
        half_size = raw_env.sim.model.geom_aabb[geom_id, 3:]
        local_corners = np.array(
            [
                center + half_size * np.asarray(signs)
                for signs in (
                    (-1, -1, -1),
                    (-1, -1, 1),
                    (-1, 1, -1),
                    (-1, 1, 1),
                    (1, -1, -1),
                    (1, -1, 1),
                    (1, 1, -1),
                    (1, 1, 1),
                )
            ]
        )
        rotation = raw_env.sim.data.geom_xmat[geom_id].reshape(3, 3)
        world = raw_env.sim.data.geom_xpos[geom_id] + local_corners @ rotation.T
        minimum = min(minimum, float(world[:, 2].min()))
        maximum = max(maximum, float(world[:, 2].max()))
    return minimum, maximum


def render_group(raw_env, camera, width, height, *, collisions, visuals):
    # Initialize the offscreen context, then explicitly select the MuJoCo geom
    # groups. Group 0 contains collision meshes; group 1 contains visual meshes.
    frame = raw_env.sim.render(width=width, height=height, camera_name=camera)[::-1]
    options = raw_env.sim._render_context_offscreen.vopt
    options.geomgroup[0] = int(collisions)
    options.geomgroup[1] = int(visuals)
    return raw_env.sim.render(width=width, height=height, camera_name=camera)[::-1]


def caption(frame, lines, *, warning=False):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    height = 10 + 15 * len(lines)
    fill = (130, 0, 0, 205) if warning else (0, 0, 0, 190)
    draw.rectangle((0, 0, image.width, height), fill=fill)
    for index, line in enumerate(lines):
        draw.text((7, 5 + 15 * index), line, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--view-size", type=int, default=480)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = paper_guided_policy_from_checkpoint(
        ckpt_path=str(args.checkpoint.resolve()), device=device
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=ENV_NAME,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    raw_env = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
    layout_seed = args.seed * 1000 + args.episode
    policy_seed = args.seed * 1_000_000 + args.episode
    random.seed(layout_seed)
    np.random.seed(layout_seed)
    env.reset()
    initial_state = env.get_state()

    rollout_args = SimpleNamespace(
        horizon=args.horizon,
        terminate_on_success=True,
        terminate_on_collision=False,
        target_object_name="Can",
        xy_only=True,
        disable_hand_collision=False,
        obstacle_cost_type="hinge",
        obstacle_cost_smoothing=0.0,
        obstacle_cost_smoothing_ratio=None,
        guidance_phase="always",
    )
    frames = []
    records = []
    closest_index = None
    closest_xy = np.inf

    def capture(frame_env, observation, step, success, collision_any):
        nonlocal closest_index, closest_xy
        del frame_env, observation
        can_pos = np.asarray(raw_env.sim.data.body_xpos[raw_env.can_body_id]).copy()
        milk_pos = np.asarray(raw_env.sim.data.body_xpos[raw_env.obstacle_body_id]).copy()
        can_bottom, can_top = collision_z_bounds(raw_env, raw_env.can)
        milk_bottom, milk_top = collision_z_bounds(raw_env, raw_env.obstacle)
        xy_distance = float(np.linalg.norm(can_pos[:2] - milk_pos[:2]))
        vertical_gap = can_bottom - milk_top
        footprint_overlap = xy_distance <= (
            float(raw_env.can.horizontal_radius) + float(raw_env.obstacle.horizontal_radius)
        )
        record = {
            "step": int(step),
            "can_pos": can_pos.tolist(),
            "milk_pos": milk_pos.tolist(),
            "can_collision_z": [can_bottom, can_top],
            "milk_collision_z": [milk_bottom, milk_top],
            "xy_center_distance_m": xy_distance,
            "footprint_overlap": bool(footprint_overlap),
            "vertical_gap_m": float(vertical_gap),
            "collision_ever": bool(collision_any),
            "success": bool(success),
        }
        records.append(record)
        if footprint_overlap and xy_distance < closest_xy:
            closest_xy = xy_distance
            closest_index = len(frames)
        if step % args.frame_skip and not success and step < args.horizon:
            return

        normal = render_group(
            raw_env,
            "agentview",
            args.view_size,
            args.view_size,
            collisions=False,
            visuals=True,
        )
        collision = render_group(
            raw_env,
            "sideview",
            args.view_size,
            args.view_size,
            collisions=True,
            visuals=False,
        )
        normal = caption(
            normal,
            [
                "NORMAL VIEW | Milk directly on tabletop | guidance OFF",
                f"seed {args.seed} ep {args.episode} | step {step} | task {'SUCCESS' if success else 'running'}",
                f"Can-Milk contact ever: {'YES' if collision_any else 'NO'}",
            ],
            warning=collision_any,
        )
        collision = caption(
            collision,
            [
                "SIDE VIEW | collision meshes only",
                f"XY centers: {100 * xy_distance:.2f} cm | footprint overlap: {'YES' if footprint_overlap else 'NO'}",
                f"Can bottom - Milk top: {100 * vertical_gap:+.2f} cm",
            ],
            warning=collision_any,
        )
        frames.append(np.concatenate((normal, collision), axis=1))

    try:
        metrics = PAIRED_EVAL.rollout_condition(
            policy,
            env,
            initial_state,
            "baseline",
            policy_seed,
            rollout_args,
            frame_callback=capture,
        )
    finally:
        raw_env.close()

    video_path = args.out_dir / f"milk_on_table_seed{args.seed}_ep{args.episode}.mp4"
    with imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    if closest_index is not None and frames:
        snapshot_index = min(closest_index, len(frames) - 1)
        imageio.imwrite(args.out_dir / "closest_footprint_overlap.png", frames[snapshot_index])
    result = {
        "environment": ENV_NAME,
        "milk_support_height_m": 0.0,
        "guidance_enabled": False,
        "seed": args.seed,
        "episode": args.episode,
        "layout_seed": layout_seed,
        "policy_seed": policy_seed,
        "metrics": metrics,
        "video": str(video_path.resolve()),
        "closest_footprint_overlap_frame": closest_index,
        "samples": records,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
