"""Render a real guided rollout of a fresh Tabletop failure state as video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.ellipsoid_guidance_utils import (
    ellipsoid_guided_policy_from_checkpoint,
)

from ellipsoid_counterfactual_rollout import (
    CF,
    PAIRED_EVAL,
    inject_backview_camera,
    make_ellipsoid_factory,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--bank-seed", type=int, default=991)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, default=45.0)
    parser.add_argument("--ellipsoid-margin", type=float, default=0.05)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def annotate(frame, text):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0, 170))
    draw.text((8, 5), text, fill=(255, 255, 255))
    return np.asarray(image)


def render_frame(raw_env, obs, step, success, collision, height, width):
    agent = raw_env.sim.render(
        camera_name="agentview",
        height=height,
        width=width,
    )[::-1]
    back = raw_env.sim.render(
        camera_name="backview",
        height=height,
        width=width,
    )[::-1]
    frame = np.concatenate((agent, back), axis=1)
    can = np.asarray(raw_env.sim.data.body_xpos[raw_env.can_body_id])
    goal = np.asarray(raw_env.goal_xy)
    text = (
        "step {:03d} | grasped {} | collision {} | success {} | "
        "can-to-goal {:.2f} m".format(
            int(step),
            bool(raw_env._check_grasp(raw_env.robots[0].gripper, raw_env.can)),
            bool(collision),
            bool(success),
            float(np.linalg.norm(can[:2] - goal)),
        )
    )
    return annotate(frame, text)


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError("output already exists: {}".format(args.out))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = ellipsoid_guided_policy_from_checkpoint(
        ckpt_path=str(args.checkpoint.resolve()),
        device=device,
    )
    env_kwargs = checkpoint["env_metadata"]["env_kwargs"]
    env_kwargs["camera_names"] = ["agentview", "backview"]
    env_kwargs["camera_heights"] = args.image_height
    env_kwargs["camera_widths"] = args.image_width
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["has_renderer"] = False
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=CF.OBSTACLE_ENV,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    rollout_args = SimpleNamespace(
        horizon=600,
        terminate_on_success=True,
        terminate_on_collision=False,
        target_object_name="Can",
        xy_only=True,
        disable_hand_collision=False,
        obstacle_cost_type="huber_hinge",
        obstacle_cost_smoothing=0.01,
        obstacle_cost_smoothing_ratio=None,
        tangent_ratio=0.0,
        guidance_phase="always",
        ellipsoid_margin=args.ellipsoid_margin,
        ellipsoid_cost="huber_hinge",
        ellipsoid_smoothing=0.01,
    )

    bank = PAIRED_EVAL.load_state_bank(args.state_bank)
    record = dict(
        bank["states"][
            "{0}:{1}:{2}".format(CF.OBSTACLE_ENV, args.bank_seed, args.scene)
        ]
    )
    record["state"] = dict(record["state"])
    record["state"]["model"] = inject_backview_camera(record["state"]["model"])
    layout = CF.materialize_obstacle_layout(env, record)
    initial_state = {
        "states": layout["states"],
        "model": record["state"]["model"],
        "ep_meta": record["state"].get("ep_meta", "{}"),
    }

    rows = []
    with imageio.get_writer(
        args.out,
        fps=args.fps,
        macro_block_size=None,
    ) as writer:
        def callback(env, observation, step, success, collision):
            raw_env = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
            frame = render_frame(
                raw_env,
                observation,
                step,
                success,
                collision,
                args.image_height,
                args.image_width,
            )
            writer.append_data(frame)
            rows.append(
                {
                    "step": int(step),
                    "success": bool(success),
                    "collision": bool(collision),
                    "can_xyz": raw_env.sim.data.body_xpos[raw_env.can_body_id].tolist(),
                }
            )

        metrics = PAIRED_EVAL.rollout_condition(
            policy,
            env,
            initial_state,
            "guided",
            args.policy_seed,
            rollout_args,
            frame_callback=callback,
            guidance_context_factory=make_ellipsoid_factory(
                args.image_height,
                args.image_width,
            ),
            guidance_scale=args.guidance_scale,
            safety_distance=0.08,
        )

    manifest = {
        "video": str(args.out.resolve()),
        "scene": args.scene,
        "bank_seed": args.bank_seed,
        "policy_seed": args.policy_seed,
        "guidance_scale": args.guidance_scale,
        "margin_m": args.ellipsoid_margin,
        "success": metrics["success"],
        "collision_any": metrics["collision_any"],
        "safe_success": metrics["safe_success"],
        "ncr": metrics["ncr"],
        "first_collision_step": metrics["first_collision_step"],
        "first_guidance_active_step": metrics["first_guidance_active_step"],
        "rows": rows,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    try:
        raw = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
        raw.close()
    except Exception:
        pass
    print(json.dumps({
        "video": str(args.out.resolve()),
        "success": metrics["success"],
        "collision_any": metrics["collision_any"],
        "ncr": metrics["ncr"],
        "first_collision_step": metrics["first_collision_step"],
        "first_guidance_active_step": metrics["first_guidance_active_step"],
    }, indent=2))


if __name__ == "__main__":
    main()
