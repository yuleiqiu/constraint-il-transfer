"""In-process margin comparison sharing one baseline and policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.ellipsoid_guidance_utils import (
    ellipsoid_guided_policy_from_checkpoint,
)

from ellipsoid_counterfactual_rollout import (
    CF,
    PAIRED_EVAL,
    combined_frame_callback,
    inject_backview_camera,
    make_ellipsoid_factory,
)


class FirstGraspRecorder:
    def __init__(self):
        self.first_grasp = None
        self.grasped_steps = 0

    def __call__(self, env, observation, step, success, collision):
        raw = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
        grasped = bool(raw._check_grasp(raw.robots[0].gripper, raw.can))
        if grasped:
            self.grasped_steps += 1
            if self.first_grasp is None:
                self.first_grasp = int(step)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--bank-seed", type=int, default=991)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, default=45.0)
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.05],
    )
    parser.add_argument(
        "--guidance-phases",
        choices=("always", "target_lifted"),
        nargs="+",
        default=["always"],
    )
    parser.add_argument("--image-height", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output already exists: {}".format(args.out_dir))
    args.out_dir.mkdir(parents=True)

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
        ellipsoid_margin=0.0,
        ellipsoid_cost="huber_hinge",
        ellipsoid_smoothing=0.01,
    )

    baseline_audit = CF.ContactPairAudit()
    baseline = PAIRED_EVAL.rollout_condition(
        policy,
        env,
        initial_state,
        "baseline",
        args.policy_seed,
        rollout_args,
        frame_callback=baseline_audit,
    )

    guided = {}
    for phase in args.guidance_phases:
        rollout_args.guidance_phase = phase
        for margin in args.margins:
            rollout_args.ellipsoid_margin = float(margin)
            audit = CF.ContactPairAudit()
            grasp = FirstGraspRecorder()
            metrics = PAIRED_EVAL.rollout_condition(
                policy,
                env,
                initial_state,
                "guided",
                args.policy_seed,
                rollout_args,
                frame_callback=combined_frame_callback(
                    audit,
                    grasp,
                ),
                guidance_context_factory=make_ellipsoid_factory(
                    args.image_height,
                    args.image_width,
                ),
                guidance_scale=args.guidance_scale,
                safety_distance=0.08,
            )
            key = "{0}:{1:.2f}".format(phase, float(margin))
            guided[key] = {
                **metrics,
                "contact_audit": audit.result(),
                "first_grasp_step": grasp.first_grasp,
                "grasped_steps": grasp.grasped_steps,
            }

    result = {
        "scene": args.scene,
        "bank_seed": args.bank_seed,
        "policy_seed": args.policy_seed,
        "guidance_scale": args.guidance_scale,
        "baseline": {
            **baseline,
            "contact_audit": baseline_audit.result(),
        },
        "guided_by_margin": guided,
        "note": (
            "All conditions run in one process with one environment, one "
            "policy instance, and one shared baseline trajectory."
        ),
    }
    (args.out_dir / "sweep.json").write_text(
        json.dumps(PAIRED_EVAL.rollout_common.json_safe(result), indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "scene": args.scene,
                "baseline": {
                    key: baseline[key]
                    for key in (
                        "success",
                        "collision_any",
                        "collision_failure",
                        "ncr",
                        "first_collision_step",
                    )
                },
                "guided_by_phase_margin": {
                    key: {
                        key: value[key]
                        for key in (
                            "success",
                            "safe_success",
                            "success_with_collision",
                            "collision_any",
                            "collision_failure",
                            "ncr",
                            "first_guidance_active_step",
                            "first_collision_step",
                            "guidance_trigger_rate",
                            "guidance_update_norm_mean",
                            "minimum_eef_obstacle_clearance_m",
                            "first_grasp_step",
                        )
                    }
                    for key, value in guided.items()
                },
            },
            indent=2,
        )
    )

    try:
        raw = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
        raw.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
