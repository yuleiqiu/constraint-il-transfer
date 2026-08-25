"""Full-rollout Tabletop counterfactual with swept-ellipsoid guidance.

This intentionally starts from the matching initial simulator state and runs
the complete episode, unlike a frozen pre-collision chunk audit. Guidance
geometry is rebuilt from the current real state at every policy chunk:
dual-view obstacle points are fused and fitted as an MVEE, while the gripper
and (once grasped) carried Can are fitted as EEF-local collision ellipsoids.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from robosuite.environments.manipulation.pick_place_can_tabletop import (
    BACKVIEW_CAMERA_POS,
    BACKVIEW_CAMERA_QUAT_WXYZ,
)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.ellipsoid_guidance_utils import (
    ellipsoid_guidance_context_from_rollout_policy,
    ellipsoid_guided_policy_from_checkpoint,
    fit_minimum_volume_enclosing_ellipsoid,
    fit_robosuite_geom_ellipsoid,
)
from robomimic.utils.target_pointcloud_utils import render_target_pointcloud

SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
CF_PATH = SCRIPT_DIR.parent / "2026-08-14_tabletop_guidance" / "counterfactual_rescue_eval.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAIRED_EVAL = load_module(PAIRED_EVAL_PATH, "mvee_ellipsoid_paired_eval")
CF = load_module(CF_PATH, "mvee_ellipsoid_counterfactual")


def inject_backview_camera(model_xml):
    """Add this branch's Backview camera to an archived state-bank XML."""

    root = ET.fromstring(model_xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("state-bank model has no worldbody")
    for camera in list(worldbody.findall("camera")):
        if camera.get("name") == "backview":
            worldbody.remove(camera)
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "backview",
            "pos": " ".join(str(value) for value in BACKVIEW_CAMERA_POS),
            "quat": " ".join(str(value) for value in BACKVIEW_CAMERA_QUAT_WXYZ),
        },
    )
    return ET.tostring(root, encoding="unicode")


def dual_view_config(height, width):
    return {
        "camera_names": ["agentview", "backview"],
        "height": int(height),
        "width": int(width),
        "num_points": 512,
        "padding_mode": "repeat",
        "max_geom_distance": None,
        "target_object": "unused",
        "include_visual_goal": False,
    }


def eef_pose(raw_env):
    eef_id = raw_env.robots[0].eef_site_id[raw_env.robots[0].arms[0]]
    position = np.asarray(raw_env.sim.data.site_xpos[eef_id]).copy()
    rotation = np.asarray(raw_env.sim.data.site_xmat[eef_id]).reshape(3, 3).copy()
    return position, rotation


def actor_ellipsoids(raw_env, eef_position, eef_rotation):
    gripper = raw_env.robots[0].gripper[raw_env.robots[0].arms[0]]
    gripper_ids = tuple(raw_env.sim.model.geom_name2id(name) for name in gripper.contact_geoms)
    gripper_local = fit_robosuite_geom_ellipsoid(
        raw_env,
        gripper_ids,
        reference_position=eef_position,
        reference_rotation=eef_rotation,
    )
    centers = [gripper_local.center]
    rotations = [gripper_local.rotation]
    axes = [gripper_local.semi_axes]
    labels = ["gripper"]

    if raw_env._check_grasp(raw_env.robots[0].gripper, raw_env.can):
        target_ids = tuple(raw_env.sim.model.geom_name2id(name) for name in raw_env.can.contact_geoms)
        target_local = fit_robosuite_geom_ellipsoid(
            raw_env,
            target_ids,
            reference_position=eef_position,
            reference_rotation=eef_rotation,
        )
        centers.append(target_local.center)
        rotations.append(target_local.rotation)
        axes.append(target_local.semi_axes)
        labels.append("target")
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(rotations, dtype=np.float64),
        np.asarray(axes, dtype=np.float64),
        tuple(labels),
    )


def make_ellipsoid_factory(height, width):
    def factory(
        *,
        policy,
        raw_env,
        current,
        obstacle_cloud,
        records,
        target_start,
        args,
        safety_distance,
        guidance_scale,
    ):
        del current, obstacle_cloud, target_start, safety_distance
        obstacle_ids = PAIRED_EVAL.obstacle_geom_ids(
            raw_env,
            records,
            args.target_object_name,
        )
        rendered = render_target_pointcloud(
            raw_env,
            config=dual_view_config(height, width),
            target_geom_ids=obstacle_ids,
            return_details=True,
        )
        if len(rendered.valid_points) == 0:
            return None
        obstacle = fit_minimum_volume_enclosing_ellipsoid(rendered.valid_points)
        eef_position, eef_rotation = eef_pose(raw_env)
        centers, rotations, axes, labels = actor_ellipsoids(
            raw_env,
            eef_position,
            eef_rotation,
        )
        return ellipsoid_guidance_context_from_rollout_policy(
            policy,
            current_eef_position=eef_position,
            current_eef_rotation=eef_rotation,
            actor_local_centers=centers,
            actor_local_rotations=rotations,
            actor_semi_axes=axes,
            obstacle_center=obstacle.center,
            obstacle_rotation=obstacle.rotation,
            obstacle_semi_axes=obstacle.semi_axes,
            guidance_scale=guidance_scale,
            clearance_margin=args.ellipsoid_margin,
            cost_type=args.ellipsoid_cost,
            cost_smoothing=args.ellipsoid_smoothing,
            actor_labels=labels,
        )

    return factory


class TrajectoryRecorder:
    def __init__(self, label):
        self.label = label
        self.rows = []

    def __call__(self, env, observation, step, success, collision):
        raw = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
        eef = np.asarray(raw.sim.data.site_xpos[raw.robots[0].eef_site_id[raw.robots[0].arms[0]]])
        can = np.asarray(raw.sim.data.body_xpos[raw.can_body_id])
        obstacle = (
            np.asarray(raw.sim.data.body_xpos[raw.obstacle_body_id])
            if raw.obstacle_body_id is not None
            else None
        )
        row = {
            "step": int(step),
            "success": bool(success),
            "collision": bool(collision),
            "grasped": bool(raw._check_grasp(raw.robots[0].gripper, raw.can)),
            "eef_xyz": eef.tolist(),
            "can_xyz": can.tolist(),
        }
        if obstacle is not None:
            row["obstacle_xyz"] = obstacle.tolist()
        self.rows.append(row)

    def result(self):
        return {
            "label": self.label,
            "steps": self.rows,
        }


def combined_frame_callback(audit, recorder):
    def callback(env, observation, step, success, collision):
        audit(env, observation, step, success, collision)
        recorder(env, observation, step, success, collision)
    return callback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=910000)
    parser.add_argument("--bank-seed", type=int, default=819)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--ellipsoid-margin", type=float, default=0.01)
    parser.add_argument("--ellipsoid-smoothing", type=float, default=0.01)
    parser.add_argument("--image-height", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.out_dir.exists():
        raise FileExistsError("output already exists: {}".format(cli.out_dir))
    cli.out_dir.mkdir(parents=True)

    bank = PAIRED_EVAL.load_state_bank(cli.state_bank)
    record = dict(
        bank["states"]["{0}:{1}:{2}".format(CF.OBSTACLE_ENV, cli.bank_seed, cli.scene)]
    )
    # The archived model XML contains the exact static obstacle layout but
    # predates this branch's Backview camera. Inject the camera while keeping
    # every layout position, then let reset_to restore the full state.
    record["state"] = dict(record["state"])
    record["state"]["model"] = inject_backview_camera(record["state"]["model"])
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = ellipsoid_guided_policy_from_checkpoint(
        ckpt_path=str(cli.checkpoint.resolve()),
        device=device,
    )
    env_kwargs = checkpoint["env_metadata"]["env_kwargs"]
    env_kwargs["camera_names"] = ["agentview", "backview"]
    env_kwargs["camera_heights"] = cli.image_height
    env_kwargs["camera_widths"] = cli.image_width
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["has_renderer"] = False
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=CF.OBSTACLE_ENV,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    args = SimpleNamespace(
        horizon=cli.horizon,
        terminate_on_success=True,
        terminate_on_collision=False,
        target_object_name="Can",
        xy_only=True,
        disable_hand_collision=False,
        obstacle_cost_type="huber_hinge",
        obstacle_cost_smoothing=cli.ellipsoid_smoothing,
        obstacle_cost_smoothing_ratio=None,
        tangent_ratio=0.0,
        guidance_phase="always",
        ellipsoid_margin=cli.ellipsoid_margin,
        ellipsoid_cost="huber_hinge",
        ellipsoid_smoothing=cli.ellipsoid_smoothing,
    )

    try:
        layout = CF.materialize_obstacle_layout(env, record)
        paired_initial_state = {
            "states": layout["states"],
            "model": record["state"]["model"],
            "ep_meta": record["state"].get("ep_meta", "{}"),
        }
        baseline_audit = CF.ContactPairAudit()
        baseline_recorder = TrajectoryRecorder("baseline")
        baseline_metrics = PAIRED_EVAL.rollout_condition(
            policy,
            env,
            paired_initial_state,
            "baseline",
            cli.policy_seed,
            args,
            frame_callback=combined_frame_callback(
                baseline_audit,
                baseline_recorder,
            ),
        )
        guided_audit = CF.ContactPairAudit()
        guided_recorder = TrajectoryRecorder("guided")
        guided_metrics = PAIRED_EVAL.rollout_condition(
            policy,
            env,
            paired_initial_state,
            "guided",
            cli.policy_seed,
            args,
            frame_callback=combined_frame_callback(
                guided_audit,
                guided_recorder,
            ),
            guidance_context_factory=make_ellipsoid_factory(
                cli.image_height,
                cli.image_width,
            ),
            guidance_scale=cli.guidance_scale,
            safety_distance=0.08,
        )
    finally:
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env).close()

    result = {
        "scene": cli.scene,
        "policy_seed": cli.policy_seed,
        "bank_seed": cli.bank_seed,
        "checkpoint": str(cli.checkpoint.resolve()),
        "guidance": {
            "scale": cli.guidance_scale,
            "margin_m": cli.ellipsoid_margin,
            "cost": args.ellipsoid_cost,
            "smoothing_m": cli.ellipsoid_smoothing,
        },
        "baseline": {
            **baseline_metrics,
            "contact_audit": baseline_audit.result(),
        },
        "guided": {
            **guided_metrics,
            "contact_audit": guided_audit.result(),
        },
        "trajectories": {
            "baseline": baseline_recorder.result(),
            "guided": guided_recorder.result(),
        },
    }
    (cli.out_dir / "result.json").write_text(
        json.dumps(PAIRED_EVAL.rollout_common.json_safe(result), indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "scene": cli.scene,
                "policy_seed": cli.policy_seed,
                "baseline_outcome": {
                    key: baseline_metrics[key]
                    for key in ("success", "collision_any", "safe_success", "collision_failure", "ncr")
                },
                "guided_outcome": {
                    key: guided_metrics[key]
                    for key in ("success", "collision_any", "safe_success", "collision_failure", "ncr")
                },
                "guided_trigger_rate": guided_metrics["guidance_trigger_rate"],
                "guided_update_norm_mean": guided_metrics["guidance_update_norm_mean"],
                "guided_minimum_clearance_m": guided_metrics.get("minimum_eef_obstacle_clearance_m"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
