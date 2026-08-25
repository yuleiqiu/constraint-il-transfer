"""Three-way clean / obstacle / guided Tabletop counterfactual evaluation."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.paper_lan_guidance_utils import paper_guided_policy_from_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
SPEC = importlib.util.spec_from_file_location("tabletop_counterfactual_paired_eval", PAIRED_EVAL_PATH)
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)

CLEAN_ENV = "PickPlaceCanTabletop"
OBSTACLE_ENV = "PickPlaceCanTabletopRaisedMilk"


def body_pose(raw_env, body_id):
    return {
        "pos": np.asarray(raw_env.sim.model.body_pos[body_id], dtype=np.float64).copy(),
        "quat": np.asarray(raw_env.sim.model.body_quat[body_id], dtype=np.float64).copy(),
    }


def set_body_pose(raw_env, body_id, pose):
    raw_env.sim.model.body_pos[body_id] = pose["pos"]
    raw_env.sim.model.body_quat[body_id] = pose["quat"]


def materialize_obstacle_layout(env, record):
    """Load a saved XML once, then return a state-only, internally consistent layout."""

    env.reset_to(record["state"])
    raw = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
    raw.goal_xy = np.asarray(raw.sim.model.body_pos[raw.goal_tray_body_id, :2]).copy()
    raw.obstacle_xy = np.asarray(raw.sim.model.body_pos[raw.obstacle_body_id, :2]).copy()
    raw.sim.forward()
    return {
        "states": np.asarray(raw.sim.get_state().flatten(), dtype=np.float64).copy(),
        "goal": body_pose(raw, raw.goal_tray_body_id),
        "milk": body_pose(raw, raw.obstacle_body_id),
        "support": body_pose(raw, raw.obstacle_support_body_id),
        "goal_xy": raw.goal_xy.copy(),
        "obstacle_xy": raw.obstacle_xy.copy(),
    }


def apply_layout(env, layout, *, obstacle):
    raw = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
    set_body_pose(raw, raw.goal_tray_body_id, layout["goal"])
    raw.goal_xy = layout["goal_xy"].copy()
    if obstacle:
        set_body_pose(raw, raw.obstacle_body_id, layout["milk"])
        set_body_pose(raw, raw.obstacle_support_body_id, layout["support"])
        raw.obstacle_xy = layout["obstacle_xy"].copy()
    raw.sim.forward()
    observation = env.reset_to({"states": layout["states"]})
    # reset_to only mutates dynamic state. Reassert evaluation-only metadata.
    raw.goal_xy = layout["goal_xy"].copy()
    if obstacle:
        raw.obstacle_xy = layout["obstacle_xy"].copy()
    raw.sim.forward()
    return observation


def initial_match(clean_env, obstacle_env, layout):
    clean_obs = apply_layout(clean_env, layout, obstacle=False)
    obstacle_obs = apply_layout(obstacle_env, layout, obstacle=True)
    clean = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(clean_env)
    obstacle = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(obstacle_env)
    fields = {
        "dynamic_state_max_abs": float(
            np.max(np.abs(clean.sim.get_state().flatten() - obstacle.sim.get_state().flatten()))
        ),
        "can_position_max_abs_m": float(
            np.max(
                np.abs(
                    clean.sim.data.body_xpos[clean.can_body_id]
                    - obstacle.sim.data.body_xpos[obstacle.can_body_id]
                )
            )
        ),
        "goal_position_max_abs_m": float(
            np.max(
                np.abs(
                    clean.sim.data.body_xpos[clean.goal_tray_body_id]
                    - obstacle.sim.data.body_xpos[obstacle.goal_tray_body_id]
                )
            )
        ),
        "eef_position_max_abs_m": float(
            np.max(
                np.abs(
                    PAIRED_EVAL.rollout_common.current_raw_eef_position(clean)
                    - PAIRED_EVAL.rollout_common.current_raw_eef_position(obstacle)
                )
            )
        ),
        "agent_pos_max_abs": float(
            np.max(np.abs(np.asarray(clean_obs["agent_pos"]) - np.asarray(obstacle_obs["agent_pos"])))
        ),
        "task_pointcloud_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(clean_obs["task_pointcloud"])
                    - np.asarray(obstacle_obs["task_pointcloud"])
                )
            )
        ),
    }
    for key in ("dynamic_state_max_abs", "can_position_max_abs_m", "goal_position_max_abs_m", "eef_position_max_abs_m"):
        if fields[key] > 1e-10:
            raise RuntimeError(f"clean/obstacle initial-state mismatch: {key}={fields[key]}")
    return fields


class ContactPairAudit:
    """Collect exact physical contact pairs plus an explicitly labelled EEF XY proxy."""

    def __init__(self):
        self.pairs = {}
        self.eef_milk_xy_footprint_overlap = False
        self.minimum_eef_milk_xy_clearance = float("inf")

    def __call__(self, env, observation, step, success, collision):
        raw = PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env)
        records = PAIRED_EVAL.rollout_common.active_object_records(raw)
        maps = PAIRED_EVAL.rollout_common.contact_maps(raw, records, "Can")
        _, target_ids, obstacle_by_geom = maps
        arm, gripper, hand, fingers = PAIRED_EVAL.categorized_robot_geom_ids(raw)
        model = raw.sim.model
        eef = PAIRED_EVAL.rollout_common.current_raw_eef_position(raw)
        if getattr(raw, "obstacle_body_id", None) is not None:
            center = np.asarray(raw.sim.data.body_xpos[raw.obstacle_body_id])
            clearance = float(np.linalg.norm(eef[:2] - center[:2]) - raw.obstacle.horizontal_radius)
            self.minimum_eef_milk_xy_clearance = min(self.minimum_eef_milk_xy_clearance, clearance)
            self.eef_milk_xy_footprint_overlap |= clearance <= 0.0

        for index in range(raw.sim.data.ncon):
            contact = raw.sim.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            for actor, other in ((geom1, geom2), (geom2, geom1)):
                obstacle_name = obstacle_by_geom.get(other)
                if obstacle_name is None:
                    continue
                categories = []
                if actor in target_ids:
                    categories.append("carried_can")
                if actor in arm:
                    categories.append("arm_link")
                if actor in gripper:
                    categories.append("gripper")
                if actor in hand:
                    categories.append("gripper_hand")
                if actor in fingers:
                    categories.append("gripper_finger")
                for category in categories:
                    key = (category, obstacle_name, model.geom_id2name(actor), model.geom_id2name(other))
                    item = self.pairs.setdefault(
                        key,
                        {"category": category, "obstacle": obstacle_name,
                         "actor_geom": key[2], "obstacle_geom": key[3],
                         "first_step": int(step), "contact_steps": 0},
                    )
                    item["contact_steps"] += 1
                break

    def result(self):
        return {
            "physical_contact_pairs": sorted(
                self.pairs.values(), key=lambda item: (
                    item["first_step"], item["category"], item["actor_geom"], item["obstacle_geom"]
                )
            ),
            "physical_eef_site_contact_defined": False,
            "eef_milk_xy_footprint_overlap_proxy": bool(self.eef_milk_xy_footprint_overlap),
            "minimum_eef_milk_xy_clearance_proxy_m": (
                self.minimum_eef_milk_xy_clearance
                if np.isfinite(self.minimum_eef_milk_xy_clearance)
                else None
            ),
        }


def rollout(policy, env, layout, policy_seed, args, *, obstacle, guided):
    apply_layout(env, layout, obstacle=obstacle)
    audit = ContactPairAudit()
    metrics = PAIRED_EVAL.rollout_condition(
        policy,
        env,
        {"states": layout["states"]},
        "guided" if guided else "baseline",
        policy_seed,
        args,
        guidance_scale=args.guidance_scale if guided else 0.0,
        safety_distance=args.safety_distance if guided else None,
        frame_callback=audit,
    )
    metrics["contact_audit"] = audit.result()
    return metrics


def summarize(rows):
    eligible = [
        row for row in rows
        if row["clean"]["success"]
        and row["obstacle_baseline"]["collision_failure"]
    ]
    rescues = [row for row in eligible if row["obstacle_guided"]["safe_success"]]
    categories = ("arm_obstacle_collision_any", "gripper_obstacle_collision_any",
                  "hand_obstacle_collision_any", "finger_obstacle_collision_any",
                  "target_obstacle_collision_any")
    return {
        "pairs": len(rows),
        "clean_successes": int(sum(row["clean"]["success"] for row in rows)),
        "obstacle_collision_failures": int(
            sum(row["obstacle_baseline"]["collision_failure"] for row in rows)
        ),
        "eligible_pairs": len(eligible),
        "guided_safe_success_rescues": len(rescues),
        "rescue_rate": len(rescues) / len(eligible) if eligible else None,
        "eligible_guided_task_successes": int(
            sum(row["obstacle_guided"]["success"] for row in eligible)
        ),
        "eligible_guided_safe_successes": len(rescues),
        "eligible_guided_collision_failures": int(
            sum(row["obstacle_guided"]["collision_failure"] for row in eligible)
        ),
        "eligible_guided_collision_free_non_completion": int(
            sum(row["obstacle_guided"]["ncr"] for row in eligible)
        ),
        "eligible_collision_categories": {
            category: {
                "baseline": int(sum(row["obstacle_baseline"][category] for row in eligible)),
                "guided": int(sum(row["obstacle_guided"][category] for row in eligible)),
            }
            for category in categories
        },
        "eligible_keys": [
            {"scene": row["scene"], "policy_seed": row["policy_seed"]} for row in eligible
        ],
        "rescued_keys": [
            {"scene": row["scene"], "policy_seed": row["policy_seed"]} for row in rescues
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--bank-seed", type=int, default=819)
    parser.add_argument("--scenes", nargs="+", type=int, default=[1, 4, 5, 7, 8])
    parser.add_argument("--policy-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--guidance-scale", type=float, default=45.0)
    parser.add_argument("--safety-distance", type=float, default=0.08)
    parser.add_argument("--smoothing", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, required=True)
    cli = parser.parse_args()
    if cli.out_dir.exists():
        raise FileExistsError(f"output already exists: {cli.out_dir}")
    cli.out_dir.mkdir(parents=True)

    bank = PAIRED_EVAL.load_state_bank(cli.state_bank)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if device.type != "cuda":
        raise RuntimeError("counterfactual evaluation requires CUDA")
    policy, checkpoint = paper_guided_policy_from_checkpoint(
        ckpt_path=str(cli.checkpoint.resolve()), device=device
    )
    clean_env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint, env_name=CLEAN_ENV, render=False, render_offscreen=True, verbose=False
    )
    obstacle_env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint, env_name=OBSTACLE_ENV, render=False, render_offscreen=True, verbose=False
    )
    args = SimpleNamespace(
        horizon=cli.horizon,
        terminate_on_success=True,
        terminate_on_collision=False,
        target_object_name="Can",
        xy_only=True,
        disable_hand_collision=False,
        obstacle_cost_type="huber_hinge",
        obstacle_cost_smoothing=cli.smoothing,
        obstacle_cost_smoothing_ratio=None,
        tangent_ratio=0.0,
        guidance_phase="always",
        guidance_scale=cli.guidance_scale,
        safety_distance=cli.safety_distance,
    )
    rows = []
    layout_audits = {}
    try:
        with (cli.out_dir / "episode_metrics.jsonl").open("w") as stream:
            for scene in cli.scenes:
                key = f"{OBSTACLE_ENV}:{cli.bank_seed}:{scene}"
                record = bank["states"][key]
                if PAIRED_EVAL.state_hash(record["state"]) != record["state_hash"]:
                    raise RuntimeError(f"state hash mismatch for {key}")
                layout = materialize_obstacle_layout(obstacle_env, record)
                layout_audits[str(scene)] = initial_match(clean_env, obstacle_env, layout)
                for policy_seed in cli.policy_seeds:
                    clean = rollout(policy, clean_env, layout, policy_seed, args, obstacle=False, guided=False)
                    obstacle_baseline = rollout(
                        policy, obstacle_env, layout, policy_seed, args, obstacle=True, guided=False
                    )
                    obstacle_guided = rollout(
                        policy, obstacle_env, layout, policy_seed, args, obstacle=True, guided=True
                    )
                    row = {
                        "scene": scene,
                        "policy_seed": policy_seed,
                        "clean": clean,
                        "obstacle_baseline": obstacle_baseline,
                        "obstacle_guided": obstacle_guided,
                    }
                    rows.append(row)
                    stream.write(json.dumps(PAIRED_EVAL.rollout_common.json_safe(row)) + "\n")
                    stream.flush()
                    eligible = clean["success"] and obstacle_baseline["collision_failure"]
                    print(
                        f"scene={scene} noise={policy_seed} clean={int(clean['success'])} "
                        f"obstacle={int(obstacle_baseline['success'])}/{int(obstacle_baseline['collision_any'])} "
                        f"guided={int(obstacle_guided['success'])}/{int(obstacle_guided['collision_any'])} "
                        f"eligible={int(eligible)} rescue={int(eligible and obstacle_guided['safe_success'])}",
                        flush=True,
                    )
    finally:
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(clean_env).close()
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(obstacle_env).close()

    summary = summarize(rows)
    (cli.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "checkpoint": str(cli.checkpoint.resolve()),
        "state_bank": str(cli.state_bank.resolve()),
        "bank_seed": cli.bank_seed,
        "scenes": cli.scenes,
        "policy_seeds": cli.policy_seeds,
        "horizon": cli.horizon,
        "guidance": {"rho": cli.guidance_scale, "Q_m": cli.safety_distance,
                     "cost": "huber_hinge", "smoothing_m": cli.smoothing,
                     "xy_only": True, "tangent_ratio": 0.0},
        "eligibility": "clean success AND obstacle-baseline collision failure",
        "rescue": "eligible AND obstacle-guided safe task success",
        "initial_state_match": layout_audits,
        "eef_collision_note": (
            "The EEF is a MuJoCo site and has no contact geom. Physical contacts are classified "
            "as gripper hand/finger, arm link, or carried Can; EEF-Milk is reported only as an "
            "XY footprint-overlap proxy."
        ),
    }
    (cli.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
