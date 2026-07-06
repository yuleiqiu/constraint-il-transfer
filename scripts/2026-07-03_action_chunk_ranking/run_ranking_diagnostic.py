"""Same-state diagnostic for action-chunk ranking.

This script asks whether the diffusion policy samples any lower-cost geometry
chunks from the same simulator state. It compares the first sampled chunk
against the chunk selected by ranking, then executes both from the identical
state for a short horizon and records actual obstacle clearance.
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ROBOMIMIC_SCRIPTS = ROOT / "third_party" / "robomimic" / "robomimic" / "scripts"
sys.path.insert(0, str(ROBOMIMIC_SCRIPTS))

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils
import robomimic.utils.osc_forward_model_utils as OSCForwardModelUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo.guided_diffusion_policy import wrap_as_guided

from run_obstacle_guided_agent import (
    env_from_checkpoint_for_guidance,
    get_current_eef_pos_from_obs,
    make_json_serializable,
    set_obstacle_guidance_context,
)


def min_oracle_xy_clearance(env, obs, args):
    centers, radii, _, _, _ = ObstacleGuidanceUtils.get_oracle_obstacle_geometry(
        env=env,
        target_object_name=args.target_object_name,
        obstacle_names=args.obstacle_names,
        xy_clearance=0.0,
    )
    if len(centers) == 0:
        return None
    eef = get_current_eef_pos_from_obs(obs=obs, obs_key=args.eef_pos_obs_key)
    centers = np.asarray(centers, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    return float(np.min(np.linalg.norm(centers[:, :2] - eef[None, :2], axis=1) - radii))


def execute_chunk(env, state, action_chunk, args):
    obs = env.reset_to(state)
    clearances = []
    rewards = []
    success = False
    for action in action_chunk[: args.execute_horizon]:
        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward))
        clearance = min_oracle_xy_clearance(env=env, obs=obs, args=args)
        if clearance is not None:
            clearances.append(clearance)
        success = bool(env.is_success()["task"])
        if done or success:
            break
    return dict(
        steps=len(rewards),
        return_sum=float(np.sum(rewards)),
        success=success,
        min_xy_clearance=None if len(clearances) == 0 else float(np.min(clearances)),
        final_xy_clearance=None if len(clearances) == 0 else float(clearances[-1]),
    )


def build_payload(args, rows):
    aggregate = {}
    for k in sorted({row["num_candidates"] for row in rows}):
        subset = [row for row in rows if row["num_candidates"] == k]
        actual_deltas = [
            row["actual_min_clearance_improvement"]
            for row in subset
            if row["actual_min_clearance_improvement"] is not None
        ]
        aggregate[str(k)] = dict(
            states=len(subset),
            mean_safe_rate=float(np.mean([row["safe_rate"] for row in subset])) if subset else 0.0,
            mean_cost_improvement=float(np.mean([row["cost_improvement"] for row in subset])) if subset else 0.0,
            actual_improved_count=int(np.sum([delta > 0.0 for delta in actual_deltas])),
            actual_compared_count=len(actual_deltas),
            mean_actual_min_clearance_improvement=(
                float(np.mean(actual_deltas)) if len(actual_deltas) > 0 else 0.0
            ),
        )

    args_payload = {k: v for k, v in vars(args).items() if k != "forward_model"}
    return dict(args=args_payload, aggregate=aggregate, rows=rows)


def write_payload(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(make_json_serializable(payload), f, indent=2)


def run_no_guidance_step(policy, env, obs, args):
    args.selection_mode = "none"
    set_obstacle_guidance_context(policy=policy, env=env, obs=obs, args=args, step_i=0)
    action = policy(ob=obs)
    next_obs, _, _, _ = env.step(action)
    return deepcopy(next_obs)


def diagnose_state(policy, env, obs, state, args, state_index):
    rows = []
    original_num_candidates = args.ranking_num_candidates
    for k in args.ranking_candidate_counts:
        policy.start_episode()
        args.selection_mode = "ranking"
        args.ranking_num_candidates = int(k)
        env.reset_to(state)
        set_obstacle_guidance_context(policy=policy, env=env, obs=obs, args=args, step_i=0)
        torch.manual_seed(int(args.seed) + state_index)
        np.random.seed(int(args.seed) + state_index)
        _ = policy(ob=obs)
        info = deepcopy(getattr(policy.policy, "last_obstacle_guidance_info", {}))

        first_chunk = np.asarray(info["ranking_first_action_chunk"], dtype=np.float32)
        selected_chunk = np.asarray(info["ranking_selected_action_chunk"], dtype=np.float32)
        first_exec = execute_chunk(env=env, state=state, action_chunk=first_chunk, args=args)
        selected_exec = execute_chunk(env=env, state=state, action_chunk=selected_chunk, args=args)

        rows.append(dict(
            state_index=state_index,
            num_candidates=int(k),
            best_index=int(info["best_index"]),
            first_cost=float(info["ranking_first_cost"]),
            best_cost=float(info["ranking_best_cost"]),
            cost_improvement=float(info["ranking_cost_improvement"]),
            safe_count=int(info["ranking_safe_count"]),
            safe_rate=float(info["ranking_safe_rate"]),
            first_pred_distance=info.get("ranking_first_distance", None),
            best_pred_distance=info.get("ranking_best_distance", None),
            pred_distance_improvement=info.get("ranking_distance_improvement", None),
            first_actual=first_exec,
            selected_actual=selected_exec,
            actual_min_clearance_improvement=(
                None if first_exec["min_xy_clearance"] is None or selected_exec["min_xy_clearance"] is None
                else selected_exec["min_xy_clearance"] - first_exec["min_xy_clearance"]
            ),
        ))
    args.ranking_num_candidates = original_num_candidates
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--env", type=str, default="PickPlaceBreadCerealCan")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_states", type=int, default=20)
    parser.add_argument("--state_stride", type=int, default=8)
    parser.add_argument("--execute_horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--horizon", type=int, default=None)

    parser.add_argument("--trajectory_backend", type=str, choices=["cumsum", "forward_model"], default="forward_model")
    parser.add_argument("--forward_model_path", type=str, default=None)
    parser.add_argument(
        "--forward_model_state_obs_keys",
        type=str,
        nargs="+",
        default=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    )
    parser.add_argument("--ranking_num_candidates", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--ranking_safe_cost_threshold", type=float, default=1e-8)
    parser.add_argument("--ranking_cost_tie_tolerance", type=float, default=1e-10)
    parser.add_argument("--ranking_only_if_first_unsafe", action="store_true")

    parser.add_argument("--guidance_geometry_source", type=str, choices=["oracle_center"], default="oracle_center")
    parser.add_argument("--guidance_mode", type=str, choices=["xy", "xyz_cylinder"], default="xy")
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--guidance_horizon", type=int, default=8)
    parser.add_argument("--guidance_schedule", type=str, default="late")
    parser.add_argument("--guidance_start_step_pct", type=float, default=0.7)
    parser.add_argument("--xy_clearance", type=float, default=0.02)
    parser.add_argument("--z_clearance", type=float, default=0.03)
    parser.add_argument("--guidance_position_only", action="store_true")
    parser.add_argument("--target_object_name", type=str, default="Can")
    parser.add_argument("--obstacle_names", type=str, nargs="*", default=None)
    parser.add_argument("--eef_pos_obs_key", type=str, default="robot0_eef_pos")
    parser.add_argument("--final_collision_refine", action="store_true")
    parser.add_argument("--collision_refine_steps", type=int, default=5)
    parser.add_argument("--collision_refine_scale", type=float, default=0.02)
    parser.add_argument("--final_collision_cost_threshold", type=float, default=1e-8)

    # Unused pointcloud fields required by the shared context helper.
    parser.add_argument("--pc_distance_mode", type=str, default="xy")
    parser.add_argument("--pc_safe_distance", type=float, default=0.02)
    parser.add_argument("--pc_camera_name", type=str, default="agentview")
    parser.add_argument("--pc_depth_obs_key", type=str, default="agentview_depth")
    parser.add_argument("--pc_voxel_size", type=float, default=0.005)
    parser.add_argument("--pc_max_points", type=int, default=1024)
    parser.add_argument("--pc_workspace_crop", action="store_true", default=True)
    parser.add_argument("--pc_debug_visualization", action="store_true")
    parser.add_argument("--pc_debug_dir", type=str, default="outputs/pc1_debug")
    parser.add_argument("--pc_debug_interval", type=int, default=25)
    parser.add_argument("--no_pc_cache", action="store_true")
    return parser.parse_args()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.ranking_candidate_counts = list(args.ranking_num_candidates)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=True)
    wrap_as_guided(policy)
    args.forward_model = None
    if args.trajectory_backend == "forward_model":
        if args.forward_model_path is None:
            raise ValueError("--trajectory_backend forward_model requires --forward_model_path")
        args.forward_model = OSCForwardModelUtils.load_osc_forward_model(args.forward_model_path, device=device)

    if args.horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        args.horizon = config.experiment.rollout.horizon

    if args.pc_depth_obs_key not in ObsUtils.OBS_KEYS_TO_MODALITIES:
        ObsUtils.OBS_KEYS_TO_MODALITIES[args.pc_depth_obs_key] = "depth"
    env = env_from_checkpoint_for_guidance(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=False,
        use_depth_obs=False,
    )
    EnvUtils.set_env_specific_obs_processing(env=env)

    rows = []
    try:
        policy.start_episode()
        obs = env.reset()
        state = env.get_state()
        obs = env.reset_to(state)
        for state_index in range(args.num_states):
            state = env.get_state()
            new_rows = diagnose_state(policy=policy, env=env, obs=obs, state=state, args=args, state_index=state_index)
            rows.extend(new_rows)
            payload = build_payload(args=args, rows=rows)
            write_payload(args.output, payload)
            print(
                "State {}/{} done; rows={}; aggregate={}".format(
                    state_index + 1,
                    args.num_states,
                    len(rows),
                    json.dumps(make_json_serializable(payload["aggregate"]), sort_keys=True),
                ),
                flush=True,
            )
            env.reset_to(state)
            policy.start_episode()
            for _ in range(args.state_stride):
                obs = run_no_guidance_step(policy=policy, env=env, obs=obs, args=args)
                if env.is_success()["task"]:
                    break
            if env.is_success()["task"]:
                break
    finally:
        try:
            env.close()
        except AttributeError:
            pass

    payload = build_payload(args=args, rows=rows)
    write_payload(args.output, payload)
    print(json.dumps(make_json_serializable(payload["aggregate"]), indent=2))
    print("Wrote diagnostic results to {}".format(args.output))


if __name__ == "__main__":
    main(parse_args())
