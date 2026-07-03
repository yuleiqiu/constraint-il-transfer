"""Counterfactual diagnostic for obstacle guidance updates.

For the same observation and the same diffusion initial noise, this script
generates:

1. an unguided action chunk,
2. a guided action chunk,
3. predicted EEF trajectories for both chunks, and
4. actual EEF trajectories by restoring the simulator state and executing each
   chunk.

The diagnostic checks whether guidance improves predicted obstacle clearance
and whether the actual executed trajectory agrees.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils
import robomimic.utils.osc_forward_model_utils as OSCForwardModelUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo.guided_diffusion_policy import wrap_as_guided
from robomimic.scripts.run_obstacle_guided_agent import (
    env_from_checkpoint_for_guidance,
    get_action_normalization_vector,
    get_current_eef_pos_from_obs,
    get_forward_model_state_from_obs,
    set_obstacle_guidance_context,
)


def make_guidance_args(args, forward_model):
    return SimpleNamespace(
        guidance_geometry_source=args.guidance_geometry_source,
        guidance_scale=args.guidance_scale,
        guidance_mode=args.guidance_mode,
        xy_clearance=args.xy_clearance,
        z_clearance=args.z_clearance,
        guidance_horizon=args.guidance_horizon,
        guidance_position_only=args.guidance_position_only,
        trajectory_backend=args.trajectory_backend,
        forward_model_path=args.forward_model_path,
        forward_model=forward_model,
        forward_model_state_obs_keys=args.forward_model_state_obs_keys,
        guidance_schedule=args.guidance_schedule,
        guidance_start_step_pct=args.guidance_start_step_pct,
        target_object_name=args.target_object_name,
        obstacle_names=args.obstacle_names,
        eef_pos_obs_key=args.eef_pos_obs_key,
        final_collision_refine=False,
        collision_refine_steps=0,
        collision_refine_scale=0.0,
        final_collision_cost_threshold=1e-8,
        pc_distance_mode="xy",
        pc_safe_distance=0.02,
        pc_camera_name="agentview",
        pc_depth_obs_key="agentview_depth",
        pc_voxel_size=0.005,
        pc_max_points=1024,
        pc_workspace_crop=True,
        pc_debug_visualization=False,
        pc_debug_dir="outputs/pc_debug",
        pc_debug_interval=25,
    )


def unnormalize_chunk(policy, chunk):
    action_scale, action_offset = get_action_normalization_vector(policy)
    chunk_t = torch.as_tensor(chunk, dtype=torch.float32)
    out = ObstacleGuidanceUtils.unnormalize_action_chunk(
        action_chunk=chunk_t,
        action_scale=action_scale,
        action_offset=action_offset,
    )
    return TensorUtils.to_numpy(out)


def predict_traj(algo, chunk, context):
    action_scale = context.get("action_scale", None)
    action_offset = context.get("action_offset", None)
    chunk_t = torch.as_tensor(chunk, dtype=torch.float32, device=algo.device)
    action_for_cost = ObstacleGuidanceUtils.unnormalize_action_chunk(
        action_chunk=chunk_t,
        action_scale=action_scale,
        action_offset=action_offset,
    )
    trajectory_model = context.get("trajectory_model", None)
    if trajectory_model is not None and action_for_cost.shape[1] < trajectory_model.horizon:
        pad_len = trajectory_model.horizon - action_for_cost.shape[1]
        pad = action_for_cost[:, -1:, :].expand(-1, pad_len, -1)
        action_for_cost = torch.cat([action_for_cost, pad], dim=1)
    traj = ObstacleGuidanceUtils.action_chunk_to_eef_xyz_traj(
        action_chunk=action_for_cost,
        current_eef_pos=context["current_eef_pos"],
        horizon=context["guidance_horizon"],
        delta_pos_scale=context.get("delta_pos_scale", 1.0),
        delta_pos_offset=context.get("delta_pos_offset", 0.0),
        trajectory_model=trajectory_model,
        trajectory_model_state=context.get("trajectory_model_state", None),
    )
    return TensorUtils.to_numpy(traj[0])


def clearance_stats(traj_xyz, centers_xyz, radii):
    if len(centers_xyz) == 0:
        return dict(min_dist=None, min_clearance=None, collision=False)
    centers_xy = np.asarray(centers_xyz, dtype=np.float64)[:, :2]
    radii = np.asarray(radii, dtype=np.float64).reshape(1, -1)
    traj_xy = np.asarray(traj_xyz, dtype=np.float64)[:, :2]
    dist = np.linalg.norm(traj_xy[:, None, :] - centers_xy[None, :, :], axis=-1)
    clearance = dist - radii
    return dict(
        min_dist=float(np.min(dist)),
        min_clearance=float(np.min(clearance)),
        collision=bool(np.min(clearance) < 0.0),
    )


def execute_chunk_from_state(env, state, chunk_unnorm, horizon, eef_obs_key):
    obs = env.reset_to({"states": state["states"]})
    traj = []
    for i in range(horizon):
        obs, _, _, _ = env.step(chunk_unnorm[i])
        traj.append(get_current_eef_pos_from_obs(obs, obs_key=eef_obs_key).tolist())
    return np.asarray(traj, dtype=np.float64)


def make_chunk(policy, obs, guidance_args, enabled, sample_seed):
    algo = policy.policy
    prepared_obs = policy._prepare_observation(obs)
    if enabled:
        set_obstacle_guidance_context(policy=policy, env=guidance_args.env, obs=obs, args=guidance_args)
    else:
        algo.set_obstacle_guidance_context(None)
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    with torch.no_grad() if not enabled else torch.enable_grad():
        chunk = algo._get_action_trajectory(obs_dict=prepared_obs)
    return chunk.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--env", default="PickPlaceBreadCerealCan")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--active_only", action="store_true", help="only save samples where guidance changes the chunk")
    parser.add_argument("--active_threshold", type=float, default=1e-8)
    parser.add_argument("--rollout_step_stride", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--output", default="outputs/diagnostics/guidance_update_effect/diagnostics.json")
    parser.add_argument("--trajectory_backend", choices=["cumsum", "forward_model"], default="forward_model")
    parser.add_argument("--forward_model_path", default="outputs/forward_model/osc_eef_forward_image_v15/model.pth")
    parser.add_argument(
        "--forward_model_state_obs_keys",
        nargs="+",
        default=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    )
    parser.add_argument("--guidance_geometry_source", choices=["oracle_center"], default="oracle_center")
    parser.add_argument("--guidance_mode", choices=["xy", "xyz_cylinder"], default="xy")
    parser.add_argument("--guidance_scale", type=float, default=0.005)
    parser.add_argument("--guidance_horizon", type=int, default=8)
    parser.add_argument("--guidance_schedule", choices=["constant", "late"], default="late")
    parser.add_argument("--guidance_start_step_pct", type=float, default=0.7)
    parser.add_argument("--guidance_position_only", action="store_true")
    parser.add_argument("--xy_clearance", type=float, default=0.02)
    parser.add_argument("--z_clearance", type=float, default=0.03)
    parser.add_argument("--target_object_name", default="Can")
    parser.add_argument("--obstacle_names", nargs="*", default=None)
    parser.add_argument("--eef_pos_obs_key", default="robot0_eef_pos")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=False)
    wrap_as_guided(policy)
    policy.start_episode()

    forward_model = None
    if args.trajectory_backend == "forward_model":
        forward_model = OSCForwardModelUtils.load_osc_forward_model(args.forward_model_path, device=device)

    env = env_from_checkpoint_for_guidance(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=False,
        use_depth_obs=False,
    )
    obs = env.reset()

    guidance_args = make_guidance_args(args, forward_model)
    guidance_args.env = env

    samples = []
    step_i = 0
    num_candidates = 0
    while len(samples) < args.n_samples and step_i < args.max_steps:
        state = env.get_state()
        sample_seed = args.seed * 100000 + num_candidates
        num_candidates += 1

        unguided_chunk = make_chunk(policy, obs, guidance_args, enabled=False, sample_seed=sample_seed)
        set_obstacle_guidance_context(policy=policy, env=env, obs=obs, args=guidance_args)
        context = policy.policy.obstacle_guidance_context
        guided_chunk = make_chunk(policy, obs, guidance_args, enabled=True, sample_seed=sample_seed)
        context = policy.policy.obstacle_guidance_context

        unguided_np = TensorUtils.to_numpy(unguided_chunk)
        guided_np = TensorUtils.to_numpy(guided_chunk)
        guided_unnorm = unnormalize_chunk(policy, guided_np)[0]

        action_delta_l2 = float(np.linalg.norm(guided_np - unguided_np))
        first_action_delta_l2 = float(np.linalg.norm(guided_np[0, 0] - unguided_np[0, 0]))
        is_active = action_delta_l2 > args.active_threshold

        if (not args.active_only) or is_active:
            pred_unguided = predict_traj(policy.policy, unguided_np, context)
            pred_guided = predict_traj(policy.policy, guided_np, context)

            unguided_unnorm = unnormalize_chunk(policy, unguided_np)[0]

            actual_unguided = execute_chunk_from_state(
                env=env,
                state=state,
                chunk_unnorm=unguided_unnorm,
                horizon=args.guidance_horizon,
                eef_obs_key=args.eef_pos_obs_key,
            )
            actual_guided = execute_chunk_from_state(
                env=env,
                state=state,
                chunk_unnorm=guided_unnorm,
                horizon=args.guidance_horizon,
                eef_obs_key=args.eef_pos_obs_key,
            )

            centers = np.asarray(context["obstacle_centers_xyz"], dtype=np.float64)
            radii = np.asarray(context["obstacle_radii"], dtype=np.float64)
            pred_u_stats = clearance_stats(pred_unguided, centers, radii)
            pred_g_stats = clearance_stats(pred_guided, centers, radii)
            actual_u_stats = clearance_stats(actual_unguided, centers, radii)
            actual_g_stats = clearance_stats(actual_guided, centers, radii)

            pred_delta = pred_g_stats["min_clearance"] - pred_u_stats["min_clearance"]
            actual_delta = actual_g_stats["min_clearance"] - actual_u_stats["min_clearance"]

            samples.append(
                dict(
                    rollout_step=step_i,
                    sample_seed=sample_seed,
                    candidate_index=num_candidates - 1,
                    pred_unguided=pred_u_stats,
                    pred_guided=pred_g_stats,
                    actual_unguided=actual_u_stats,
                    actual_guided=actual_g_stats,
                    pred_clearance_delta=float(pred_delta),
                    actual_clearance_delta=float(actual_delta),
                    delta_sign_agrees=bool(np.sign(pred_delta) == np.sign(actual_delta)),
                    action_delta_l2=action_delta_l2,
                    first_action_delta_l2=first_action_delta_l2,
                )
            )

        obs = env.reset_to({"states": state["states"]})
        for i in range(args.guidance_horizon):
            obs, _, _, _ = env.step(guided_unnorm[i])
        step_i += args.guidance_horizon

    pred_improved = [s["pred_clearance_delta"] > 0 for s in samples]
    actual_improved = [s["actual_clearance_delta"] > 0 for s in samples]
    sign_agrees = [s["delta_sign_agrees"] for s in samples]
    active_updates = [s["action_delta_l2"] > 1e-8 for s in samples]
    result = dict(
        args=vars(args),
        num_samples=len(samples),
        num_candidates=num_candidates,
        summary=dict(
            active_update_count=int(np.sum(active_updates)),
            pred_improved_count=int(np.sum(pred_improved)),
            actual_improved_count=int(np.sum(actual_improved)),
            sign_agreement_count=int(np.sum(sign_agrees)),
            mean_pred_clearance_delta=float(np.mean([s["pred_clearance_delta"] for s in samples])) if samples else None,
            mean_actual_clearance_delta=float(np.mean([s["actual_clearance_delta"] for s in samples])) if samples else None,
            mean_action_delta_l2=float(np.mean([s["action_delta_l2"] for s in samples])) if samples else None,
            mean_first_action_delta_l2=float(np.mean([s["first_action_delta_l2"] for s in samples])) if samples else None,
        ),
        samples=samples,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    print("Wrote diagnostics to {}".format(output_path))


if __name__ == "__main__":
    main()
