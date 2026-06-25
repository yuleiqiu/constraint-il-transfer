"""
Diagnose what obstacle guidance cost signals look like during diffusion denoising.

Monkey-patches the algorithm to record per-denoising-step cost, grad_norm,
and computed oracle cost (using exact object geometry) for each guided step.

Usage:
    uv run python scripts/diagnose_guidance_gradient.py \
        --agent /path/to/model.pth \
        --env PickPlaceBreadCerealMilkCan \
        --output /tmp/diag_gradients.json

Analysis questions this answers:
  (1) Is cost dominated by denoising noise (high at early steps, low at late steps)?
  (2) Is pointcloud cost much smaller than oracle cost (sparse point cloud)?
  (3) Does cost vary between episodes, or is it constant across diffusion calls?
  (4) Does oracle cost remain very low → strategy's x0_hat doesn't penetrate obstacles?
"""
import argparse
import json
import sys
import os
import numpy as np

import torch
import torch.nn.functional as F

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils
from robomimic.scripts.run_obstacle_guided_agent import (
    env_from_checkpoint_for_guidance,
    set_obstacle_guidance_context,
    get_current_eef_pos_from_obs,
)
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo.guided_diffusion_policy import wrap_as_guided


def oracle_xyz_cylinder_cost(action_chunk, centers_xyz, radii, top_z,
                              action_scale=None, action_offset=None,
                              delta_pos_scale=None, delta_pos_offset=None,
                              z_clearance=0.03, eef_z_max=1.2):
    """Compute penetration cost of action waypoints into oracle obstacle cylinders."""
    if len(centers_xyz) == 0 or len(radii) == 0:
        return torch.tensor(0.0, device=action_chunk.device), dict(
            min_distance=np.nan, min_xy_distance=np.nan,
            min_z_clearance=np.nan, num_obstacles=0)

    device = action_chunk.device
    traj = action_chunk.float()

    if action_scale is not None and action_offset is not None:
        action_scale = torch.as_tensor(action_scale, device=device, dtype=torch.float32)
        action_offset = torch.as_tensor(action_offset, device=device, dtype=torch.float32)
        traj = traj * action_scale + action_offset

    centers = torch.as_tensor(centers_xyz, device=device, dtype=torch.float32)
    r = torch.as_tensor(radii, device=device, dtype=torch.float32)
    tz = torch.as_tensor(top_z, device=device, dtype=torch.float32)

    B, H, D = traj.shape
    waypoints_xy = traj[..., :2]
    waypoints_z = traj[..., 2]

    centers_xy = centers[None, None, :, :2]
    r = r[None, None, :]
    tz = tz[None, None, :]

    dist_xy = torch.sqrt(((waypoints_xy[:, :, None, :] - centers_xy) ** 2).sum(dim=-1) + 1e-8)
    pen_xy = F.relu(r - dist_xy)
    pen_z = F.relu((tz + z_clearance) - waypoints_z[:, :, None])
    costs = pen_xy.pow(2) * pen_z.pow(2)
    total_cost = costs.sum()

    min_xy_dist = (dist_xy - r).min().item()
    min_z = (waypoints_z[:, :, None] - tz).min().item()

    return total_cost, dict(
        min_distance=min_xy_dist,
        min_xy_distance=min_xy_dist,
        min_z_clearance=min_z,
        num_obstacles=len(radii),
    )


def run_diagnostic_rollout(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, device=device, verbose=False)
    wrap_as_guided(policy)  # add guidance interface

    needs_depth = (args.guidance_geometry_source == "pointcloud")
    if needs_depth:
        ObsUtils.OBS_KEYS_TO_MODALITIES[args.pc_depth_obs_key] = "depth"

    env = env_from_checkpoint_for_guidance(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=True,
        use_depth_obs=needs_depth,
    )

    algo = getattr(policy, "policy", policy)
    if not hasattr(algo, "set_obstacle_guidance_context"):
        raise ValueError("Loaded policy does not support obstacle guidance context")

    all_chunk_logs = []
    all_chunk_meta = []

    # Intercept _guided_scheduler_step per chunk to collect diagnostics
    original_guided_step = algo._guided_scheduler_step

    def diag_guided_step(self, nets, naction, timestep, obs_cond, step_index, num_steps, guidance_start_step=0):
        result = original_guided_step(
            nets=nets, naction=naction, timestep=timestep,
            obs_cond=obs_cond, step_index=step_index,
            num_steps=num_steps, guidance_start_step=guidance_start_step,
        )
        info = self.last_obstacle_guidance_info
        if info and info.get("applied"):
            all_chunk_logs.append(dict(
                step_index=step_index,
                cost=info.get("cost", 0.0),
                grad_norm=info.get("grad_norm", None),
                rho_t=info.get("rho_t", 0.0),
            ))
        return result

    algo._guided_scheduler_step = diag_guided_step.__get__(algo, type(algo))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    step = 0
    success = False
    eef_history = []

    for step in range(args.horizon):
        obstacle_info = set_obstacle_guidance_context(
            policy=policy, env=env, obs=obs, args=args, step_i=step)
        print("[step {:3d}] guidance_mode={}  pc_points={}".format(
            step,
            obstacle_info["guidance_mode"],
            obstacle_info.get("pc_stats", {}).get("point_count", "N/A"),
        ), flush=True)

        act = policy(ob=obs)
        next_obs, reward, done, _ = env.step(act)
        eef = get_current_eef_pos_from_obs(obs=next_obs, obs_key=args.eef_pos_obs_key)
        eef_history.append(eef.tolist())

        success = env.is_success()["task"]
        if done or success:
            break
        obs = next_obs

    print("\nRollout: horizon={}  success={}".format(step + 1, success))

    result = dict(
        horizon=step + 1,
        success=bool(success),
        eef_history=eef_history,
        num_diffusion_calls=len(all_chunk_logs),
        chunks=[],
    )

    for ci, chunk_log in enumerate(all_chunk_logs):
        chunk_entry = dict(call_index=ci, num_guided_steps=1, steps=[chunk_log])
        result["chunks"].append(chunk_entry)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else float(x))
        print("Wrote diagnostics to {}".format(args.output))

    # Print summary
    print("\n=== Guidance Diagnostic Summary ===")
    for ci, chunk_log in enumerate(all_chunk_logs):
        print("chunk {}: cost={:.6f}  rho_t={:.4f}".format(
            ci, chunk_log.get("cost", 0), chunk_log.get("rho_t", 0)))

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str,
        default="/home/yulei/codes/robomimic/robomimic/runs/trained_models/diffusion_policy_can_yq_masked_image/20260506153143/models/model_epoch_140_image_v15_can_mask_success_1.0.pth",
        help="path to checkpoint")
    parser.add_argument("--env", type=str, default="PickPlaceBreadCerealMilkCan")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=600)
    parser.add_argument("--output", type=str, default="/tmp/diag_guidance.json")

    parser.add_argument("--guidance_geometry_source", type=str, default="pointcloud")
    parser.add_argument("--guidance_scale", type=float, default=0.03)
    parser.add_argument("--guidance_mode", type=str, default="xyz_cylinder")
    parser.add_argument("--guidance_schedule", type=str, default="late")
    parser.add_argument("--guidance_start_step_pct", type=float, default=0.7)
    parser.add_argument("--guidance_horizon", type=int, default=8)
    parser.add_argument("--xy_clearance", type=float, default=0.02)
    parser.add_argument("--z_clearance", type=float, default=0.03)
    parser.add_argument("--target_object_name", type=str, default="Can")
    parser.add_argument("--obstacle_names", type=str, nargs="*", default=None)
    parser.add_argument("--eef_pos_obs_key", type=str, default="robot0_eef_pos")
    parser.add_argument("--final_collision_refine", action="store_true")
    parser.add_argument("--collision_refine_steps", type=int, default=5)
    parser.add_argument("--collision_refine_scale", type=float, default=0.02)
    parser.add_argument("--final_collision_cost_threshold", type=float, default=1e-8)

    parser.add_argument("--pc_camera_name", type=str, default="agentview")
    parser.add_argument("--pc_depth_obs_key", type=str, default="agentview_depth")
    parser.add_argument("--pc_safe_distance", type=float, default=0.02)
    parser.add_argument("--pc_distance_mode", type=str, default="xy")
    parser.add_argument("--pc_voxel_size", type=float, default=0.005)
    parser.add_argument("--pc_max_points", type=int, default=1024)
    parser.add_argument("--pc_workspace_crop", action="store_true", default=True)
    parser.add_argument("--pc_no_workspace_crop", action="store_false", dest="pc_workspace_crop")
    parser.add_argument("--pc_debug_visualization", action="store_true")
    parser.add_argument("--pc_debug_dir", type=str, default="outputs/pc1_debug")
    parser.add_argument("--pc_debug_interval", type=int, default=25)
    parser.add_argument("--no_pc_cache", action="store_true")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["agentview"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video_skip", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    run_diagnostic_rollout(parse_args())
