"""Audit one pre-collision Tabletop action chunk through all DDPM timesteps."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.guided_denoising_utils import unnormalize_action_points
from robomimic.utils.paper_lan_guidance_utils import (
    closest_obstacle_point,
    paper_guidance_gradient,
    paper_guided_policy_from_checkpoint,
    paper_lan_guidance_context_from_rollout_policy,
    paper_obstacle_cost,
    predicted_clean_action,
)

from counterfactual_rescue_eval import (
    OBSTACLE_ENV,
    PAIRED_EVAL,
    apply_layout,
    materialize_obstacle_layout,
)


def copy_observation(observation, policy):
    return {
        key: np.asarray(observation[key]).copy()
        for key in policy.policy.global_config.all_obs_keys
    }


def capture_pre_collision_chunk(policy, env, layout, policy_seed, args):
    apply_layout(env, layout, obstacle=True)
    snapshots = []

    def capture(_env, observation, step, current, obstacle_cloud, _context):
        snapshots.append(
            {
                "step": int(step),
                "observation": copy_observation(observation, policy),
                "current_eef": np.asarray(current).copy(),
                "obstacle_cloud": np.asarray(obstacle_cloud).copy(),
                "cpu_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_state": torch.cuda.get_rng_state(policy.policy.device).clone(),
            }
        )

    metrics = PAIRED_EVAL.rollout_condition(
        policy,
        env,
        {"states": layout["states"]},
        "baseline",
        policy_seed,
        args,
        chunk_callback=capture,
    )
    collision_step = metrics["first_collision_step"]
    if collision_step is None:
        raise RuntimeError("selected baseline rollout never collides")
    candidates = [item for item in snapshots if item["step"] < collision_step]
    if not candidates:
        raise RuntimeError("no chunk start precedes the first collision")
    return metrics, max(candidates, key=lambda item: item["step"])


def cloud_distances(raw_action, obstacle_cloud):
    positions = raw_action[..., :2]
    cloud = torch.as_tensor(
        obstacle_cloud[:, :2], device=positions.device, dtype=positions.dtype
    )
    return torch.min(
        torch.linalg.vector_norm(positions.unsqueeze(-2) - cloud.reshape(1, 1, -1, 2), dim=-1),
        dim=-1,
    ).values


def run_reverse_audit(policy, snapshot, context, initial_noisy, *, apply_guidance):
    prepared = policy._prepare_observation(snapshot["observation"])
    algo = policy.policy
    nets = algo.ema.averaged_model if algo.ema is not None else algo.nets
    inputs = {"obs": prepared, "goal": None}
    with torch.no_grad():
        features = TensorUtils.time_distributed(
            inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True
        )
        observation_condition = features.flatten(start_dim=1)
    scheduler = algo.noise_scheduler
    if algo.algo_config.ddpm.enabled:
        inference_steps = algo.algo_config.ddpm.num_inference_timesteps
    else:
        inference_steps = algo.algo_config.ddim.num_inference_timesteps
    scheduler.set_timesteps(inference_steps)
    fixed_point = closest_obstacle_point(
        context.current_eef_pos,
        torch.as_tensor(
            context.obstacle_points,
            device=algo.device,
            dtype=initial_noisy.dtype,
        ),
        xy_only=context.xy_only,
        dimensions=2,
    )
    context = type(context)(
        **{**context.__dict__, "obstacle_points": fixed_point.detach().reshape(1, -1)}
    )
    current = initial_noisy.clone()
    rows = []
    start = int(algo.algo_config.horizon.observation_horizon) - 1
    end = start + int(algo.algo_config.horizon.action_horizon)
    for timestep in scheduler.timesteps:
        with torch.enable_grad():
            differentiable = current.detach().requires_grad_(True)
            model_output = nets["policy"]["noise_pred_net"](
                sample=differentiable,
                timestep=timestep,
                global_cond=observation_condition,
            )
            clean = predicted_clean_action(
                differentiable, model_output, timestep, scheduler
            )
            raw_clean = unnormalize_action_points(
                clean, context.action_scale, context.action_offset
            )
            _, full_fixed_distances, full_penetrations = paper_obstacle_cost(
                raw_clean,
                fixed_point,
                safety_distance=context.safety_distance,
                xy_only=True,
                norm_epsilon=context.norm_epsilon,
                position_indices=context.position_indices,
                cost_type=context.obstacle_cost_type,
                smoothing=context.obstacle_cost_smoothing,
            )
            executable_raw = raw_clean[:, start:end]
            executable_cost, executable_fixed_distances, executable_penetrations = paper_obstacle_cost(
                executable_raw,
                fixed_point,
                safety_distance=context.safety_distance,
                xy_only=True,
                norm_epsilon=context.norm_epsilon,
                position_indices=context.position_indices,
                cost_type=context.obstacle_cost_type,
                smoothing=context.obstacle_cost_smoothing,
            )
            update, diagnostics = paper_guidance_gradient(
                noisy_action=differentiable,
                model_output=model_output,
                timestep=timestep,
                scheduler=scheduler,
                context=context,
            )
        with torch.no_grad():
            reverse = scheduler.step(
                model_output=model_output.detach(),
                timestep=timestep,
                sample=differentiable.detach(),
            ).prev_sample
            next_action = reverse - update if apply_guidance else reverse
            full_cloud = cloud_distances(raw_clean, snapshot["obstacle_cloud"])
            executable_cloud = full_cloud[:, start:end]
            executable_update = update[:, start:end]
            non_executable_update = torch.cat((update[:, :start], update[:, end:]), dim=1)
            scale = float(context.guidance_scale)
        rows.append(
            {
                "timestep": int(timestep.item()),
                "implemented_full_horizon": {
                    "minimum_fixed_point_distance_m": float(full_fixed_distances.min().item()),
                    "minimum_obstacle_cloud_distance_m": float(full_cloud.min().item()),
                    "active_waypoints": int(torch.count_nonzero(full_penetrations > 0).item()),
                    "cost": float(diagnostics.cost),
                },
                "executable_slice": {
                    "start": start,
                    "end_exclusive": end,
                    "minimum_fixed_point_distance_m": float(executable_fixed_distances.min().item()),
                    "minimum_obstacle_cloud_distance_m": float(executable_cloud.min().item()),
                    "active_waypoints": int(torch.count_nonzero(executable_penetrations > 0).item()),
                    "cost": float(executable_cost.detach().item()),
                },
                "noisy_action_gradient_norm": float(diagnostics.noisy_action_gradient_norm),
                "guidance_update_norm": float(diagnostics.applied_update_norm),
                "executable_noisy_action_gradient_norm": float(
                    torch.linalg.vector_norm(executable_update).item() / scale
                ),
                "executable_guidance_update_norm": float(
                    torch.linalg.vector_norm(executable_update).item()
                ),
                "non_executable_guidance_update_norm": float(
                    torch.linalg.vector_norm(non_executable_update).item()
                ),
                "scheduler_reverse_delta_norm": float(
                    torch.linalg.vector_norm(reverse - differentiable.detach()).item()
                ),
                "actual_net_delta_norm": float(
                    torch.linalg.vector_norm(next_action - differentiable.detach()).item()
                ),
                "guidance_applied": bool(apply_guidance),
            }
        )
        current = next_action
    raw_final = unnormalize_action_points(
        current, context.action_scale, context.action_offset
    ).detach()
    final_cloud = cloud_distances(raw_final, snapshot["obstacle_cloud"])
    return {
        "reverse_steps": rows,
        "final_normalized": current.detach().cpu().numpy(),
        "final_raw": raw_final.cpu().numpy(),
        "final_full_minimum_obstacle_cloud_distance_m": float(final_cloud.min().item()),
        "final_executable_minimum_obstacle_cloud_distance_m": float(
            final_cloud[:, start:end].min().item()
        ),
        "executable_start": start,
        "executable_end_exclusive": end,
        "fixed_obstacle_point": fixed_point.detach().cpu().numpy(),
    }


def compact_reverse(result):
    return {
        key: value
        for key, value in result.items()
        if key not in ("final_normalized", "final_raw")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--bank-seed", type=int, default=819)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
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
    record = bank["states"][f"{OBSTACLE_ENV}:{cli.bank_seed}:{cli.scene}"]
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, checkpoint = paper_guided_policy_from_checkpoint(
        ckpt_path=str(cli.checkpoint.resolve()), device=device
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=OBSTACLE_ENV,
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
        obstacle_cost_smoothing=cli.smoothing,
        obstacle_cost_smoothing_ratio=None,
        tangent_ratio=0.0,
        guidance_phase="always",
    )
    try:
        layout = materialize_obstacle_layout(env, record)
        baseline_metrics, snapshot = capture_pre_collision_chunk(
            policy, env, layout, cli.policy_seed, args
        )
        context = paper_lan_guidance_context_from_rollout_policy(
            policy,
            current_eef_pos=snapshot["current_eef"],
            obstacle_points=snapshot["obstacle_cloud"],
            guidance_scale=cli.guidance_scale,
            safety_distance=cli.safety_distance,
            xy_only=True,
            obstacle_cost_type="huber_hinge",
            obstacle_cost_smoothing=cli.smoothing,
        )
        torch.set_rng_state(snapshot["cpu_rng_state"])
        torch.cuda.set_rng_state(snapshot["cuda_rng_state"], policy.policy.device)
        prediction_horizon = int(policy.policy.algo_config.horizon.prediction_horizon)
        initial_noisy = torch.randn(
            (1, prediction_horizon, policy.policy.ac_dim), device=policy.policy.device
        )
        baseline = run_reverse_audit(
            policy, snapshot, context, initial_noisy, apply_guidance=False
        )
        guided = run_reverse_audit(
            policy, snapshot, context, initial_noisy, apply_guidance=True
        )
    finally:
        PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env).close()

    start, end = baseline["executable_start"], baseline["executable_end_exclusive"]
    final_delta = guided["final_raw"] - baseline["final_raw"]
    executable_xy_delta = np.linalg.norm(final_delta[:, start:end, :2], axis=-1)
    guided_rows = guided["reverse_steps"]
    result = {
        "scene": cli.scene,
        "policy_seed": cli.policy_seed,
        "baseline_rollout": baseline_metrics,
        "audited_chunk_start_step": snapshot["step"],
        "first_collision_step": baseline_metrics["first_collision_step"],
        "obstacle_cloud_points": int(len(snapshot["obstacle_cloud"])),
        "guidance": {"rho": cli.guidance_scale, "Q_m": cli.safety_distance,
                     "cost": "huber_hinge", "smoothing_m": cli.smoothing},
        "baseline_reverse": compact_reverse(baseline),
        "guided_reverse": compact_reverse(guided),
        "final_baseline_guided_difference": {
            "full_raw_l2": float(np.linalg.norm(final_delta)),
            "executable_xy_mean_m": float(np.mean(executable_xy_delta)),
            "executable_xy_max_m": float(np.max(executable_xy_delta)),
        },
        "diagnostic_flags": {
            "any_full_horizon_active_waypoint": any(
                row["implemented_full_horizon"]["active_waypoints"] > 0 for row in guided_rows
            ),
            "any_executable_active_waypoint": any(
                row["executable_slice"]["active_waypoints"] > 0 for row in guided_rows
            ),
            "any_nonzero_gradient": any(row["noisy_action_gradient_norm"] > 0 for row in guided_rows),
            "final_executable_changed_over_1mm": bool(np.max(executable_xy_delta) > 0.001),
            "guided_final_executable_all_outside_Q": bool(
                guided["final_executable_minimum_obstacle_cloud_distance_m"] >= cli.safety_distance
            ),
        },
    }
    (cli.out_dir / "audit.json").write_text(
        json.dumps(PAIRED_EVAL.rollout_common.json_safe(result), indent=2) + "\n"
    )
    np.savez_compressed(
        cli.out_dir / "final_trajectories.npz",
        initial_noisy=initial_noisy.detach().cpu().numpy(),
        baseline_full_raw=baseline["final_raw"],
        guided_full_raw=guided["final_raw"],
        obstacle_cloud=snapshot["obstacle_cloud"],
        current_eef=snapshot["current_eef"],
    )
    print(json.dumps({key: result[key] for key in (
        "scene", "policy_seed", "audited_chunk_start_step", "first_collision_step",
        "final_baseline_guided_difference", "diagnostic_flags"
    )}, indent=2))


if __name__ == "__main__":
    main()
