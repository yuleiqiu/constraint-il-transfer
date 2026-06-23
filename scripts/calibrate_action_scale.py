"""
Calibrate the mapping from policy action to actual EEF delta.

Records action[:3] and resulting EEF delta for 50 env steps.
Computes the ratio delta/action per dimension to determine the
correct delta_pos_scale for the obstacle guidance cost function.

Usage:
    uv run python scripts/calibrate_action_scale.py [--steps 50]
"""
import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.scripts.run_obstacle_guided_agent import (
    env_from_checkpoint_for_guidance,
    get_current_eef_pos_from_obs,
    get_action_normalization_vector,
)
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils


def run_calibration(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, device=device, verbose=False)

    ObsUtils.OBS_KEYS_TO_MODALITIES[args.pc_depth_obs_key] = "depth"

    env = env_from_checkpoint_for_guidance(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=True,
        use_depth_obs=True,
    )

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    action_scale, action_offset = get_action_normalization_vector(policy)
    delta_pos_scale, delta_pos_offset = ObstacleGuidanceUtils.get_controller_delta_pos_mapping(env)

    print("=== Controller / Action config ===")
    print(f"action_scale[:3]  = {action_scale.flatten()[:3].tolist() if action_scale is not None else 'NONE'}")
    print(f"action_offset[:3] = {action_offset.flatten()[:3].tolist() if action_offset is not None else 'NONE'}")
    print(f"delta_pos_scale[:3]  = {delta_pos_scale[:3].tolist()}")
    print(f"delta_pos_offset[:3] = {delta_pos_offset[:3].tolist()}")
    print()

    ratios_x, ratios_y, ratios_z = [], [], []
    actions_x, actions_y, actions_z = [], [], []
    deltas_x, deltas_y, deltas_z = [], [], []

    for step in range(args.steps):
        eef_before = get_current_eef_pos_from_obs(obs=obs, obs_key=args.eef_pos_obs_key)
        act = policy(ob=obs)
        an = act[0].numpy() if hasattr(act, 'numpy') else np.array(act)
        next_obs, _r, _d, _ = env.step(act)
        eef_after = get_current_eef_pos_from_obs(obs=next_obs, obs_key=args.eef_pos_obs_key)

        actual_delta = eef_after[:3] - eef_before[:3]
        action_3d = an[:3]

        # Current cost-function mapping
        predicted_delta = action_3d * delta_pos_scale[:3] + delta_pos_offset[:3]

        for i, dim_name in enumerate(["X", "Y", "Z"]):
            if abs(action_3d[i]) > 1e-6:
                ratio = actual_delta[i] / action_3d[i]
            else:
                ratio = float('nan')
            if not np.isnan(ratio) and not np.isinf(ratio):
                [ratios_x, ratios_y, ratios_z][i].append(ratio)
            [actions_x, actions_y, actions_z][i].append(action_3d[i])
            [deltas_x, deltas_y, deltas_z][i].append(actual_delta[i])

        if step % 10 == 0:
            print(f"[step {step:3d}] action={action_3d.round(4)}  "
                  f"delta={actual_delta.round(4)}  "
                  f"predicted={predicted_delta.round(4)}")

        obs = next_obs

    print()
    print("=== Calibration results ===")
    for dim_name, ratios, actions_list, deltas_list in [
        ("X", ratios_x, actions_x, deltas_x),
        ("Y", ratios_y, actions_y, deltas_y),
        ("Z", ratios_z, actions_z, deltas_z),
    ]:
        valid = [r for r in ratios if not np.isnan(r)]
        if valid:
            mean_r = np.mean(valid)
            std_r = np.std(valid)
            print(f"  {dim_name}: ratio delta/action = {mean_r:.4f} ± {std_r:.4f}  "
                  f"(n={len(valid)})  "
                  f"action range=[{min(actions_list):.4f}, {max(actions_list):.4f}]  "
                  f"delta range=[{min(deltas_list):.4f}, {max(deltas_list):.4f}]")
        else:
            print(f"  {dim_name}: no valid ratios (actions too small)")

    # Compare with linear fit
    all_action = np.concatenate([actions_x, actions_y, actions_z])
    all_delta = np.concatenate([deltas_x, deltas_y, deltas_z])
    # Compute least squares: delta = a * action (no intercept, because offset=0)
    a = np.dot(all_action, all_delta) / np.dot(all_action, all_action) if np.dot(all_action, all_action) > 1e-12 else 0
    print(f"\n  Linear fit delta = a * action:  a = {a:.4f}")
    print(f"  Original delta_pos_scale:       {delta_pos_scale[0]:.4f}")
    print(f"  Ratio: a / delta_pos_scale =    {a / delta_pos_scale[0]:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str,
        default="/home/yulei/codes/robomimic/robomimic/runs/trained_models/diffusion_policy_can_yq_masked_image/20260506153143/models/model_epoch_140_image_v15_can_mask_success_1.0.pth")
    parser.add_argument("--env", type=str, default="PickPlaceBreadCerealMilkCan")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--pc_depth_obs_key", type=str, default="agentview_depth")
    parser.add_argument("--eef_pos_obs_key", type=str, default="robot0_eef_pos")
    run_calibration(parser.parse_args())
