"""
Offline gradient sanity check for the OSC forward model.

Run from repo root:
    uv run python scripts/check_osc_forward_model_grad.py
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from osc_forward_model import load_osc_forward_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15.hdf5"
DEFAULT_MODEL = REPO_ROOT / "outputs/forward_model/osc_eef_forward_image_v15/model.pth"


def decode_demo_names(raw_names):
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names]


def load_sample(dataset, split="valid", demo_index=0, step=50, horizon=16):
    with h5py.File(dataset, "r") as f:
        demo_names = decode_demo_names(f["mask"][split][:])
        demo_name = demo_names[demo_index]
        grp = f[f"data/{demo_name}"]
        max_start = grp["actions"].shape[0] - horizon
        if max_start < 0:
            raise ValueError(f"{demo_name} too short for horizon {horizon}")
        step = min(step, max_start)
        state = np.concatenate(
            [
                grp["obs/robot0_eef_pos"][step],
                grp["obs/robot0_eef_quat"][step],
                grp["obs/robot0_gripper_qpos"][step],
            ],
            axis=0,
        ).astype(np.float32)
        actions = grp["actions"][step : step + horizon].astype(np.float32)
        target_abs = grp["next_obs/robot0_eef_pos"][step : step + horizon].astype(np.float32)
    return demo_name, step, state, actions, target_abs


def obstacle_cost_xy(abs_traj, obstacle_xy, radius):
    diff = abs_traj[..., :2] - obstacle_xy.reshape(1, 1, 2)
    dist = torch.linalg.norm(diff, dim=-1)
    penetration = torch.relu(radius - dist)
    return torch.mean(penetration**2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--radius", type=float, default=0.05)
    args = parser.parse_args()

    loaded = load_osc_forward_model(args.model)
    horizon = loaded.horizon
    demo_name, step, state_np, actions_np, target_abs_np = load_sample(
        dataset=args.dataset,
        split=args.split,
        demo_index=args.demo_index,
        step=args.step,
        horizon=horizon,
    )

    state = torch.from_numpy(state_np[None]).to(loaded.device)
    actions = torch.from_numpy(actions_np[None]).to(loaded.device).requires_grad_(True)
    target_abs = torch.from_numpy(target_abs_np[None]).to(loaded.device)

    pred_abs = loaded.predict_abs_traj(state=state, action_chunk=actions)
    pred_err_cm = torch.linalg.norm(pred_abs - target_abs, dim=-1).mean().item() * 100.0
    terminal_err_cm = torch.linalg.norm(pred_abs[:, -1] - target_abs[:, -1], dim=-1).mean().item() * 100.0

    # Put a synthetic obstacle on the middle predicted waypoint so the cost is positive.
    obstacle_xy = pred_abs[:, horizon // 2, :2].detach()
    cost = obstacle_cost_xy(pred_abs, obstacle_xy=obstacle_xy, radius=float(args.radius))
    cost.backward()

    grad = actions.grad.detach()
    grad_norm = torch.linalg.norm(grad).item()
    grad_xyz_norm = torch.linalg.norm(grad[..., :3]).item()
    grad_rot_grip_norm = torch.linalg.norm(grad[..., 3:]).item()

    print(f"model: {args.model}")
    print(f"dataset: {args.dataset}")
    print(f"sample: split={args.split} demo={demo_name} step={step} horizon={horizon}")
    print(f"pred mean error: {pred_err_cm:.4f} cm")
    print(f"pred terminal error: {terminal_err_cm:.4f} cm")
    print(f"synthetic obstacle cost: {cost.item():.8f}")
    print(f"action grad shape: {tuple(grad.shape)}")
    print(f"action grad norm: {grad_norm:.8f}")
    print(f"action xyz grad norm: {grad_xyz_norm:.8f}")
    print(f"action rot+grip grad norm: {grad_rot_grip_norm:.8f}")
    if not np.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("Gradient sanity check failed: non-positive or non-finite action gradient")


if __name__ == "__main__":
    main()
