"""Conditional-RBF demonstration generation and robomimic HDF5 export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .environment import (
    GOAL,
    START,
    WORKSPACE_HIGH,
    WORKSPACE_LOW,
    EnvEmpty2DExtraSphere,
    EnvEmpty2DExtraSquare,
)


NUM_TRAJECTORIES = 100
NUM_POINTS = 256
TRAIN_COUNT = 90
LENGTH_SCALE = 1.1
KERNEL_AMPLITUDE = 8.0
SEED = 42
MAX_STEP = 0.03


def conditional_rbf_covariance(
    num_points: int = NUM_POINTS,
    length_scale: float = LENGTH_SCALE,
    amplitude: float = KERNEL_AMPLITUDE,
) -> np.ndarray:
    time = np.linspace(0.0, 1.0, num_points)
    squared_distance = (time[:, None] - time[None, :]) ** 2
    kernel = amplitude * np.exp(-0.5 * squared_distance / length_scale**2)
    endpoints = np.asarray([0, num_points - 1])
    endpoint_kernel = kernel[np.ix_(endpoints, endpoints)]
    conditioned = kernel - kernel[:, endpoints] @ np.linalg.solve(
        endpoint_kernel, kernel[endpoints]
    )
    conditioned = 0.5 * (conditioned + conditioned.T)
    return conditioned + 1e-10 * np.eye(num_points)


def generate_trajectories(
    *,
    seed: int = SEED,
    num_trajectories: int = NUM_TRAJECTORIES,
    max_attempts: int = 100_000,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    covariance = conditional_rbf_covariance()
    cholesky = np.linalg.cholesky(covariance)
    fraction = np.linspace(0.0, 1.0, NUM_POINTS)
    mean = START[None] + fraction[:, None] * (GOAL - START)[None]
    accepted: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < num_trajectories and attempts < max_attempts:
        attempts += 1
        residual = np.column_stack(
            (cholesky @ rng.standard_normal(NUM_POINTS), cholesky @ rng.standard_normal(NUM_POINTS))
        )
        path = mean + residual
        path[0] = START
        path[-1] = GOAL
        if np.any(path < WORKSPACE_LOW) or np.any(path > WORKSPACE_HIGH):
            continue
        if np.linalg.norm(np.diff(path, axis=0), axis=1).max() > MAX_STEP:
            continue
        accepted.append(path.astype(np.float32))
    if len(accepted) != num_trajectories:
        raise RuntimeError(
            f"Accepted {len(accepted)}/{num_trajectories} trajectories after {attempts} attempts"
        )
    return np.stack(accepted), attempts


def _side_label(path: np.ndarray) -> str:
    central = path[np.abs(path[:, 0]) <= 0.35, 1]
    value = float(np.mean(central if len(central) else path[:, 1]))
    return "upper" if value > 0 else "lower"


def support_statistics(trajectories: np.ndarray) -> dict:
    train = trajectories[:TRAIN_COUNT]
    result = {}
    for name, environment in (
        ("circle", EnvEmpty2DExtraSphere()),
        ("square", EnvEmpty2DExtraSquare()),
    ):
        safe_mask = np.asarray([not environment.trajectory_collides(path) for path in train])
        safe_paths = train[safe_mask]
        upper = sum(_side_label(path) == "upper" for path in safe_paths)
        result[name] = {
            "safe": int(safe_mask.sum()),
            "collision": int((~safe_mask).sum()),
            "upper_safe": int(upper),
            "lower_safe": int(len(safe_paths) - upper),
        }
    return result


def validate_trajectories(trajectories: np.ndarray) -> dict:
    if trajectories.shape != (NUM_TRAJECTORIES, NUM_POINTS, 2):
        raise ValueError(f"Unexpected trajectory shape {trajectories.shape}")
    endpoint_error = max(
        float(np.abs(trajectories[:, 0] - START).max()),
        float(np.abs(trajectories[:, -1] - GOAL).max()),
    )
    workspace_violation = float(
        max(
            np.maximum(WORKSPACE_LOW - trajectories, 0).max(),
            np.maximum(trajectories - WORKSPACE_HIGH, 0).max(),
        )
    )
    maximum_step = float(np.linalg.norm(np.diff(trajectories, axis=1), axis=-1).max())
    support = support_statistics(trajectories)
    gates = {
        "endpoint_exact": endpoint_error == 0.0,
        "workspace": workspace_violation == 0.0,
        "max_step": maximum_step <= MAX_STEP + 1e-7,
    }
    for geometry, statistics in support.items():
        gates[f"{geometry}_safe_20"] = statistics["safe"] >= 20
        gates[f"{geometry}_upper_8"] = statistics["upper_safe"] >= 8
        gates[f"{geometry}_lower_8"] = statistics["lower_safe"] >= 8
        gates[f"{geometry}_collision_40"] = statistics["collision"] >= 40
    return {
        "endpoint_max_abs_error": endpoint_error,
        "workspace_max_violation": workspace_violation,
        "maximum_adjacent_step": maximum_step,
        "support": support,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _env_args() -> str:
    return json.dumps(
        {
            "env_name": "EnvEmpty2D",
            "type": 1,
            "env_version": "empty2d-guidance-v1",
            "env_kwargs": {
                "workspace": [[-1.0, -1.0], [1.0, 1.0]],
                "start": START.tolist(),
                "goal": GOAL.tolist(),
                "max_step": MAX_STEP,
                "robot_radius": 0.02,
                "goal_tolerance": 0.03,
                "horizon": 320,
            },
        },
        sort_keys=True,
    )


def write_hdf5(path: Path, trajectories: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        data.attrs["env_args"] = _env_args()
        data.attrs["total"] = int(np.prod(trajectories.shape[:2]))
        for index, trajectory in enumerate(trajectories):
            demo = data.create_group(f"demo_{index}")
            actions = np.concatenate((trajectory[1:], GOAL[None].astype(np.float32)), axis=0)
            rewards = np.zeros(NUM_POINTS, dtype=np.float32)
            rewards[-1] = 1.0
            dones = np.zeros(NUM_POINTS, dtype=np.int64)
            dones[-1] = 1
            demo.create_dataset("actions", data=actions, compression="gzip")
            demo.create_dataset("states", data=trajectory, compression="gzip")
            demo.create_dataset("rewards", data=rewards, compression="gzip")
            demo.create_dataset("dones", data=dones, compression="gzip")
            obs = demo.create_group("obs")
            obs.create_dataset("agent_pos", data=trajectory, compression="gzip")
            demo.attrs["num_samples"] = NUM_POINTS
        mask = output.create_group("mask")
        mask.create_dataset("train", data=np.asarray([f"demo_{i}" for i in range(TRAIN_COUNT)], dtype="S"))
        mask.create_dataset("valid", data=np.asarray([f"demo_{i}" for i in range(TRAIN_COUNT, NUM_TRAJECTORIES)], dtype="S"))


def validate_hdf5(path: Path) -> dict:
    gates = {
        "demo_count": True,
        "split_90_10_disjoint": True,
        "action_shift": True,
        "absorbing_absolute_terminal": True,
        "sequence_shapes": True,
    }
    with h5py.File(path, "r") as source:
        gates["demo_count"] = len(source["data"]) == NUM_TRAJECTORIES
        train = [item.decode() for item in source["mask/train"][:]]
        valid = [item.decode() for item in source["mask/valid"][:]]
        gates["split_90_10_disjoint"] = (
            len(train) == TRAIN_COUNT
            and len(valid) == NUM_TRAJECTORIES - TRAIN_COUNT
            and set(train).isdisjoint(valid)
        )
        for name in train + valid:
            demo = source[f"data/{name}"]
            obs = demo["obs/agent_pos"][:]
            actions = demo["actions"][:]
            gates["sequence_shapes"] &= (
                obs.shape == actions.shape == (NUM_POINTS, 2)
                and demo["states"].shape == (NUM_POINTS, 2)
                and demo["rewards"].shape == demo["dones"].shape == (NUM_POINTS,)
            )
            gates["action_shift"] &= np.array_equal(actions[:-1], obs[1:])
            gates["absorbing_absolute_terminal"] &= np.array_equal(
                actions[-1], GOAL.astype(np.float32)
            )
    return {"gates": gates, "passed": all(gates.values())}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_support_plot(path: Path, trajectories: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for axis, (name, environment) in zip(
        axes,
        (("Circle", EnvEmpty2DExtraSphere()), ("Square", EnvEmpty2DExtraSquare())),
    ):
        for trajectory in trajectories[:TRAIN_COUNT]:
            safe = not environment.trajectory_collides(trajectory)
            axis.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue" if safe else "tab:red", alpha=0.25)
        if environment.obstacle_kind == "circle":
            axis.add_patch(Circle((0, 0), 0.3, color="black", alpha=0.25))
        else:
            axis.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, color="black", alpha=0.25))
        axis.set(title=name, xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
