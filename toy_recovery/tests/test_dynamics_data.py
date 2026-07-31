from __future__ import annotations

import h5py
import numpy as np
import torch

from toy_recovery.config import PolicyConfig
from toy_recovery.data import denormalize_actions
from toy_recovery.env import PointNavEnv, make_layout
from toy_recovery.rollout import (
    _rollout_with_open_loop_queries,
    load_dynamics_batches,
)


class ColorMeanEncoder(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=(2, 3))


class FixedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = PolicyConfig()

    def sample(self, rgb: torch.Tensor, **_: object) -> torch.Tensor:
        actions = torch.zeros(
            rgb.shape[0], self.config.prediction_horizon, 2, device=rgb.device
        )
        actions[..., 0] = 0.8
        return actions


def test_dynamics_query_executes_the_same_predicted_chunk_open_loop() -> None:
    layout = make_layout(0, 22_000)
    trajectory, queries = _rollout_with_open_loop_queries(
        policy=FixedPolicy(),
        layout=layout,
        rollout_index=0,
        checkpoint_index=0,
        policy_seed=0,
        config=PointNavEnv().config,
        device="cpu",
    )
    assert trajectory.action.shape[0] == 80
    first = queries[0]
    assert first["normalized_action_chunk"].shape == (16, 2)
    env = PointNavEnv(layout=layout)
    env.reset(layout.seed, layout)
    done = False
    for action in denormalize_actions(first["normalized_action_chunk"]):
        if done:
            break
        _, _, done, _ = env.step(action)
    assert np.array_equal(env.render(), first["future_rgb"])


def test_dynamics_split_is_rollout_level_and_stratified(tmp_path) -> None:
    path = tmp_path / "dynamics.hdf5"
    rollout_ids = []
    difficulties = []
    checkpoints = []
    for difficulty in range(3):
        for checkpoint in range(5):
            for repeat in range(2):
                rollout_ids.append(difficulty * 10 + checkpoint * 2 + repeat)
                difficulties.append(difficulty)
                checkpoints.append(checkpoint)
    count = len(rollout_ids)
    current = np.zeros((count, 64, 64, 3), dtype=np.uint8)
    future = np.zeros_like(current)
    for index in range(count):
        current[index, ..., 0] = index
        future[index, ..., 1] = index
    with h5py.File(path, "w") as handle:
        group = handle.create_group("dynamics_queries")
        group.attrs["schema"] = "open-loop-predicted-chunk-v1"
        group.create_dataset("rollout_index", data=np.asarray(rollout_ids))
        group.create_dataset("difficulty", data=np.asarray(difficulties))
        group.create_dataset("checkpoint_index", data=np.asarray(checkpoints))
        group.create_dataset("current_rgb", data=current)
        group.create_dataset("future_rgb", data=future)
        group.create_dataset(
            "normalized_action_chunk",
            data=np.zeros((count, 16, 2), dtype=np.float32),
        )
    train, validation, manifest = load_dynamics_batches(
        path, ColorMeanEncoder(), batch_size=8, split_seed=3
    )
    assert train and validation
    assert manifest["train_rollouts"] == 15
    assert manifest["validation_rollouts"] == 15
    assert set(manifest["train_rollout_ids"]).isdisjoint(
        manifest["validation_rollout_ids"]
    )
    assert manifest["uses_outcome_labels"] is False
