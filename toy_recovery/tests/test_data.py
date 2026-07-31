from __future__ import annotations

import numpy as np
import json
import torch

from toy_recovery.data import (
    EvaluationRecord,
    TrajectoryWindowDataset,
    denormalize_actions,
    generate_oracle_demonstrations,
    iter_trajectories,
    normalize_actions,
    replay_trajectory,
    read_evaluation_records,
    write_evaluation_records,
    write_trajectories,
)


def test_hdf5_round_trip_replay(tmp_path) -> None:
    trajectories = generate_oracle_demonstrations(2, seed_start=70)
    path = tmp_path / "trajectories.hdf5"
    write_trajectories(path, trajectories)
    loaded = list(iter_trajectories(path))
    assert len(loaded) == 2
    assert all(replay_trajectory(trajectory) for trajectory in loaded)
    dataset = TrajectoryWindowDataset([path])
    item = dataset[0]
    assert item["rgb"].shape == (2, 3, 64, 64)
    assert item["action"].shape == (16, 2)
    assert item["valid"].shape == (16,)
    assert not item["valid"][0]
    assert torch.equal(item["action"][0], item["action"][1])
    final = dataset[len(dataset) - 1]
    assert final["valid"].sum().item() == 2
    assert torch.all(final["action"][1:] == final["action"][1])


def test_action_normalization_round_trip_and_clipping() -> None:
    actions = np.asarray(((0.025, -0.05), (0.1, -0.1)), dtype=np.float32)
    normalized = normalize_actions(actions)
    assert np.all(np.abs(normalized) <= 1.0)
    reconstructed = denormalize_actions(normalized)
    assert np.allclose(reconstructed[0], actions[0])
    assert np.allclose(reconstructed[1], (0.05, -0.05))


def test_d0_clearance_serializes_as_json_null(tmp_path) -> None:
    record = EvaluationRecord(
        method="source",
        difficulty=0,
        layout_seed=1,
        layout_id="one",
        policy_seed=0,
        success=True,
        collision=False,
        timeout=False,
        steps=10,
        path_length=0.5,
        minimum_clearance=None,
        guidance_activations=0,
    )
    path = tmp_path / "evaluation.jsonl"
    write_evaluation_records(path, [record])
    raw = json.loads(path.read_text())
    assert raw["minimum_clearance"] is None
    assert read_evaluation_records(path)[0].minimum_clearance is None
