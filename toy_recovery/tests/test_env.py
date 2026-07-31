from __future__ import annotations

from dataclasses import replace

import numpy as np

from toy_recovery.config import EnvironmentConfig
from toy_recovery.env import (
    LayoutSpec,
    Obstacle,
    PointNavEnv,
    load_layout_manifest,
    make_layout,
    swept_clearance,
    write_layout_manifest,
)
from toy_recovery.planner import (
    d1_mode_path,
    oracle_actions,
    path_to_actions,
    rollout_oracle,
    shortest_path,
)


def test_layout_seed_determinism_and_difficulty() -> None:
    for difficulty in range(3):
        first = make_layout(difficulty, 1234)
        second = make_layout(difficulty, 1234)
        assert first == second
        assert first.identifier == second.identifier
        assert len(first.obstacles) == difficulty


def test_collision_and_success_boundaries() -> None:
    cfg = EnvironmentConfig()
    collision_layout = LayoutSpec(
        difficulty=1,
        seed=1,
        start=(-0.20, 0.0),
        goal=(0.8, 0.0),
        obstacles=(Obstacle(0.0, 0.0, 0.12),),
    )
    env = PointNavEnv(layout=collision_layout, config=cfg)
    env.reset(1, collision_layout)
    _, _, done, info = env.step(np.asarray((0.05, 0.0)))
    assert done
    assert info["collision"]
    assert not info["success"]

    success_layout = LayoutSpec(
        difficulty=0,
        seed=2,
        start=(0.0, 0.0),
        goal=(0.13, 0.0),
    )
    env = PointNavEnv(layout=success_layout, config=cfg)
    env.reset(2, success_layout)
    _, _, done, info = env.step(np.asarray((0.05, 0.0)))
    assert done
    assert info["success"]
    assert not info["collision"]


def test_clone_restore_is_exact() -> None:
    layout = make_layout(2, 55)
    env = PointNavEnv(layout=layout)
    env.reset(55, layout)
    action = oracle_actions(layout)[0]
    env.step(action)
    cloned = env.clone_state()
    observation = env.observation()
    env.step(oracle_actions(layout)[1])
    restored = env.restore_state(cloned)
    assert np.array_equal(restored["rgb"], observation["rgb"])
    assert np.array_equal(restored["state"], observation["state"])
    assert env.clone_state().path_length == cloned.path_length


def test_planner_obstacle_permutation_invariance() -> None:
    layout = make_layout(2, 81)
    permuted = replace(layout, obstacles=tuple(reversed(layout.obstacles)))
    first = shortest_path(layout)
    second = shortest_path(permuted)
    assert first is not None and second is not None
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.allclose(a, b)


def test_oracle_feasibility_sample() -> None:
    for difficulty in range(3):
        for seed in range(10):
            layout = make_layout(difficulty, 10_000 + seed)
            success, collision, steps = rollout_oracle(layout)
            assert success
            assert not collision
            assert steps <= 80


def test_d1_has_two_modes_and_d2_has_sustained_structure() -> None:
    d1 = make_layout(1, 18_001)
    assert d1_mode_path(d1, mode="upper") is not None
    assert d1_mode_path(d1, mode="lower") is not None
    assert rollout_oracle(d1, mode="upper")[0]
    assert rollout_oracle(d1, mode="lower")[0]

    d2 = make_layout(2, 18_002)
    assert all(
        swept_clearance(
            np.asarray(d2.start), np.asarray(d2.goal), (obstacle,), 0.04
        )
        <= 0.0
        for obstacle in d2.obstacles
    )
    path = shortest_path(d2)
    assert path is not None and len(path) >= 4
    actions = path_to_actions(path, 0.05, ease=False)
    positions = np.concatenate(
        (
            np.asarray(d2.start)[None],
            np.asarray(d2.start)[None] + np.cumsum(actions, axis=0),
        )
    )
    closest = [
        int(np.argmin(np.linalg.norm(positions - obstacle.center, axis=1)))
        for obstacle in d2.obstacles
    ]
    assert max(closest) - min(closest) >= 8


def test_d0_manifest_splits_are_hashed(tmp_path) -> None:
    layout = make_layout(0, 19_000)
    env = PointNavEnv(layout=layout)
    env.reset(layout.seed, layout)
    path = tmp_path / "manifest.json"
    write_layout_manifest(
        path,
        split_seeds={
            "calibration": {0: [1], 1: [2], 2: [3]},
            "scale_tuning": {0: [4], 1: [5], 2: [6]},
            "screen": {0: [7], 1: [8], 2: [9]},
            "test": {0: [10], 1: [11], 2: [12]},
        },
    )
    manifest = load_layout_manifest(path)
    assert set(manifest["splits"]) == {
        "calibration",
        "scale_tuning",
        "screen",
        "test",
    }
