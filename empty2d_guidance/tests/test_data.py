from pathlib import Path

import h5py
import numpy as np

from empty2d_guidance.data import (
    GOAL,
    generate_trajectories,
    validate_trajectories,
    write_hdf5,
)


def test_fixed_gp_dataset_passes_counterfactual_support_gate():
    trajectories, attempts = generate_trajectories()
    validation = validate_trajectories(trajectories)
    assert trajectories.shape == (100, 256, 2)
    assert attempts == 112
    assert validation["passed"]
    assert validation["support"]["circle"] == {
        "safe": 33,
        "collision": 57,
        "upper_safe": 19,
        "lower_safe": 14,
    }
    assert validation["support"]["square"] == {
        "safe": 20,
        "collision": 70,
        "upper_safe": 11,
        "lower_safe": 9,
    }


def test_hdf5_action_shift_and_absorbing_terminal(tmp_path: Path):
    trajectories, _ = generate_trajectories()
    destination = tmp_path / "data.hdf5"
    write_hdf5(destination, trajectories)
    with h5py.File(destination, "r") as source:
        demo = source["data/demo_0"]
        np.testing.assert_array_equal(demo["actions"][:-1], demo["obs/agent_pos"][1:])
        np.testing.assert_array_equal(demo["actions"][-1], GOAL.astype(np.float32))
        assert len(source["mask/train"]) == 90
        assert len(source["mask/valid"]) == 10
