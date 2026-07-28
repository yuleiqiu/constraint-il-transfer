"""CPU-only checks for fixed-state diagnostic selection helpers."""

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("same_state_diagnostic.py")
SPEC = importlib.util.spec_from_file_location("same_state_diagnostic", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pointcloud_observation_validation():
    valid = {"task_pointcloud": np.ones((2, 512, 3), dtype=np.float32)}
    assert MODULE.validate_policy_observation(valid) == (True, None)

    zero_frame = valid["task_pointcloud"].copy()
    zero_frame[1] = 0.0
    ok, reason = MODULE.validate_policy_observation(
        {"task_pointcloud": zero_frame}
    )
    assert not ok
    assert "all-zero frame" in reason

    nonfinite = valid["task_pointcloud"].copy()
    nonfinite[0, 0, 0] = np.nan
    ok, reason = MODULE.validate_policy_observation(
        {"task_pointcloud": nonfinite}
    )
    assert not ok
    assert "non-finite" in reason


def test_obstacle_contact_must_strictly_precede_target():
    check = MODULE.obstacle_contact_precedes_target
    assert check(
        target_before_prefix=False,
        first_obstacle_contact_offset=2,
        first_target_contact_offset=None,
    )
    assert check(
        target_before_prefix=False,
        first_obstacle_contact_offset=2,
        first_target_contact_offset=3,
    )
    assert not check(
        target_before_prefix=False,
        first_obstacle_contact_offset=3,
        first_target_contact_offset=2,
    )
    assert not check(
        target_before_prefix=False,
        first_obstacle_contact_offset=2,
        first_target_contact_offset=2,
    )
    assert not check(
        target_before_prefix=True,
        first_obstacle_contact_offset=0,
        first_target_contact_offset=None,
    )


if __name__ == "__main__":
    test_pointcloud_observation_validation()
    test_obstacle_contact_must_strictly_precede_target()
    print("same-state diagnostic helper checks passed")
