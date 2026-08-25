import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from empty2d_guidance.environment import EnvEmpty2DExtraSphere


SCRIPT = Path(__file__).with_name("paired_ablation.py")


def load_module():
    spec = importlib.util.spec_from_file_location("empty2d_smooth_ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_offset_layouts_are_reproducible_mirrored_and_obstructing():
    module = load_module()
    first = module.frozen_offset_layouts(3)
    second = module.frozen_offset_layouts(3)
    assert first == second
    assert len(first) == 6
    for upper, lower in zip(first[::2], first[1::2]):
        assert upper.obstacle_center[0] == lower.obstacle_center[0]
        assert upper.obstacle_center[1] == -lower.obstacle_center[1]
        assert module.OFFSET_X_RANGE[0] <= upper.obstacle_center[0] <= module.OFFSET_X_RANGE[1]
        assert module.OFFSET_ABS_Y_RANGE[0] <= upper.obstacle_center[1] <= module.OFFSET_ABS_Y_RANGE[1]
        for layout in (upper, lower):
            environment = EnvEmpty2DExtraSphere(
                horizon=module.HORIZON,
                obstacle_center=layout.obstacle_center,
            )
            assert environment.segment_collides(
                np.asarray([-0.875, 0.0]), np.asarray([0.875, 0.0])
            )


def test_centered_layouts_preserve_ids_and_policy_seeds():
    module = load_module()
    offset = module.frozen_offset_layouts(2)
    centered = module.layouts_for_mode(offset, "centered")
    assert [row.layout_id for row in centered] == [row.layout_id for row in offset]
    assert [row.policy_seed for row in centered] == [row.policy_seed for row in offset]
    assert all(row.obstacle_center == (0.0, 0.0) for row in centered)
    with pytest.raises(ValueError, match="Unknown center mode"):
        module.layouts_for_mode(offset, "unknown")


def test_frozen_successor_contract_uses_horizon_ladder_and_requested_huber_hinge():
    module = load_module()
    assert module.HORIZON == 600
    assert module.HORIZON_MILESTONES == (400, 500, 600)
    assert module.GUIDANCE_SCALE == 1.0
    assert module.SAFETY_DISTANCE == 0.12
    assert module.HUBER_SMOOTHING == 0.03
    smooth = [
        condition
        for condition in module.CONDITIONS
        if condition.obstacle_cost_type == "huber_hinge"
    ]
    assert len(smooth) == 2
    assert all(
        condition.obstacle_cost_smoothing == module.HUBER_SMOOTHING
        for condition in smooth
    )
    assert {condition.center_mode for condition in module.CONDITIONS} == {
        "centered",
        "offset",
    }


def test_contract_round_trip_is_stable(tmp_path):
    module = load_module()
    checkpoint = tmp_path / "model.pth"
    contract = module.build_contract(
        checkpoint,
        "circle",
        1,
        module.frozen_offset_layouts(1),
    )
    assert contract == module.json.loads(module.json.dumps(contract))
