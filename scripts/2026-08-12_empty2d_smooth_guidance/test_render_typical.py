import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("render_typical.py")


def load_module():
    spec = importlib.util.spec_from_file_location("empty2d_render_typical", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(condition, layout_id, *, reversals, steps, backward, safe=True):
    mode, cost = condition.split("_", 1)
    return {
        "condition": condition,
        "layout_id": layout_id,
        "center_mode": mode,
        "obstacle_cost_type": "huber_hinge" if cost == "huber_hinge" else "hinge",
        "x_direction_reversals": reversals,
        "steps": steps,
        "backward_distance": backward,
        "safe_success": safe,
        "success": safe,
        "collision_any": not safe,
    }


def test_selection_uses_frozen_metric_rules():
    module = load_module()
    rows = {
        "centered_hinge": [
            row("centered_hinge", "a", reversals=90, steps=350, backward=0.3),
            row("centered_hinge", "b", reversals=70, steps=360, backward=0.4),
        ],
        "centered_huber_hinge": [
            row("centered_huber_hinge", "a", reversals=30, steps=300, backward=0.2),
            row("centered_huber_hinge", "b", reversals=60, steps=310, backward=0.3),
        ],
        "offset_hinge": [
            row("offset_hinge", "a", reversals=10, steps=250, backward=0.05),
            row("offset_hinge", "b", reversals=12, steps=260, backward=0.06),
        ],
        "offset_huber_hinge": [
            row("offset_huber_hinge", "a", reversals=8, steps=240, backward=0.04),
            row("offset_huber_hinge", "b", reversals=5, steps=200, backward=0.03),
        ],
    }
    cases = module.select_cases(rows)
    assert len(cases) == 2
    assert cases[0]["first"]["layout_id"] == "a"
    assert cases[1]["first"]["layout_id"] == "b"
