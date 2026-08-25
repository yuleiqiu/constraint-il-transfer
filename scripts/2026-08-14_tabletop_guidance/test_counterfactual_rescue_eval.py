"""CPU-only checks for the counterfactual rescue denominator and numerator."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("counterfactual_rescue_eval.py")
SPEC = importlib.util.spec_from_file_location("counterfactual_rescue_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metrics(*, success=False, collision=False, safe=False):
    return {
        "success": success,
        "safe_success": safe,
        "collision_failure": (not success) and collision,
        "ncr": (not success) and (not collision),
        "arm_obstacle_collision_any": False,
        "gripper_obstacle_collision_any": collision,
        "hand_obstacle_collision_any": collision,
        "finger_obstacle_collision_any": False,
        "target_obstacle_collision_any": False,
    }


def test_only_clean_success_obstacle_collision_failure_is_eligible():
    rows = [
        {"scene": 1, "policy_seed": 10,
         "clean": metrics(success=True),
         "obstacle_baseline": metrics(collision=True),
         "obstacle_guided": metrics(success=True, safe=True)},
        {"scene": 1, "policy_seed": 11,
         "clean": metrics(success=False),
         "obstacle_baseline": metrics(collision=True),
         "obstacle_guided": metrics(success=True, safe=True)},
        {"scene": 2, "policy_seed": 10,
         "clean": metrics(success=True),
         "obstacle_baseline": metrics(success=True, collision=True),
         "obstacle_guided": metrics(success=True, safe=True)},
    ]
    summary = MODULE.summarize(rows)
    assert summary["pairs"] == 3
    assert summary["eligible_pairs"] == 1
    assert summary["guided_safe_success_rescues"] == 1
    assert summary["rescue_rate"] == 1.0


if __name__ == "__main__":
    test_only_clean_success_obstacle_collision_failure_is_eligible()
    print("counterfactual rescue tests passed")
