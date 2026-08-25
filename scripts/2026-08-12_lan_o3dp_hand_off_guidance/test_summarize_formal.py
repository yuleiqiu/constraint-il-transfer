import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize_formal.py")
SPEC = importlib.util.spec_from_file_location("hand_off_summary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(success, collision):
    return {
        "success": success,
        "collision_any": collision,
        "safe_success": success and not collision,
    }


def test_exact_mcnemar_and_transition_directions():
    before = {
        0: row(False, True),
        1: row(True, False),
        2: row(True, True),
    }
    after = {
        0: row(True, False),
        1: row(False, True),
        2: row(True, False),
    }
    result = MODULE.transition_summary(before, after)
    assert result["task"]["zero_to_one"] == 1
    assert result["task"]["one_to_zero"] == 1
    assert result["collision"]["zero_to_one"] == 1
    assert result["collision"]["one_to_zero"] == 2
    assert MODULE.exact_mcnemar(0, 0) == 1.0
    assert MODULE.exact_mcnemar(0, 3) == 0.25
