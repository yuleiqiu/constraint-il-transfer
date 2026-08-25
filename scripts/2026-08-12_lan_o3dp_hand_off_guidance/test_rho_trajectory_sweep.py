import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("rho_trajectory_sweep.py")
SPEC = importlib.util.spec_from_file_location("rho_trajectory_sweep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SUMMARY_PATH = Path(__file__).with_name("summarize_rho_sweep.py")
SUMMARY_SPEC = importlib.util.spec_from_file_location("summarize_rho_sweep", SUMMARY_PATH)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY)


def test_trajectory_separation_and_sustained_thresholds():
    baseline = np.zeros((7, 3), dtype=np.float64)
    guided = np.zeros((7, 3), dtype=np.float64)
    guided[:, 0] = [0.0, 0.004, 0.011, 0.012, 0.021, 0.022, 0.0]
    guided[:, 2] = 0.003

    result = MODULE.trajectory_separation(baseline, guided)

    assert result["common_samples"] == 7
    assert result["max_xy_separation_m"] == 0.022
    assert result["max_z_separation_m"] == 0.003
    assert result["samples_xy_ge_10mm"] == 4
    assert result["longest_run_xy_ge_10mm"] == 4
    assert result["first_step_xy_ge_10mm"] == 2
    assert result["longest_run_xy_ge_20mm"] == 2


def test_separation_uses_only_common_executed_prefix():
    baseline = np.zeros((5, 3), dtype=np.float64)
    guided = np.ones((3, 3), dtype=np.float64) * 0.01
    result = MODULE.trajectory_separation(baseline, guided)
    assert result["common_samples"] == 3
    assert result["common_executed_steps"] == 2


def test_causal_window_uses_activation_and_earliest_collision():
    baseline = np.zeros((8, 3), dtype=np.float64)
    guided = np.zeros((8, 3), dtype=np.float64)
    guided[:, 0] = [0.0, 0.0, 0.001, 0.002, 0.003, 0.02, 0.03, 0.04]
    row = {
        "baseline_outcome": {"first_collision_step": 4},
        "outcome": {
            "first_guidance_active_step": 2,
            "first_collision_step": 6,
        },
    }
    SUMMARY.add_causal_window(row, baseline, guided)
    causal = row["causal_window"]
    assert causal["earliest_collision_step_either_trajectory"] == 4
    assert causal["max_xy_through_earliest_collision_m"] == 0.003
    assert causal["max_xy_first_64_steps_after_activation_m"] == 0.04
