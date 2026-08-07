from empty2d_guidance.metrics import exact_mcnemar_p, paired_summary


def row(seed, success, collision):
    return {
        "policy_seed": seed,
        "success": success,
        "collision_any": collision,
        "minimum_clearance": 0.1,
        "trajectory_length": 1.0,
        "clip_rate": 0.0,
    }


def test_exact_mcnemar_and_paired_transitions():
    baseline = [row(index, index >= 12, index < 12) for index in range(20)]
    guided = [row(index, True, False) for index in range(20)]
    result = paired_summary(baseline, guided)
    assert result["failure_to_success"] == 12
    assert result["success_to_failure"] == 0
    assert result["net_rescues"] == 12
    assert result["exact_mcnemar_p"] == exact_mcnemar_p(12, 0)
    assert result["exact_mcnemar_p"] < 0.05
