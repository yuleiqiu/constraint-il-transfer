import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("paired_evaluation.py")


def load_module():
    spec = importlib.util.spec_from_file_location("empty2d_paired_evaluation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_row(environment_name, policy_seed, guidance_scale, safety_distance, **_):
    return {
        "environment": environment_name,
        "policy_seed": policy_seed,
        "guidance_scale": guidance_scale,
        "safety_distance": safety_distance,
        "success": True,
        "collision_any": False,
        "safe_success": True,
        "ncr": False,
        "steps": 2,
        "minimum_clearance": 0.1,
        "trigger_rate": 0.0,
        "reverse_step_trigger_rate": 0.0,
        "trajectory_length": 1.0,
        "clip_rate": 0.0,
    }


def test_calibration_and_held_out_progress_are_resumable(tmp_path, capsys):
    module = load_module()
    module.CALIBRATION_SEEDS = [1, 2]
    module.TEST_SEEDS = [3, 4]
    module.RHO_VALUES = (0.1,)
    module.Q_VALUES = (0.2,)
    calls = []
    module.load_guidable_policy = lambda *_: (object(), {})

    def rollout(*_, **kwargs):
        calls.append(kwargs)
        return fake_row(**kwargs)

    module.rollout = rollout
    checkpoint = tmp_path / "model.pth"
    _, _, selected = module.evaluate_grid(checkpoint, "circle", "cpu", tmp_path)
    assert selected == {"rho": 0.1, "q": 0.2}
    assert len(calls) == 4
    assert "candidate 1/1" in capsys.readouterr().out

    module.evaluate_grid(checkpoint, "circle", "cpu", tmp_path)
    assert len(calls) == 4
    assert "resumed" in capsys.readouterr().out

    module.held_out_test(checkpoint, "circle", selected, "cpu", tmp_path)
    assert len(calls) == 8
    assert "held-out pair 2/2" in capsys.readouterr().out
    module.held_out_test(checkpoint, "circle", selected, "cpu", tmp_path)
    assert len(calls) == 8
