from types import SimpleNamespace
import numpy as np
from runtime import LocalPlanner


class FakeGeometry:
    def __init__(self, clearance=1.0, route=None):
        self.clearance, self.route, self.bypass_calls = clearance, route, 0

    def chunk_clearance(self, p, r, actions):
        return self.clearance

    def bypass(self, *args):
        self.bypass_calls += 1
        return self.route


def planner(geometry, grasp=True):
    p = LocalPlanner.__new__(LocalPlanner)
    p.raw = SimpleNamespace(
        _check_grasp=lambda *args: grasp,
        robots=[SimpleNamespace(gripper=None)],
        can=None,
        sim=SimpleNamespace(
            data=SimpleNamespace(body_xpos=np.array([[0.0, 0.0, 0.95]]))
        ),
        can_body_id=0,
        goal_xy=np.array([0.3, 0]),
        table_full_size=np.array([0.8, 0.8, 0.05]),
    )
    p.mode, p.triggers, p.index = "source", 0, 1
    p.pose = lambda: (np.array([-0.2, 0.0, 0.95]), np.eye(3))
    p.geometry = lambda *args: geometry
    return p


def source():
    return np.tile([-0.19, 0.0, 0.95, 0.0, 0.0, 0.0, 1.0], (8, 1))


def test_clear_source_exact_passthrough():
    g = FakeGeometry()
    p = planner(g)
    s = source()
    out, info = p.control(s)
    assert out is s and not info["changed"] and g.bypass_calls == 0


def test_no_grasp_does_not_override_source():
    p = planner(FakeGeometry(-1.0), grasp=False)
    s = source()
    assert p.control(s)[0] is s


def test_infeasible_trigger_holds_and_does_not_resample_route():
    g = FakeGeometry(-1.0)
    p = planner(g)
    out, info = p.control(source())
    assert p.mode == "blocked" and info["trigger"]
    assert np.all(out[:, :3] == p.pose()[0]) and np.all(out[:, 6] == 1)
    p.control(source())
    assert g.bypass_calls == 1


def test_active_route_latches_height_rotation_and_side():
    g = FakeGeometry(1.0)
    p = planner(g)
    p.mode = "bypass"
    p.fixed_rotation = np.eye(3)
    p.route = {
        "points": np.array([[-0.2, 0, 0.95], [-0.2, 0.15, 0.95], [0.15, 0.15, 0.95]]),
        "side": 1,
        "forward": np.array([1.0, 0, 0]),
    }
    s = source()
    s[:, 2] = 0.8
    s[:, 6] = -1
    out, info = p.control(s)
    assert g.bypass_calls == 0 and info["route_side"] == 1
    assert np.all(out[:, 2] == 0.95) and np.all(out[:, 6] == 1)
    assert np.allclose(
        np.linalg.norm(np.diff(np.vstack([p.pose()[0], out[:, :3]]), axis=0), axis=1),
        0.006,
    )


def test_route_completion_does_not_handoff_into_risk():
    p = planner(FakeGeometry(-1.0))
    p.mode = "bypass"
    p.fixed_rotation = np.eye(3)
    p.route = {
        "points": np.array([p.pose()[0], p.pose()[0]]),
        "side": 1,
        "forward": np.array([1.0, 0, 0]),
    }
    out, info = p.control(source())
    assert info["reason"] == "route_complete_wait_for_source" and p.mode == "bypass"
    p.geometry = lambda *args: FakeGeometry(1.0)
    s = source()
    assert p.control(s)[0] is s and p.mode == "handed_back"
