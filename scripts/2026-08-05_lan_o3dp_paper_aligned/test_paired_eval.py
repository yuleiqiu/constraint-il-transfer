import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("paired_eval.py")
SPEC = importlib.util.spec_from_file_location("lan_paper_paired_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeModel:
    def __init__(self):
        self.names = [
            "robot0_hand_collision",
            "robot0_finger1_collision",
            "robot0_finger1_pad_collision",
            "robot0_finger2_collision",
            "robot0_finger2_pad_collision",
        ]
        self.geom_contype = np.asarray([1, 0, 1, 0, 1], dtype=np.int32)
        self.geom_conaffinity = np.ones(5, dtype=np.int32)

    def geom_name2id(self, name):
        return self.names.index(name)


class FakeGripper:
    contact_geoms = [
        "robot0_hand_collision",
        "robot0_finger1_collision",
        "robot0_finger1_pad_collision",
        "robot0_finger2_collision",
        "robot0_finger2_pad_collision",
    ]


class FakeRethinkGripper:
    contact_geoms = [
        "gripper0_right_gripper_base_col",
        "gripper0_right_l_finger_g0",
        "gripper0_right_r_fingerpad_g0",
    ]


class FakeRobotModel:
    contact_geoms = []


class FakeRobot:
    robot_model = FakeRobotModel()
    gripper = FakeGripper()


class FakeSim:
    def __init__(self):
        self.model = FakeModel()
        self.forward_count = 0

    def forward(self):
        self.forward_count += 1


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.robots = [FakeRobot()]


def test_hand_collision_toggle_preserves_all_finger_masks():
    env = FakeEnv()
    original_contype = env.sim.model.geom_contype.copy()
    original_conaffinity = env.sim.model.geom_conaffinity.copy()

    disabled = MODULE.configure_hand_collision(env, enabled=False)
    assert disabled["enabled"] is False
    assert disabled["hand_masks"] == {"robot0_hand_collision": (0, 0)}
    np.testing.assert_array_equal(
        env.sim.model.geom_contype[1:], original_contype[1:]
    )
    np.testing.assert_array_equal(
        env.sim.model.geom_conaffinity[1:], original_conaffinity[1:]
    )

    enabled = MODULE.configure_hand_collision(env, enabled=True)
    assert enabled["enabled"] is True
    np.testing.assert_array_equal(env.sim.model.geom_contype, original_contype)
    np.testing.assert_array_equal(env.sim.model.geom_conaffinity, original_conaffinity)
    assert env.sim.forward_count == 2


def test_geom_categories_separate_hand_from_fingers():
    env = FakeEnv()
    _, gripper, hand, fingers = MODULE.categorized_robot_geom_ids(env)
    assert set(gripper.values()) == set(FakeGripper.contact_geoms)
    assert set(hand.values()) == {"robot0_hand_collision"}
    assert set(fingers.values()) == set(FakeGripper.contact_geoms[1:])


def test_geom_categories_recognize_rethink_gripper_base_as_hand():
    env = FakeEnv()
    env.robots[0].gripper = {"right": FakeRethinkGripper()}
    env.sim.model.names = list(FakeRethinkGripper.contact_geoms)
    env.sim.model.geom_contype = np.ones(3, dtype=np.int32)
    env.sim.model.geom_conaffinity = np.ones(3, dtype=np.int32)

    _, gripper, hand, fingers = MODULE.categorized_robot_geom_ids(env)
    assert set(gripper.values()) == set(FakeRethinkGripper.contact_geoms)
    assert set(hand.values()) == {"gripper0_right_gripper_base_col"}
    assert set(fingers.values()) == set(FakeRethinkGripper.contact_geoms[1:])
