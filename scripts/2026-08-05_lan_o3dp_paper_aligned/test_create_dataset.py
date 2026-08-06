import importlib.util
from pathlib import Path

import numpy as np
from robosuite.utils import transform_utils as T


MODULE_PATH = Path(__file__).with_name("create_dataset.py")
SPEC = importlib.util.spec_from_file_location("lan_paper_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixed_split_matches_retained_lan_subset():
    names = ["demo_{}".format(index) for index in range(1, 201)]
    selected, train, valid = MODULE.select_demo_split(names)
    assert len(selected) == 40
    assert len(train) == 38
    assert valid == ["demo_18", "demo_164"]


def test_agent_pos_is_world_xyz_plus_site_axis_angle():
    class Group(dict):
        pass

    rotations = np.asarray([[0.1, -0.2, 0.3], [-0.3, 0.2, 0.1]])
    quaternions = np.asarray([T.axisangle2quat(value) for value in rotations])
    group = Group(
        robot0_eef_pos=np.asarray([[0.4, 0.1, 0.9], [0.5, 0.2, 1.0]]),
        robot0_eef_quat_site=quaternions,
    )
    actual = MODULE.build_agent_pos(group)
    assert actual.shape == (2, 6)
    np.testing.assert_allclose(actual[:, :3], group["robot0_eef_pos"])
    np.testing.assert_allclose(actual[:, 3:], rotations, atol=1e-7)


def test_profiles_freeze_camera_and_point_count(tmp_path):
    source = tmp_path / "source.hdf5"
    source.touch()
    names = ["demo_{}".format(index) for index in range(1, 41)]
    selected, train, valid = MODULE.select_demo_split(names)
    paper = MODULE.build_config(
        source,
        "paper256",
        selected,
        train,
        valid,
    )
    official = MODULE.build_config(
        source,
        "official512",
        selected,
        train,
        valid,
    )
    reference = T.axisangle2quat(np.array([0.0, 0.0, np.pi - 0.1]))
    corrected = MODULE.build_config(
        source,
        "paper256_hires_continuous",
        selected,
        train,
        valid,
        reference_quaternion=reference,
    )
    assert paper["pointcloud"]["height"] == 120
    assert paper["pointcloud"]["width"] == 120
    assert paper["pointcloud"]["num_points"] == 256
    assert paper["pointcloud"]["padding_mode"] == "repeat"
    assert official["pointcloud"]["num_points"] == 512
    assert official["pointcloud"]["padding_mode"] == "zero"
    assert corrected["pointcloud"]["height"] == 256
    assert corrected["pointcloud"]["width"] == 256
    assert corrected["pointcloud"]["num_points"] == 256
    assert corrected["rotation_vector"]["mode"] == "continuous"


def test_continuous_agent_and_action_cross_pi_without_a_jump():
    class Group(dict):
        pass

    first = T.axisangle2quat(np.array([0.0, 0.0, np.pi - 0.01]))
    second = -T.axisangle2quat(np.array([0.0, 0.0, np.pi + 0.01]))
    rotation_config = {
        "mode": "continuous",
        "reference_quaternion_xyzw": first.tolist(),
    }
    obs = Group(
        robot0_eef_pos=np.zeros((2, 3)),
        robot0_eef_quat_site=np.stack((first, second)),
    )
    agent = MODULE.build_agent_pos(obs, rotation_config)
    assert np.linalg.norm(agent[1, 3:6] - agent[0, 3:6]) < 0.03

    demo = Group(
        **{
            "next_obs/robot0_eef_pos": np.zeros((2, 3)),
            "next_obs/robot0_eef_quat_site": np.stack((first, second)),
        },
        actions=np.column_stack((np.zeros((2, 6)), -np.ones(2))),
    )
    action = MODULE.build_absolute_action(demo, rotation_config)
    assert np.linalg.norm(action[1, 3:6] - action[0, 3:6]) < 0.03
