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
    assert paper["pointcloud"]["height"] == 120
    assert paper["pointcloud"]["width"] == 120
    assert paper["pointcloud"]["num_points"] == 256
    assert paper["pointcloud"]["padding_mode"] == "repeat"
    assert official["pointcloud"]["num_points"] == 512
    assert official["pointcloud"]["padding_mode"] == "zero"
