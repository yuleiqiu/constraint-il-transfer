import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


SCRIPT = Path(__file__).with_name("create_lan40_subset.py")
SPEC = importlib.util.spec_from_file_location("create_lan40_subset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestCreateLan40Subset(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.trajectory_source = root / "pointcloud.hdf5"
        self.pose_source = root / "pose.hdf5"
        self.dest = root / "lan40.hdf5"
        self.manifest = root / "lan40.manifest.json"

        with h5py.File(self.trajectory_source, "w") as output:
            data = output.create_group("data")
            data.attrs["total"] = 600
            for index in range(1, 201):
                demo = data.create_group("demo_{}".format(index))
                demo.attrs["num_samples"] = 3
                demo.create_dataset(
                    "states",
                    data=np.full((3, 4), index, dtype=np.float32),
                    compression="gzip",
                )
                obs = demo.create_group("obs")
                obs.create_dataset(
                    "task_pointcloud",
                    data=np.full((3, 8, 3), index, dtype=np.float32),
                    compression="gzip",
                )

        with h5py.File(self.pose_source, "w") as output:
            data = output.create_group("data")
            for index in range(1, 201):
                demo = data.create_group("demo_{}".format(index))
                obs = demo.create_group("obs")
                object_state = np.zeros((1, 14), dtype=np.float64)
                object_state[0, 7:10] = [index / 1000.0, -index / 2000.0, 0.86]
                object_state[0, 13] = 1.0
                obs.create_dataset("object", data=object_state)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_selection_copy_and_source_integrity(self):
        before_trajectory = MODULE.file_fingerprint(self.trajectory_source)
        before_pose = MODULE.file_fingerprint(self.pose_source)
        manifest = MODULE.build_subset(
            self.trajectory_source,
            self.pose_source,
            self.dest,
            self.manifest,
        )
        self.assertEqual(
            manifest["selection"]["valid_2"], ["demo_18", "demo_164"]
        )
        self.assertEqual(len(manifest["selection"]["selected_40"]), 40)
        self.assertEqual(len(manifest["selection"]["train_38"]), 38)
        self.assertTrue(manifest["integrity"]["train_valid_disjoint"])
        self.assertTrue(manifest["integrity"]["train_valid_union_equals_selected"])
        self.assertEqual(
            before_trajectory, MODULE.file_fingerprint(self.trajectory_source)
        )
        self.assertEqual(before_pose, MODULE.file_fingerprint(self.pose_source))

        with h5py.File(self.trajectory_source, "r") as source, h5py.File(
            self.dest, "r"
        ) as output:
            selected = MODULE.numeric_demo_sort(output["data"].keys())
            self.assertEqual(selected, manifest["selection"]["selected_40"])
            for demo_name in selected:
                np.testing.assert_array_equal(
                    source["data/{}/states".format(demo_name)][:],
                    output["data/{}/states".format(demo_name)][:],
                )
            train = {item.decode() for item in output["mask/train"][:]}
            valid = {item.decode() for item in output["mask/valid"][:]}
            self.assertFalse(train.intersection(valid))
            self.assertEqual(train.union(valid), set(selected))


if __name__ == "__main__":
    unittest.main()
