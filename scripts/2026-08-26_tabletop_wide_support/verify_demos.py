"""Validate wide-support Tabletop data and its frozen coverage contract."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
DIRECT_VERIFY_PATH = SCRIPT_DIR.parent / "2026-08-14_tabletop_guidance" / "verify_demos.py"
COLLECT_PATH = SCRIPT_DIR / "collect_demos.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIRECT_VERIFY = load_module(DIRECT_VERIFY_PATH, "tabletop_direct_verify")
COLLECT = load_module(COLLECT_PATH, "tabletop_wide_collect")


def validate_base(path):
    """Validate the original contract with a robust FPS replay diagnostic.

    A single farthest-point-sampling tie can replace one of 256 otherwise
    matching points. Gate the 99th percentile tightly and retain a bounded
    Hausdorff diagnostic instead of letting one tied sample dominate.
    """

    with h5py.File(path, "r") as source:
        data = source["data"]
        demos = DIRECT_VERIFY.sorted_demo_names(data)
        total = 0
        for name in demos:
            demo = data[name]
            length = int(demo.attrs["num_samples"])
            total += length
            assert bool(demo.attrs["teacher_success"])
            assert demo["states"].shape[0] == length
            assert demo["actions"].shape == (length, 7)
            assert demo[DIRECT_VERIFY.ACTION_KEY].shape == (length, 7)
            assert demo["dones"][-1] == 1
            assert np.all(np.isfinite(demo[DIRECT_VERIFY.ACTION_KEY][:]))
            np.testing.assert_allclose(
                demo[DIRECT_VERIFY.ACTION_KEY][:, :6], demo["next_obs/agent_pos"][:], atol=1e-6
            )
            np.testing.assert_allclose(
                demo[DIRECT_VERIFY.ACTION_KEY][:, 6], demo["actions"][:, 6], atol=1e-6
            )
            assert demo["obs/agent_pos"].shape == (length, 6)
            assert demo["obs/task_pointcloud"].shape == (length, 256, 3)
            assert np.all(np.isfinite(demo["obs/task_pointcloud"][:]))
        assert total == int(data.attrs["total"])
        train = DIRECT_VERIFY.decode_mask(source["mask/train"][:])
        valid = DIRECT_VERIFY.decode_mask(source["mask/valid"][:])
        assert set(train).isdisjoint(valid)
        assert set(train) | set(valid) == set(demos)
        env_meta = json.loads(data.attrs["env_args"])
        first = data[demos[0]]
        replay_state = {"states": first["states"][0], "model": first.attrs["model_file"]}
        expected_agent = first["obs/agent_pos"][0]
        expected_cloud = first["obs/task_pointcloud"][0]
        first_controller_action = first["actions"][0]

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["agent_pos"], "rgb": [], "depth": [], "scan": ["task_pointcloud"]}}
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False
    )
    try:
        replayed = env.reset_to(replay_state)
        agent_error = float(np.max(np.abs(replayed["agent_pos"] - expected_agent)))
        replay_cloud = replayed["task_pointcloud"]
        forward = cKDTree(replay_cloud).query(expected_cloud)[0]
        backward = cKDTree(expected_cloud).query(replay_cloud)[0]
        distances = np.concatenate((forward, backward))
        cloud_hausdorff = float(distances.max())
        cloud_p99 = float(np.quantile(distances, 0.99))
        cloud_outlier_fraction = float(np.mean(distances > 1e-4))
        if agent_error > 1e-5 or cloud_p99 > 1e-4 or cloud_hausdorff > 1e-2:
            raise AssertionError(
                "initial observation replay mismatch "
                f"agent={agent_error} cloud_p99={cloud_p99} cloud_hausdorff={cloud_hausdorff}"
            )
        next_obs, _, _, _ = env.step(np.asarray(first_controller_action, dtype=np.float32))
        assert next_obs["agent_pos"].shape == (6,)
        assert next_obs["task_pointcloud"].shape == (256, 3)
    finally:
        env.env.close()
    return {
        "demos": len(demos),
        "transitions": total,
        "train_demos": len(train),
        "valid_demos": len(valid),
        "agent_replay_max_abs_error": agent_error,
        "pointcloud_replay_p99_set_error": cloud_p99,
        "pointcloud_replay_hausdorff_error": cloud_hausdorff,
        "pointcloud_replay_outlier_fraction_gt_0.1mm": cloud_outlier_fraction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    path = args.dataset.resolve()
    base = validate_base(path)
    with h5py.File(path, "r") as source:
        data = source["data"]
        route_config = json.loads(data.attrs["route_config"])
        train = DIRECT_VERIFY.decode_mask(source["mask/train"][:])
        support = COLLECT.route_support(data, train)
        stored = json.loads(data.attrs["train_route_support"])
        if support != stored:
            raise AssertionError("recomputed route support differs from stored contract")
        if route_config["obstacle_used_for_generation"]:
            raise AssertionError("wide-support source must remain obstacle-free")
        if not support["passed"]:
            raise AssertionError(f"route-support gate failed: {support['gates']}")
    print(json.dumps({"base": base, "route_config": route_config, "train_route_support": support}, indent=2))


if __name__ == "__main__":
    main()
