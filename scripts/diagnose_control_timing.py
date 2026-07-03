"""Diagnose action / EEF trajectory timing in robosuite datasets.

This script answers:
- how long one env.step(action) holds an OSC action,
- how often EEF poses are recorded in the dataset,
- how many MuJoCo substeps happen inside one env.step(action), and
- how many intermediate EEF poses exist if we instrument internal substeps.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite.macros as macros
from robomimic.envs.env_robosuite import EnvRobosuite


def load_env_meta(dataset_path):
    with h5py.File(dataset_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demo_key = sorted(f["data"].keys())[0]
        demo = f["data"][demo_key]
        num_samples = int(demo.attrs["num_samples"])
        action_shape = demo["actions"].shape
        eef_shape = demo["obs"]["robot0_eef_pos"].shape
        next_eef_shape = demo["next_obs"]["robot0_eef_pos"].shape
    return env_args, demo_key, num_samples, action_shape, eef_shape, next_eef_shape


def make_env(dataset_path):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {
            "obs": {
                "low_dim": ["robot0_eef_pos"],
                "rgb": [],
                "depth": [],
                "scan": [],
            }
        }
    )
    return EnvRobosuite(
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        **env_meta["env_kwargs"],
    )


def get_eef_pos(env):
    obs = env.get_observation()
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)


def instrument_one_step(env, action):
    """Patch sim step functions to record internal EEF poses for one env.step."""
    raw_env = env.env
    sim = raw_env.sim
    records = []

    original_step1 = sim.step1
    original_step2 = sim.step2
    original_step = sim.step

    def record(label):
        records.append(
            {
                "label": label,
                "time": float(sim.data.time),
                "eef_pos": get_eef_pos(env).tolist(),
            }
        )

    def wrapped_step1(*args, **kwargs):
        out = original_step1(*args, **kwargs)
        record("after_step1")
        return out

    def wrapped_step2(*args, **kwargs):
        out = original_step2(*args, **kwargs)
        record("after_step2")
        return out

    def wrapped_step(*args, **kwargs):
        out = original_step(*args, **kwargs)
        record("after_step")
        return out

    sim.step1 = wrapped_step1
    sim.step2 = wrapped_step2
    sim.step = wrapped_step
    try:
        before_time = float(sim.data.time)
        before_eef = get_eef_pos(env)
        env.step(action)
        after_time = float(sim.data.time)
        after_eef = get_eef_pos(env)
    finally:
        sim.step1 = original_step1
        sim.step2 = original_step2
        sim.step = original_step

    return {
        "before_time": before_time,
        "after_time": after_time,
        "elapsed_time": after_time - before_time,
        "before_eef_pos": before_eef.tolist(),
        "after_eef_pos": after_eef.tolist(),
        "num_internal_records": len(records),
        "internal_records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="third_party/robomimic/datasets/can/yq/image_v15.hdf5",
    )
    parser.add_argument("--output", default="outputs/control_timing/control_timing.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    env_args, demo_key, num_samples, action_shape, eef_shape, next_eef_shape = load_env_meta(dataset_path)

    env = make_env(str(dataset_path))
    env.reset()
    raw_env = env.env
    action = np.zeros(env.action_dimension, dtype=np.float32)
    step_info = instrument_one_step(env, action)

    control_freq = float(raw_env.control_freq)
    control_timestep = float(raw_env.control_timestep)
    model_timestep = float(raw_env.model_timestep)
    expected_substeps = int(control_timestep / model_timestep)

    result = {
        "dataset": str(dataset_path),
        "env_name": env_args["env_name"],
        "env_control_freq_from_metadata": env_args["env_kwargs"].get("control_freq", None),
        "env_control_freq_runtime_hz": control_freq,
        "control_timestep_seconds": control_timestep,
        "model_timestep_seconds": model_timestep,
        "robosuite_macro_simulation_timestep": float(macros.SIMULATION_TIMESTEP),
        "expected_mujoco_substeps_per_env_step": expected_substeps,
        "dataset_first_demo": demo_key,
        "dataset_first_demo_num_samples": num_samples,
        "dataset_actions_shape": action_shape,
        "dataset_obs_eef_shape": eef_shape,
        "dataset_next_obs_eef_shape": next_eef_shape,
        "dataset_eef_recording_timestep_seconds": control_timestep,
        "external_mapping": {
            "one_action_produces_recorded_next_eef_poses": 1,
            "one_recorded_eef_pose_per_seconds": control_timestep,
            "action_chunk_horizon_16_duration_seconds": 16 * control_timestep,
        },
        "instrumented_one_env_step": step_info,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
