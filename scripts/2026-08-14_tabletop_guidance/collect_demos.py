"""Collect obstacle-free point-cloud demos from the direct waypoint teacher."""

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_robosuite import EnvRobosuite


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import waypoint_teacher as TEACHER  # noqa: E402


OBS_KEYS = ("agent_pos", "task_pointcloud")
ACTION_KEY = "abs_eef_pose_action"


def initialize_observation_modalities():
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {
            "obs": {
                "low_dim": ["agent_pos"],
                "rgb": [],
                "depth": [],
                "scan": ["task_pointcloud"],
            }
        }
    )


def make_env(seed):
    initialize_observation_modalities()
    return EnvRobosuite(
        env_name="PickPlaceCanTabletop",
        render=False,
        render_offscreen=True,
        use_image_obs=False,
        use_depth_obs=False,
        robots="Panda",
        gripper_types="RethinkGripper",
        controller_configs=TEACHER.absolute_osc_controller_config(),
        control_freq=20,
        horizon=600,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        eef_pose_observation={
            "enabled": True,
            "obs_key": "agent_pos",
            "robot_prefix": "robot0",
            "rotation_vector_mode": "principal",
        },
        target_pointcloud={
            "enabled": True,
            "obs_key": "task_pointcloud",
            "camera_name": "agentview",
            "height": 256,
            "width": 256,
            "target_object": "Can",
            "include_visual_goal": False,
            "goal_objects": ["GoalTray"],
            "num_points": 256,
            "padding_mode": "repeat",
            "max_geom_distance": 0.15,
        },
        seed=seed,
    )


class EpisodeBuffer:
    def __init__(self):
        self.states = []
        self.obs = {key: [] for key in OBS_KEYS}
        self.next_obs = {key: [] for key in OBS_KEYS}
        self.controller_actions = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.success = []
        self.phases = []

    def append(self, *, state_before, obs, next_obs, controller_action, reward, done, success, phase):
        self.states.append(np.asarray(state_before))
        for key in OBS_KEYS:
            self.obs[key].append(np.asarray(obs[key], dtype=np.float32))
            self.next_obs[key].append(np.asarray(next_obs[key], dtype=np.float32))
        controller_action = np.asarray(controller_action, dtype=np.float32)
        self.controller_actions.append(controller_action)
        self.actions.append(
            np.concatenate(
                (
                    np.asarray(next_obs["agent_pos"], dtype=np.float32),
                    controller_action[6:7],
                )
            )
        )
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.success.append(bool(success))
        self.phases.append(phase)

    def __len__(self):
        return len(self.actions)


def write_episode(data, demo_name, buffer, result, model_xml, seed):
    group = data.create_group(demo_name)
    group.create_dataset("states", data=np.asarray(buffer.states), compression="gzip")
    group.create_dataset("actions", data=np.asarray(buffer.controller_actions), compression="gzip")
    group.create_dataset(ACTION_KEY, data=np.asarray(buffer.actions), compression="gzip")
    rewards = np.asarray(buffer.rewards, dtype=np.float32)
    dones = np.asarray(buffer.dones, dtype=np.uint8)
    dones[-1] = 1
    group.create_dataset("rewards", data=rewards, compression="gzip")
    group.create_dataset("dones", data=dones, compression="gzip")
    group.create_dataset("success", data=np.asarray(buffer.success, dtype=np.uint8), compression="gzip")
    group.create_dataset("phases", data=np.asarray(buffer.phases, dtype=h5py.string_dtype("utf-8")))
    obs_group = group.create_group("obs")
    next_obs_group = group.create_group("next_obs")
    for key in OBS_KEYS:
        obs_group.create_dataset(key, data=np.asarray(buffer.obs[key]), compression="gzip")
        next_obs_group.create_dataset(key, data=np.asarray(buffer.next_obs[key]), compression="gzip")
    group.attrs["num_samples"] = len(buffer)
    group.attrs["model_file"] = model_xml
    group.attrs["seed"] = int(seed)
    group.attrs["teacher_success"] = bool(result.success)
    group.attrs["teacher_grasped"] = bool(result.grasped)
    group.attrs["can_start"] = result.can_start
    group.attrs["goal_xy"] = result.goal_xy
    group.attrs["max_transport_line_deviation_m"] = result.max_transport_line_deviation_m


def write_masks(output_file, demo_names, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(demo_names, dtype=object)[rng.permutation(len(demo_names))]
    validation_count = max(1, int(round(len(demo_names) * validation_fraction)))
    validation_count = min(validation_count, len(demo_names) - 1)
    valid = sorted(shuffled[:validation_count].tolist(), key=lambda name: int(name.split("_")[-1]))
    train = sorted(shuffled[validation_count:].tolist(), key=lambda name: int(name.split("_")[-1]))
    mask = output_file.create_group("mask")
    mask.create_dataset("train", data=np.asarray(train, dtype="S"))
    mask.create_dataset("valid", data=np.asarray(valid, dtype="S"))
    return train, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.episodes < 2:
        raise ValueError("--episodes must be at least 2")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    env = make_env(args.seed)
    teacher = TEACHER.DirectPickPlaceTeacher()
    env_meta = env.serialize()
    demo_names = []
    total = 0
    try:
        with h5py.File(output, "w") as output_file:
            data = output_file.create_group("data")
            data.attrs["env_args"] = json.dumps(env_meta, indent=4)
            data.attrs["action_key"] = ACTION_KEY
            data.attrs["action_contract"] = "next achieved absolute world EEF xyz + site axis-angle + gripper command"
            data.attrs["teacher"] = "direct minimum-jerk Cartesian waypoints; no obstacle and no lateral detour"
            for episode in range(args.episodes):
                episode_seed = args.seed + episode
                buffer = EpisodeBuffer()
                result = teacher.rollout(env, seed=episode_seed, transition_callback=buffer.append)
                if not result.success:
                    raise RuntimeError(
                        f"teacher failed at episode {episode} seed {episode_seed}; refusing a partial dataset"
                    )
                demo_name = f"demo_{episode}"
                model_xml = env.env.sim.model.get_xml()
                write_episode(data, demo_name, buffer, result, model_xml=model_xml, seed=episode_seed)
                demo_names.append(demo_name)
                total += len(buffer)
                print(
                    json.dumps(
                        {
                            "demo": demo_name,
                            "seed": episode_seed,
                            "steps": len(buffer),
                            "success": result.success,
                            "line_deviation_m": result.max_transport_line_deviation_m,
                        }
                    ),
                    flush=True,
                )
            data.attrs["total"] = total
            train, valid = write_masks(output_file, demo_names, args.validation_fraction, args.seed + 991)
            data.attrs["train_demos"] = len(train)
            data.attrs["valid_demos"] = len(valid)
    finally:
        env.env.close()
    print(f"wrote={output} demos={len(demo_names)} transitions={total} train={len(train)} valid={len(valid)}")


if __name__ == "__main__":
    main()
