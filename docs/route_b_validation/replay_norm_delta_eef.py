"""
Replay delta_eef_action through simulator with normalization, compare methods.

Tests three open-loop replay strategies:
  1. raw: delta_eef sent directly as controller action
  2. gaussian: (delta_eef - mean) / std sent as controller action
  3. min_max: (delta_eef - min) / (max-min) * 2 - 1 sent as controller action

Records video for the best-performing method.
"""

import argparse
import json
import os
import h5py
import numpy as np
import imageio

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


def compute_stats(hdf5_data, train_filter_mask):
    all_demos = sorted(hdf5_data["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    training_demos = [d for d, m in zip(all_demos, train_filter_mask) if m]

    all_delta = []
    for demo_name in training_demos:
        all_delta.append(hdf5_data[f"data/{demo_name}/delta_eef_action"][:])
    all_delta = np.concatenate(all_delta, axis=0)

    return dict(
        mean=all_delta.mean(axis=0).astype(np.float64),
        std=all_delta.std(axis=0).astype(np.float64),
        min=all_delta.min(axis=0).astype(np.float64),
        max=all_delta.max(axis=0).astype(np.float64),
    )


def norm_gaussian(action, stats):
    return (action - stats["mean"]) / (stats["std"] + 1e-8)


def norm_minmax(action, stats):
    rng = stats["max"] - stats["min"]
    rng = np.where(rng < 1e-4, 1.0, rng)
    return (action - stats["min"]) / rng * 2.0 - 1.0


def replay_and_measure(env_factory, initial_state, delta_eef, transform_fn):
    env = env_factory()
    obs = env.reset_to(initial_state)
    positions = [obs["robot0_eef_pos"].copy()]
    for t in range(len(delta_eef)):
        action = transform_fn(delta_eef[t])
        obs, _, _, _ = env.step(action)
        positions.append(obs["robot0_eef_pos"].copy())
    return np.array(positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-demos", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs/replay_norm_delta_eef")
    args = parser.parse_args()

    config = config_factory("bc")
    ObsUtils.initialize_obs_utils_with_config(config)

    hdf5_data = h5py.File(args.dataset, "r")
    env_meta = FileUtils.get_env_metadata_from_dataset(args.dataset)
    train_mask = hdf5_data["mask/train"][:]
    stats = compute_stats(hdf5_data, train_mask)

    print("delta_eef stats (training set):")
    for i, name in enumerate(["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip"]):
        print(f"  {name}: mean={stats['mean'][i]:+.4f}  std={stats['std'][i]:.4f}  "
              f"min={stats['min'][i]:+.4f}  max={stats['max'][i]:+.4f}")

    demos = sorted(hdf5_data["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    selected = demos[:min(args.n_demos, len(demos))]

    def make_env():
        return EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False
        )

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {"original_actions": [], "raw": [], "gaussian": [], "minmax": []}
    for demo_name in selected:
        grp = hdf5_data[f"data/{demo_name}"]
        orig_pos = grp["obs/robot0_eef_pos"][:]
        delta_eef = grp["delta_eef_action"][:]
        actions = grp["actions"][:]
        initial_state = dict(states=grp["states"][0])
        initial_state["model"] = grp.attrs["model_file"]
        initial_state["ep_meta"] = grp.attrs.get("ep_meta", None)

        # original OSC actions replay
        pos = replay_and_measure(make_env, initial_state, actions, lambda x: x)
        end_err = np.linalg.norm(pos[-1] - orig_pos[-1])
        mean_err = np.mean(np.abs(pos[:-1] - orig_pos))
        all_results["original_actions"].append((mean_err, end_err))

        for method, transform_fn in [
            ("raw", lambda x: x),
            ("gaussian", lambda x: norm_gaussian(x, stats)),
            ("minmax", lambda x: norm_minmax(x, stats)),
        ]:
            pos = replay_and_measure(make_env, initial_state, delta_eef, transform_fn)
            end_err = np.linalg.norm(pos[-1] - orig_pos[-1])
            mean_err = np.mean(np.abs(pos[:-1] - orig_pos))
            all_results[method].append((mean_err, end_err))

    # Print summary
    print(f"\nReplay error summary ({len(selected)} demos):")
    print(f"{'Method':<12} {'mean_err(m)':<14} {'end_err(m)':<14}")
    print("-" * 40)
    for method in ["original_actions", "raw", "gaussian", "minmax"]:
        means = [r[0] for r in all_results[method]]
        ends = [r[1] for r in all_results[method]]
        print(f"{method:<12} {np.mean(means):.4f} ± {np.std(means):.4f}   "
              f"{np.mean(ends):.4f} ± {np.std(ends):.4f}")

    # Record video with best method
    best_method = min(all_results, key=lambda m: np.mean([r[1] for r in all_results[m]]))
    print(f"\nBest method: {best_method}")
    transform_fn = {
        "original_actions": lambda x: x,
        "raw": lambda x: x,
        "gaussian": lambda x: norm_gaussian(x, stats),
        "minmax": lambda x: norm_minmax(x, stats),
    }[best_method]

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False
    )
    for demo_name in selected[:3]:
        grp = hdf5_data[f"data/{demo_name}"]
        if best_method == "original_actions":
            action_data = grp["actions"][:]
        else:
            action_data = grp["delta_eef_action"][:]
        initial_state = dict(states=grp["states"][0])
        initial_state["model"] = grp.attrs["model_file"]
        initial_state["ep_meta"] = grp.attrs.get("ep_meta", None)

        env.reset_to(initial_state)
        frames = []
        for t in range(len(action_data)):
            frames.append(env.render(mode="rgb_array", height=256, width=256, camera_name="agentview"))
            env.step(transform_fn(action_data[t]))
        frames.append(env.render(mode="rgb_array", height=256, width=256, camera_name="agentview"))

        video_path = os.path.join(args.output_dir, f"{demo_name}_{best_method}.mp4")
        writer = imageio.get_writer(video_path, fps=20, format="FFMPEG", codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"  {demo_name} -> {video_path}")

    hdf5_data.close()
    print(f"\nDone. Videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
