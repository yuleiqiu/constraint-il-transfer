"""
Replay delta_eef_action labels through the simulator and save videos.

For each selected demo:
  1. Reset env to initial state
  2. Step through delta_eef_action as the controller input (already in [-1,1] OSC units)
  3. Record agentview video frames
  4. Save as .mp4 in outputs/replay_delta_eef/

Usage:
    MUJOCO_GL=egl uv run python scripts/replay_delta_eef_to_video.py \
        --dataset third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5 \
        --demos 0 1 2 3 4
"""

import argparse
import json
import os
import h5py
import numpy as np
import imageio

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


def replay_single_demo(env, hdf5_data, demo_name: str, output_dir: str, video_fps: int = 20):
    """Replay one demo and save video."""
    grp = hdf5_data[f"data/{demo_name}"]
    initial_state = dict(states=grp["states"][0])
    initial_state["model"] = grp.attrs["model_file"]
    initial_state["ep_meta"] = grp.attrs.get("ep_meta", None)

    delta_eef = grp["delta_eef_action"][:]
    T = delta_eef.shape[0]

    env.reset_to(initial_state)

    frames = []
    for t in range(T):
        # Render current frame
        frame = env.render(mode="rgb_array", height=256, width=256, camera_name="agentview")
        frames.append(frame)

        # Step with delta_eef as action (already in [-1, 1] OSC units)
        action = delta_eef[t]
        env.step(action)

    # Render final frame
    frame = env.render(mode="rgb_array", height=256, width=256, camera_name="agentview")
    frames.append(frame)

    # Save video
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{demo_name}.mp4")
    writer = imageio.get_writer(video_path, fps=video_fps, format="FFMPEG", codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    return video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--demos", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=str, default="outputs/replay_delta_eef")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    args = parser.parse_args()

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta = FileUtils.get_env_metadata_from_dataset(args.dataset)
    hdf5_data = h5py.File(args.dataset, "r")

    demos = sorted(hdf5_data["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    selected = [demos[i] for i in args.demos if i < len(demos)]

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Environ: {env_meta['env_name']}")
    print(f"Replaying {len(selected)} demos: {selected}")
    print()

    for demo_name in selected:
        video_path = replay_single_demo(
            env=env,
            hdf5_data=hdf5_data,
            demo_name=demo_name,
            output_dir=args.output_dir,
            video_fps=args.video_fps,
        )
        print(f"  {demo_name} -> {video_path}")

    hdf5_data.close()

    print(f"\nDone. Videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
