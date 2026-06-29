"""
Render videos for Panda + WholeBodyMinkIK EEF replay failures.

Run from repo root:
    MUJOCO_GL=egl uv run python docs/route_b_validation/render_panda_mink_failure_videos.py
"""

import argparse
import json
from pathlib import Path

import h5py
import imageio
import numpy as np

import robosuite
import robosuite.examples.third_party_controller.mink_controller  # noqa: F401 - registers WHOLE_BODY_MINK_IK
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5"
DEFAULT_CONTROLLER_CONFIG = (
    REPO_ROOT / "third_party/robosuite/robosuite/controllers/config/default/composite/panda_mink_ik.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/route_b_validation/panda_mink_controller/failure_videos_default"
DEFAULT_DEMOS = ["demo_140", "demo_100", "demo_101", "demo_104"]


def parse_demos(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_controller_config(path):
    return robosuite.load_composite_controller_config(controller=str(path))


def make_env(env_meta, controller_config_path, mode):
    env_kwargs = json.loads(json.dumps(env_meta["env_kwargs"]))
    if mode == "mink":
        env_kwargs["controller_configs"] = load_controller_config(controller_config_path)
    return EnvUtils.create_env_from_metadata(
        env_meta=dict(
            type=env_meta["type"],
            env_name=env_meta["env_name"],
            env_version=env_meta.get("env_version"),
            env_kwargs=env_kwargs,
        ),
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )


def reset_controller_refs(env):
    for ctrl in env.env.robots[0].part_controllers.values():
        if hasattr(ctrl, "update"):
            ctrl.update(force=True)
        if hasattr(ctrl, "reset_goal"):
            ctrl.reset_goal()
        if hasattr(ctrl, "user_sensitivity"):
            ctrl.user_sensitivity = 1.0


def render_frame(env, camera_names, height, width):
    frames = [
        env.render(mode="rgb_array", height=height, width=width, camera_name=camera_name)
        for camera_name in camera_names
    ]
    return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]


def make_action(env, grp, t, mode):
    if mode == "osc":
        return grp["actions"][t]

    action = np.zeros(env.env.action_dim, dtype=np.float64)
    action[:3] = grp["next_obs/robot0_eef_pos"][t]
    action[3:6] = 0.0
    action[6] = grp["actions"][t, 6]
    return action


def render_demo(env, grp, demo_name, mode, output_dir, camera_names, height, width, fps):
    obs = env.reset_to(
        {
            "states": grp["states"][0],
            "model": grp.attrs["model_file"],
            "ep_meta": grp.attrs.get("ep_meta", None),
        }
    )
    if mode == "mink":
        reset_controller_refs(env)

    output_path = output_dir / f"{demo_name}_{mode}_{'_'.join(camera_names)}.mp4"
    success_per_step = []
    rewards = []
    eef_err_to_orig = []

    with imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=1) as writer:
        writer.append_data(render_frame(env, camera_names, height, width))
        for t in range(grp["actions"].shape[0]):
            eef_err_to_orig.append(float(np.linalg.norm(obs["robot0_eef_pos"] - grp["obs/robot0_eef_pos"][t]) * 100))
            obs, reward, _, info = env.step(make_action(env, grp, t, mode))
            rewards.append(float(reward))
            success_per_step.append(bool(info["is_success"]["task"]))
            writer.append_data(render_frame(env, camera_names, height, width))

    return {
        "demo": demo_name,
        "mode": mode,
        "video": str(output_path),
        "n_steps": int(grp["actions"].shape[0]),
        "success_any": bool(np.any(success_per_step)),
        "success_final": bool(success_per_step[-1]) if success_per_step else False,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        "reward_final": float(rewards[-1]) if rewards else 0.0,
        "eef_err_to_orig_mean_cm": float(np.mean(eef_err_to_orig)) if eef_err_to_orig else 0.0,
        "eef_err_to_orig_max_cm": float(np.max(eef_err_to_orig)) if eef_err_to_orig else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--controller-config", type=Path, default=DEFAULT_CONTROLLER_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demos", type=str, default=",".join(DEFAULT_DEMOS))
    parser.add_argument("--mode", choices=["mink", "osc"], default="mink")
    parser.add_argument("--camera-names", type=str, default="agentview,robot0_eye_in_hand")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    controller_config_path = args.controller_config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_names = parse_demos(args.demos)
    camera_names = parse_demos(args.camera_names)

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))

    results = []
    with h5py.File(dataset_path, "r") as f:
        env = make_env(env_meta, controller_config_path, args.mode)
        try:
            for demo_name in demo_names:
                grp = f[f"data/{demo_name}"]
                result = render_demo(
                    env=env,
                    grp=grp,
                    demo_name=demo_name,
                    mode=args.mode,
                    output_dir=output_dir,
                    camera_names=camera_names,
                    height=args.height,
                    width=args.width,
                    fps=args.fps,
                )
                results.append(result)
                print(
                    f"{demo_name}: video={result['video']} "
                    f"success_final={int(result['success_final'])} "
                    f"reward_final={result['reward_final']:.1f} "
                    f"eef_err_max={result['eef_err_to_orig_max_cm']:.3f}cm"
                )
        finally:
            env.env.close()

    summary = {
        "dataset": str(dataset_path),
        "controller_config": str(controller_config_path) if args.mode == "mink" else None,
        "mode": args.mode,
        "camera_names": camera_names,
        "results": results,
    }
    summary_path = output_dir / f"video_summary_{args.mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
