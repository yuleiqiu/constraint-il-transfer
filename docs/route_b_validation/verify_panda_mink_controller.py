"""
Verify Panda + WholeBodyMinkIK as an EEF-native action interface.

This script replays recorded EEF positions from the delta-eef dataset using a
Panda-specific Mink IK composite controller:

    action[t] = [next_obs/robot0_eef_pos[t], 0, 0, 0, gripper[t]]

The purpose is to test whether robosuite can execute absolute EEF targets
reliably enough to support Route B.

Run from repo root:
    MUJOCO_GL=egl uv run python docs/route_b_validation/verify_panda_mink_controller.py
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import robosuite
import robosuite.examples.third_party_controller.mink_controller  # noqa: F401 - registers WHOLE_BODY_MINK_IK
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from robosuite.utils import transform_utils as T


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5"
DEFAULT_CONTROLLER_CONFIG = (
    REPO_ROOT / "third_party/robosuite/robosuite/controllers/config/default/composite/panda_mink_ik.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/route_b_validation/panda_mink_controller"


def decode_demo_names(raw_names):
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names]


def get_demo_names(h5_file, split, n_demos):
    if "mask" in h5_file and split in h5_file["mask"]:
        demo_names = decode_demo_names(h5_file["mask"][split][:])
    else:
        demo_names = sorted(h5_file["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    return demo_names[:n_demos]


def load_controller_config(path):
    return robosuite.load_composite_controller_config(controller=str(path))


def make_env(env_meta, controller_config_path):
    env_kwargs = json.loads(json.dumps(env_meta["env_kwargs"]))
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


def quat_angle_error(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.clip(abs(float(np.dot(q1, q2))), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def get_target_axis_angle(grp, t, orientation_source):
    if orientation_source == "none":
        return np.zeros(3, dtype=np.float64)
    return T.quat2axisangle(grp[f"next_obs/{orientation_source}"][t].copy())


def replay(env, grp, label, orientation_source):
    states = grp["states"][:]
    obs = env.reset_to(
        {
            "states": states[0],
            "model": grp.attrs["model_file"],
            "ep_meta": grp.attrs.get("ep_meta", None),
        }
    )
    reset_controller_refs(env)

    n_steps = grp["actions"].shape[0]
    err_to_target_list = []
    err_to_orig_list = []
    desired_dpos_list = []
    actual_dpos_list = []
    ori_err_to_target_list = []
    ori_err_to_orig_list = []
    rewards = []
    success_per_step = []
    replay_traj = [obs["robot0_eef_pos"].copy()]

    for t in range(n_steps):
        before = obs["robot0_eef_pos"].copy()
        target_pos = grp["next_obs/robot0_eef_pos"][t]
        action = np.zeros(env.env.action_dim, dtype=np.float64)
        action[:3] = target_pos
        action[3:6] = get_target_axis_angle(grp, t, orientation_source)
        action[6] = grp["actions"][t, 6]

        err_to_orig_list.append(np.linalg.norm(before - grp["obs/robot0_eef_pos"][t]))
        if orientation_source != "none":
            ori_err_to_orig_list.append(
                quat_angle_error(obs[orientation_source], grp[f"obs/{orientation_source}"][t])
            )
        obs, reward, _, info = env.step(action)
        after = obs["robot0_eef_pos"].copy()
        replay_traj.append(after)
        rewards.append(float(reward))
        success_per_step.append(bool(info["is_success"]["task"]))
        if orientation_source != "none":
            ori_err_to_target_list.append(
                quat_angle_error(obs[orientation_source], grp[f"next_obs/{orientation_source}"][t])
            )

        desired_dpos_list.append(target_pos - before)
        actual_dpos_list.append(after - before)
        err_to_target_list.append(np.linalg.norm(after - target_pos))

    replay_traj = np.asarray(replay_traj)
    desired_mags = np.asarray([np.linalg.norm(x) for x in desired_dpos_list])
    actual_mags = np.asarray([np.linalg.norm(x) for x in actual_dpos_list])
    nontrivial = desired_mags > 0.005
    if np.any(nontrivial):
        ratios = actual_mags[nontrivial] / desired_mags[nontrivial]
        tracking_median = float(np.median(ratios))
        tracking_p25 = float(np.percentile(ratios, 25))
        tracking_p75 = float(np.percentile(ratios, 75))
    else:
        tracking_median = tracking_p25 = tracking_p75 = float("nan")

    return {
        "label": label,
        "n_steps": int(n_steps),
        "desired_dpos_mag_mean_cm": float(np.mean(desired_mags) * 100),
        "actual_dpos_mag_mean_cm": float(np.mean(actual_mags) * 100),
        "tracking_median": tracking_median,
        "tracking_p25": tracking_p25,
        "tracking_p75": tracking_p75,
        "end_err_to_target_cm": float(np.linalg.norm(replay_traj[-1] - grp["next_obs/robot0_eef_pos"][-1]) * 100),
        "err_to_target_mean_cm": float(np.mean(err_to_target_list) * 100),
        "err_to_target_max_cm": float(np.max(err_to_target_list) * 100),
        "end_err_to_orig_cm": float(np.linalg.norm(replay_traj[-1] - grp["obs/robot0_eef_pos"][-1]) * 100),
        "orientation_source": orientation_source,
        "ori_err_to_target_mean_deg": float(np.mean(ori_err_to_target_list) * 180 / np.pi)
        if ori_err_to_target_list
        else None,
        "ori_err_to_target_max_deg": float(np.max(ori_err_to_target_list) * 180 / np.pi)
        if ori_err_to_target_list
        else None,
        "ori_err_to_orig_mean_deg": float(np.mean(ori_err_to_orig_list) * 180 / np.pi)
        if ori_err_to_orig_list
        else None,
        "success_any": bool(np.any(success_per_step)),
        "success_final": bool(success_per_step[-1]) if success_per_step else False,
        "success_first_step": int(np.argmax(success_per_step)) if np.any(success_per_step) else None,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        "reward_final": float(rewards[-1]) if rewards else 0.0,
        "err_to_orig_per_step_cm": [float(x * 100) for x in err_to_orig_list],
        "err_to_target_per_step_cm": [float(x * 100) for x in err_to_target_list],
        "ori_err_to_target_per_step_deg": [float(x * 180 / np.pi) for x in ori_err_to_target_list],
        "success_per_step": success_per_step,
        "reward_per_step": rewards,
        "replay_traj": replay_traj.tolist(),
        "data_traj": grp["obs/robot0_eef_pos"][:].tolist(),
    }


def summarize(results):
    ori_target_vals = [r["ori_err_to_target_mean_deg"] for r in results if r["ori_err_to_target_mean_deg"] is not None]
    ori_target_max_vals = [r["ori_err_to_target_max_deg"] for r in results if r["ori_err_to_target_max_deg"] is not None]
    return {
        "n_demos": len(results),
        "desired_dpos_mag_mean_cm": float(np.mean([r["desired_dpos_mag_mean_cm"] for r in results])),
        "actual_dpos_mag_mean_cm": float(np.mean([r["actual_dpos_mag_mean_cm"] for r in results])),
        "tracking_median": float(np.nanmean([r["tracking_median"] for r in results])),
        "err_to_target_mean_cm": float(np.mean([r["err_to_target_mean_cm"] for r in results])),
        "end_err_to_orig_cm": float(np.mean([r["end_err_to_orig_cm"] for r in results])),
        "end_err_to_target_cm": float(np.mean([r["end_err_to_target_cm"] for r in results])),
        "success_any_rate": float(np.mean([r["success_any"] for r in results])),
        "success_final_rate": float(np.mean([r["success_final"] for r in results])),
        "reward_max_mean": float(np.mean([r["reward_max"] for r in results])),
        "reward_final_mean": float(np.mean([r["reward_final"] for r in results])),
        "ori_err_to_target_mean_deg": float(np.mean(ori_target_vals)) if ori_target_vals else None,
        "ori_err_to_target_max_deg": float(np.max(ori_target_max_vals)) if ori_target_max_vals else None,
    }


def write_summary(output_dir, dataset, controller_config, split, demo_names, orientation_source, summary):
    lines = [
        "# Panda Mink IK Controller Replay",
        "",
        f"Dataset: `{dataset}`",
        f"Controller: `{controller_config}`",
        f"Split: `{split}`",
        f"Orientation source: `{orientation_source}`",
        f"Demos: `{', '.join(demo_names)}`",
        "",
        "## Summary",
        "",
        "| desired cm | actual cm | tracking | mean target err cm | end orig err cm | ori target err deg | success any | success final |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {desired:.3f} | {actual:.3f} | {track:.3f} | {target:.3f} | {end:.3f} | {ori} | {succ_any:.3f} | {succ_final:.3f} |".format(
            desired=summary["desired_dpos_mag_mean_cm"],
            actual=summary["actual_dpos_mag_mean_cm"],
            track=summary["tracking_median"],
            target=summary["err_to_target_mean_cm"],
            end=summary["end_err_to_orig_cm"],
            ori="n/a"
            if summary["ori_err_to_target_mean_deg"] is None
            else f"{summary['ori_err_to_target_mean_deg']:.3f}",
            succ_any=summary["success_any_rate"],
            succ_final=summary["success_final_rate"],
        ),
        "",
        "Pass criterion for Route B controller validation:",
        "",
        "```text",
        "mean target error < 1 cm",
        "end orig error < 1 cm",
        "```",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--controller-config", type=Path, default=DEFAULT_CONTROLLER_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--n-demos", type=int, default=5)
    parser.add_argument(
        "--orientation-source",
        choices=["none", "robot0_eef_quat_site", "robot0_eef_quat"],
        default="none",
        help="Use recorded absolute EEF orientation from next_obs/<source>; 'none' keeps position-only replay.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    controller_config_path = args.controller_config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))

    results = []
    with h5py.File(dataset_path, "r") as f:
        demo_names = get_demo_names(f, args.split, args.n_demos)
        env = make_env(env_meta, controller_config_path)
        try:
            print(f"Action dim: {env.env.action_dim}")
            for demo_name in demo_names:
                grp = f[f"data/{demo_name}"]
                result = replay(env, grp, f"panda_mink:{demo_name}", orientation_source=args.orientation_source)
                results.append(result)
                print(
                    f"{demo_name}: desired={result['desired_dpos_mag_mean_cm']:.3f}cm "
                    f"actual={result['actual_dpos_mag_mean_cm']:.3f}cm "
                    f"track={result['tracking_median']:.3f} "
                    f"target_err={result['err_to_target_mean_cm']:.3f}cm "
                    f"ori_err={result['ori_err_to_target_mean_deg'] if result['ori_err_to_target_mean_deg'] is not None else 'n/a'}deg "
                    f"end_orig={result['end_err_to_orig_cm']:.3f}cm "
                    f"succ_any={int(result['success_any'])} "
                    f"succ_final={int(result['success_final'])}"
                )
        finally:
            env.env.close()

    summary = summarize(results)
    out = {
        "dataset": str(dataset_path),
        "controller_config": str(controller_config_path),
        "split": args.split,
        "orientation_source": args.orientation_source,
        "demo_names": demo_names,
        "summary": summary,
        "results": results,
    }
    (output_dir / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    write_summary(output_dir, dataset_path, controller_config_path, args.split, demo_names, args.orientation_source, summary)

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {output_dir / 'results.json'}")
    print(f"Wrote summary to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
