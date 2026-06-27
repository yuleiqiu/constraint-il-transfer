"""
Verify the "position controller" approach for Route B.

Hypothesis: if the supervision signal is `delta_eef_action` (real EEF displacement in OSC units),
then we can execute it via OSC's `input_type="absolute"` mode by sending the cumulative
target EEF position each step. The OSC PD controller will drive the EEF to the absolute target.

This script tests open-loop replay fidelity:
  - For each demo, reset env to demo's initial state.
  - At each step t, send the data's `next_obs_eef_pos[t]` as the absolute position target.
  - After env.step, record the achieved EEF.
  - Compare to:
      a) the absolute target we sent (tracking error of the controller)
      b) the recorded `next_obs_eef_pos[t]` (overall replay fidelity)

If both are small (sub-cm), `delta_eef_action` is a viable supervision signal
and the policy can be trained to predict it.
"""

import argparse
import json
import os
import h5py
import numpy as np

import robosuite.utils.transform_utils as T

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


def make_env_with_absolute_osc(env_meta, kp, damping_ratio, ref_frame):
    """Reconstruct env with OSC in absolute mode, overriding the dataset's delta config."""
    env_kwargs = json.loads(json.dumps(env_meta["env_kwargs"]))

    for body_part in env_kwargs["controller_configs"]["body_parts"].values():
        if body_part.get("type") == "OSC_POSE":
            body_part["input_type"] = "absolute"
            body_part["input_ref_frame"] = ref_frame
            body_part["kp"] = kp
            body_part["damping_ratio"] = damping_ratio
            body_part["impedance_mode"] = "fixed"
            # CRITICAL: uncouple_pos_ori must be False in absolute mode.
            # With uncoupling=True, the OSC applies force in the wrong direction
            # when given an absolute target (verified empirically).
            body_part["uncouple_pos_ori"] = False

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


def replay_one_demo(env, demo_grp, use_rotation, label):
    """Replay a single demo by sending absolute EEF targets to OSC."""
    states = demo_grp["states"][:]
    model_xml = demo_grp.attrs["model_file"]
    ep_meta = demo_grp.attrs.get("ep_meta", None)

    obs_eef_pos = demo_grp["obs/robot0_eef_pos"][:]
    next_eef_pos = demo_grp["next_obs/robot0_eef_pos"][:]
    next_eef_quat = demo_grp["next_obs/robot0_eef_quat"][:]
    actions = demo_grp["actions"][:]

    env.reset_to({
        "states": states[0],
        "model": model_xml,
        "ep_meta": ep_meta,
    })

    # CRITICAL: env.reset() inside reset_to set the controller's ref_pos to the
    # env's initial qpos, but then set_state_from_flattened overwrote the sim
    # state. Force the controller to re-read its reference.
    for part_name, ctrl in env.env.robots[0].part_controllers.items():
        if hasattr(ctrl, "update"):
            ctrl.update(force=True)

    T_steps = actions.shape[0]
    replay_traj = [obs_eef_pos[0].copy()]
    for t in range(T_steps):
        target_pos = next_eef_pos[t]
        if use_rotation:
            target_aa = T.quat2axisangle(next_eef_quat[t])
        else:
            target_aa = np.zeros(3)
        gripper = actions[t, 6]
        action = np.concatenate([target_pos, target_aa, [gripper]])
        action = np.clip(action, -1, 1)  # OSC input range

        obs, _, _, _ = env.step(action)
        replay_traj.append(obs["robot0_eef_pos"].copy())

    replay_traj = np.array(replay_traj)  # (T+1, 3)

    # err_to_target: replay_traj[1:] (after each step) vs next_eef_pos (target at each step)
    err_to_target = np.linalg.norm(replay_traj[1:] - next_eef_pos, axis=1)
    # err_to_original: replay_traj[:-1] (before each step) vs obs_eef_pos (recorded start of each step)
    err_to_original = np.linalg.norm(replay_traj[:-1] - obs_eef_pos, axis=1)

    return {
        "label": label,
        "n_steps": T_steps,
        "err_target_mean_cm": float(np.mean(err_to_target) * 100),
        "err_target_max_cm": float(np.max(err_to_target) * 100),
        "err_target_end_cm": float(err_to_target[-1] * 100),
        "err_orig_mean_cm": float(np.mean(err_to_original) * 100),
        "err_orig_max_cm": float(np.max(err_to_original) * 100),
        "err_orig_end_cm": float(err_to_original[-1] * 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-demos", type=int, default=5)
    parser.add_argument("--kp", type=float, nargs="+", default=[150.0, 300.0],
                        help="One or more kp values to test")
    parser.add_argument("--ref-frame", type=str, default="world", choices=["world", "base"])
    parser.add_argument("--use-rotation", action="store_true",
                        help="Also send absolute orientation targets")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/verify_position_controller")
    args = parser.parse_args()

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    print(f"Env: {env_meta['env_name']}, robots: {env_meta['env_kwargs']['robots']}")
    print(f"ref_frame={args.ref_frame}, use_rotation={args.use_rotation}")
    print(f"Testing kp values: {args.kp}")
    print()

    f = h5py.File(args.dataset, "r")
    all_demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    selected = all_demos[:args.n_demos]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for kp in args.kp:
        print(f"\n{'='*70}\nkp = {kp}\n{'='*70}")
        env = make_env_with_absolute_osc(
            env_meta, kp=kp, damping_ratio=1.0, ref_frame=args.ref_frame,
        )

        results = []
        for demo_name in selected:
            grp = f[f"data/{demo_name}"]
            r = replay_one_demo(env, grp, args.use_rotation,
                                label=f"kp={kp},{demo_name}")
            results.append(r)
            print(f"  {demo_name}: "
                  f"err_to_target mean={r['err_target_mean_cm']:.2f}cm  "
                  f"max={r['err_target_max_cm']:.2f}cm  "
                  f"end={r['err_target_end_cm']:.2f}cm | "
                  f"err_to_orig mean={r['err_orig_mean_cm']:.2f}cm  "
                  f"end={r['err_orig_end_cm']:.2f}cm")

        all_results[f"kp={kp}"] = results

        avg_t = np.mean([r["err_target_mean_cm"] for r in results])
        avg_o = np.mean([r["err_orig_mean_cm"] for r in results])
        max_t = np.max([r["err_target_max_cm"] for r in results])
        max_o = np.max([r["err_orig_max_cm"] for r in results])
        print(f"\n  AVG: target_mean={avg_t:.2f}cm  target_max={max_t:.2f}cm  "
              f"orig_mean={avg_o:.2f}cm  orig_max={max_o:.2f}cm")
        try:
            env.env.close()
        except Exception:
            pass

    f.close()

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as fp:
        for k, results in all_results.items():
            fp.write(f"{k}\n")
            for r in results:
                fp.write(json.dumps(r) + "\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
