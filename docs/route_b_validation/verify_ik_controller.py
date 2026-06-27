"""
Verify that IK controller can replay a trajectory when fed cumulative EEF deltas.

Approach: IK controller in delta mode, but the deltas are the ACTUAL EEF
displacements (in meters). The IK's internal target becomes next_eef_pos[t]
cumulatively. A position-controller on the joints (JointPositionController)
then drives the EEF to that target.

Action sent to env (7D):
  [dx_meters, dy_meters, dz_meters, ax, ay, az, gripper]
  - dx_meters = next_eef_pos[t] - current_eef_pos[t]  (actual displacement)
  - ax, ay, az = 0 (don't change orientation)
  - gripper = from data
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


def make_env_with_ik(env_meta, kp=100, kv=10):
    """Create env with IK_POSE controller, delta mode."""
    env_kwargs = json.loads(json.dumps(env_meta["env_kwargs"]))

    for body_part in env_kwargs["controller_configs"]["body_parts"].values():
        if body_part.get("type") == "OSC_POSE":
            # Swap to IK_POSE - drop OSC-specific fields
            osc_keys = list(body_part.keys())
            for k in osc_keys:
                if k not in ("type", "gripper"):
                    del body_part[k]
            body_part["type"] = "IK_POSE"
            body_part["ik_pos_limit"] = 0.05
            body_part["ik_ori_limit"] = 0.5
            body_part["interpolation"] = None

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


def replay_one_demo(env, demo_grp, use_orientation, label):
    """Replay using actual EEF deltas fed to IK controller."""
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

    # CRITICAL fix: ref_pos and reference_target_pos are stale after reset_to
    for part_name, ctrl in env.env.robots[0].part_controllers.items():
        if hasattr(ctrl, "update"):
            ctrl.update(force=True)
        if hasattr(ctrl, "reset_goal"):
            ctrl.reset_goal()
        # Override IK's hard-coded user_sensitivity=0.3 → 1.0
        if hasattr(ctrl, "user_sensitivity"):
            ctrl.user_sensitivity = 1.0
        # DISABLE nullspace control — pulls joints toward home pose, causing 5x overshoot
        if hasattr(ctrl, "Kn"):
            ctrl.Kn = np.zeros_like(ctrl.Kn)

    T_steps = actions.shape[0]
    replay_traj = [obs_eef_pos[0].copy()]
    for t in range(T_steps):
        # Compute actual displacement
        dpos_meters = next_eef_pos[t] - obs_eef_pos[t]
        if not use_orientation:
            drot = np.zeros(3)
        else:
            drot = T.quat2axisangle(next_eef_quat[t])
        gripper = actions[t, 6]

        # IK controller expects action in [-1, 1] for delta, scaled by user_sensitivity
        # We pass meters directly (user_sensitivity=1.0 default)
        # ik_pos_limit clips to ±0.05m
        action = np.concatenate([dpos_meters, drot, [gripper]])

        obs, _, _, _ = env.step(action)
        replay_traj.append(obs["robot0_eef_pos"].copy())

    replay_traj = np.array(replay_traj)
    err_to_target = np.linalg.norm(replay_traj[1:] - next_eef_pos, axis=1)
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
    parser.add_argument("--kp", type=float, nargs="+", default=[100, 200])
    parser.add_argument("--use-orientation", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/verify_ik_controller")
    args = parser.parse_args()

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    print(f"Env: {env_meta['env_name']}, robots: {env_meta['env_kwargs']['robots']}")
    print(f"use_orientation={args.use_orientation}")
    print(f"Testing kp values: {args.kp}")
    print()

    f = h5py.File(args.dataset, "r")
    all_demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    selected = all_demos[:args.n_demos]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for kp in args.kp:
        print(f"\n{'='*70}\nkp = {kp}\n{'='*70}")
        env = make_env_with_ik(env_meta, kp=kp)
        # Verify the actual controller
        ctrl_name = list(env.env.robots[0].part_controllers.keys())
        print(f"Controllers: {ctrl_name}")
        ctrl = env.env.robots[0].part_controllers[ctrl_name[0]]
        print(f"Controller type: {type(ctrl).__name__}")
        print(f"ik_pos_limit: {ctrl.ik_pos_limit}, ik_ori_limit: {ctrl.ik_ori_limit}")
        print(f"use_delta: {ctrl.use_delta}")

        results = []
        for demo_name in selected:
            grp = f[f"data/{demo_name}"]
            r = replay_one_demo(env, grp, args.use_orientation,
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
