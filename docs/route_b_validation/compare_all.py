"""
Run a clean comparative experiment across all candidate supervision signals.

Records per-step "desired" vs "actual" EEF displacement, plus trajectory error
metrics. Writes results.json into the same directory as this script.

Run from this directory (or with no args; all paths are derived from __file__):
    MUJOCO_GL=egl uv run python compare_all.py
"""

import json
import os
import sys
import h5py
import numpy as np

import robosuite.utils.transform_utils as T

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5",
)
N_DEMOS = 5
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_env(env_meta, controller_overrides):
    """Apply overrides to the OSC body part. For IK swaps, we strip OSC fields."""
    env_kwargs = json.loads(json.dumps(env_meta["env_kwargs"]))
    for body_part in env_kwargs["controller_configs"]["body_parts"].values():
        if body_part.get("type") in ("OSC_POSE", "IK_POSE"):
            new_type = controller_overrides.get("type", body_part["type"])
            if new_type == "IK_POSE":
                osc_keys = list(body_part.keys())
                for k in osc_keys:
                    if k not in ("type", "gripper"):
                        del body_part[k]
                # IK factory requires these
                body_part["interpolation"] = None
                body_part["ramp_ratio"] = 0.2
            for k, v in controller_overrides.items():
                if v is None:
                    if k in body_part:
                        body_part.pop(k)
                else:
                    body_part[k] = v
    return EnvUtils.create_env_from_metadata(
        env_meta=dict(
            type=env_meta["type"],
            env_name=env_meta["env_name"],
            env_version=env_meta.get("env_version"),
            env_kwargs=env_kwargs,
        ),
        render=False, render_offscreen=True, use_image_obs=False,
    )


def replay(env, grp, action_fn, label, action_kind="delta_normalized"):
    """Run replay and return per-step (desired_dpos, actual_dpos, err_to_target) stats.

    `action_kind`:
      - "delta_normalized": action[:3] is in [-1,1] normalized, intended dpos = action * 0.05 m
      - "delta_meters": action[:3] is the desired EEF delta in meters directly
      - "absolute": action[:3] is the absolute EEF target in world frame; intended dpos = target - current
    """
    states = grp["states"][:]
    obs = env.reset_to({
        "states": states[0],
        "model": grp.attrs["model_file"],
        "ep_meta": grp.attrs.get("ep_meta", None),
    })

    # Fix stale internal state
    for part_name, ctrl in env.env.robots[0].part_controllers.items():
        if hasattr(ctrl, "update"):
            ctrl.update(force=True)
        if hasattr(ctrl, "reset_goal"):
            ctrl.reset_goal()
        if hasattr(ctrl, "user_sensitivity"):
            ctrl.user_sensitivity = 1.0

    T_steps = grp["actions"].shape[0]
    desired_dpos_list = []
    actual_dpos_list = []
    err_to_target_list = []
    err_to_orig_list = []  # NEW: |replay_eef - data.obs_eef_pos[t]|
    replay_traj = [obs["robot0_eef_pos"].copy()]

    for t in range(T_steps):
        action = action_fn(t, grp, obs, env)
        before = obs["robot0_eef_pos"].copy()
        # err_to_orig: |replay_start - data_start| (compare same-step start)
        err_to_orig_list.append(np.linalg.norm(before - grp["obs/robot0_eef_pos"][t]))
        obs, _, _, _ = env.step(action)
        after = obs["robot0_eef_pos"]
        replay_traj.append(after.copy())

        if action_kind == "delta_normalized":
            desired_dpos = action[:3] * 0.05
        elif action_kind == "delta_meters":
            desired_dpos = action[:3]
        elif action_kind == "absolute":
            desired_dpos = action[:3] - before
        else:
            raise ValueError(f"Unknown action_kind: {action_kind}")
        actual_dpos = after - before
        desired_dpos_list.append(desired_dpos)
        actual_dpos_list.append(actual_dpos)
        err_to_target_list.append(np.linalg.norm(after - grp["next_obs/robot0_eef_pos"][t]))

    replay_traj = np.array(replay_traj)
    desired_mags = np.array([np.linalg.norm(d) for d in desired_dpos_list])
    actual_mags = np.array([np.linalg.norm(a) for a in actual_dpos_list])
    nontrivial = desired_mags > 0.005
    if nontrivial.sum() > 0:
        ratios = actual_mags[nontrivial] / desired_mags[nontrivial]
        track_median = float(np.median(ratios))
        track_p25 = float(np.percentile(ratios, 25))
        track_p75 = float(np.percentile(ratios, 75))
    else:
        track_median = track_p25 = track_p75 = float('nan')

    return {
        "label": label,
        "n_steps": T_steps,
        "desired_dpos_mag_mean_cm": float(np.mean(desired_mags) * 100),
        "actual_dpos_mag_mean_cm": float(np.mean(actual_mags) * 100),
        "tracking_median": track_median,
        "tracking_p25": track_p25,
        "tracking_p75": track_p75,
        "end_err_to_target_cm": float(np.linalg.norm(replay_traj[-1] - grp["next_obs/robot0_eef_pos"][-1]) * 100),
        "err_to_target_mean_cm": float(np.mean(err_to_target_list) * 100),
        "err_to_target_max_cm": float(np.max(err_to_target_list) * 100),
        "end_err_to_orig_cm": float(np.linalg.norm(replay_traj[-1] - grp["obs/robot0_eef_pos"][-1]) * 100),
        "err_to_orig_per_step_cm": [float(x * 100) for x in err_to_orig_list],  # NEW
        "err_to_target_per_step_cm": [float(x * 100) for x in err_to_target_list],  # NEW
        "replay_traj": replay_traj.tolist(),  # NEW
        "data_traj": grp["obs/robot0_eef_pos"][:].tolist(),  # NEW
    }


def main():
    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=DATASET)
    f = h5py.File(DATASET, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    selected = demos[:N_DEMOS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    # === Plan A: original action in OSC delta mode (baseline, known to work) ===
    print("=== Plan A: original action, OSC delta mode ===")
    env = make_env(env_meta, {"type": "OSC_POSE"})
    results = []
    for demo_name in selected:
        grp = f[f"data/{demo_name}"]
        r = replay(env, grp, lambda t, g, o, e: g["actions"][t], f"A:{demo_name}",
                   action_kind="delta_normalized")
        results.append(r)
        print(f"  {demo_name}: desired={r['desired_dpos_mag_mean_cm']:.3f}cm  "
              f"actual={r['actual_dpos_mag_mean_cm']:.3f}cm  "
              f"track_med={r['tracking_median']:.3f}x  "
              f"end_err_target={r['end_err_to_target_cm']:.3f}cm  "
              f"end_err_orig={r['end_err_to_orig_cm']:.3f}cm")
    all_results["plan_A_delta_action_osc"] = results
    env.env.close()

    # === Plan B-1: delta_eef_action (real EEF delta in OSC units) as OSC action ===
    print("\n=== Plan B-1: delta_eef_action in OSC delta mode ===")
    env = make_env(env_meta, {"type": "OSC_POSE"})
    results = []
    for demo_name in selected:
        grp = f[f"data/{demo_name}"]
        def b1_action(t, g, o, e, _grp=grp):
            return _grp["delta_eef_action"][t]
        r = replay(env, grp, b1_action, f"B1:{demo_name}", action_kind="delta_normalized")
        results.append(r)
        print(f"  {demo_name}: desired={r['desired_dpos_mag_mean_cm']:.3f}cm  "
              f"actual={r['actual_dpos_mag_mean_cm']:.3f}cm  "
              f"track_med={r['tracking_median']:.3f}x  "
              f"end_err_target={r['end_err_to_target_cm']:.3f}cm  "
              f"end_err_orig={r['end_err_to_orig_cm']:.3f}cm")
    all_results["plan_B1_delta_eef_osc"] = results
    env.env.close()

    # === Plan B-2: next_eef_pos as absolute target, OSC absolute mode ===
    print("\n=== Plan B-2: next_eef_pos absolute, OSC absolute mode ===")
    env = make_env(env_meta, {
        "type": "OSC_POSE", "input_type": "absolute",
        "input_ref_frame": "world", "kp": 150.0,
        "uncouple_pos_ori": False,
    })
    results = []
    for demo_name in selected:
        grp = f[f"data/{demo_name}"]
        def b2_action(t, g, o, e, _grp=grp):
            return np.concatenate([_grp["next_obs/robot0_eef_pos"][t], np.zeros(3), [_grp["actions"][t, 6]]])
        r = replay(env, grp, b2_action, f"B2:{demo_name}", action_kind="absolute")
        results.append(r)
        print(f"  {demo_name}: desired={r['desired_dpos_mag_mean_cm']:.3f}cm  "
              f"actual={r['actual_dpos_mag_mean_cm']:.3f}cm  "
              f"track_med={r['tracking_median']:.3f}x  "
              f"end_err_target={r['end_err_to_target_cm']:.3f}cm  "
              f"end_err_orig={r['end_err_to_orig_cm']:.3f}cm")
    all_results["plan_B2_absolute_osc"] = results
    env.env.close()

    # === Plan C: next_eef_pos cumulative delta, IK delta mode ===
    print("\n=== Plan C: cumulative next_eef_pos delta, IK delta mode ===")
    env = make_env(env_meta, {
        "type": "IK_POSE", "ik_pos_limit": 0.05, "ik_ori_limit": 0.5,
    })
    results = []
    for demo_name in selected:
        grp = f[f"data/{demo_name}"]
        def c_action(t, g, o, e, _grp=grp):
            return np.concatenate([_grp["next_obs/robot0_eef_pos"][t] - _grp["obs/robot0_eef_pos"][t],
                                   np.zeros(3), [_grp["actions"][t, 6]]])
        r = replay(env, grp, c_action, f"C:{demo_name}", action_kind="delta_meters")
        results.append(r)
        print(f"  {demo_name}: desired={r['desired_dpos_mag_mean_cm']:.3f}cm  "
              f"actual={r['actual_dpos_mag_mean_cm']:.3f}cm  "
              f"track_med={r['tracking_median']:.3f}x  "
              f"end_err_target={r['end_err_to_target_cm']:.3f}cm  "
              f"end_err_orig={r['end_err_to_orig_cm']:.3f}cm")
    all_results["plan_C_cumulative_ik"] = results
    env.env.close()

    f.close()

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 110)
    print("SUMMARY (averages over 5 demos; track = actual/desired for steps where desired>5mm)")
    print("=" * 110)
    print(f"{'Plan':<35} {'desired':<10} {'actual':<10} {'track_med':<10} "
          f"{'err_target':<14} {'err_orig':<12}")
    print("-" * 110)
    for plan, results in all_results.items():
        avg_desired = np.mean([r["desired_dpos_mag_mean_cm"] for r in results])
        avg_actual = np.mean([r["actual_dpos_mag_mean_cm"] for r in results])
        avg_ratio = np.nanmean([r['tracking_median'] for r in results])
        avg_err_target = np.mean([r["err_to_target_mean_cm"] for r in results])
        avg_err_orig = np.mean([r["end_err_to_orig_cm"] for r in results])
        print(f"{plan:<35} {avg_desired:>6.3f}cm   {avg_actual:>6.3f}cm   "
              f"{avg_ratio:>7.3f}x    {avg_err_target:>10.3f}cm   {avg_err_orig:>8.3f}cm")


if __name__ == "__main__":
    main()
