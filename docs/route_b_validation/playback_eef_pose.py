"""
Playback dataset demos using absolute EEF pose as the control signal.

Constructs OSC absolute-mode actions from next_obs/robot0_eef_pos
and next_obs/robot0_eef_quat_site, then replays them open-loop.
Reports per-step state divergence and EEF tracking error.

Args:
    --dataset        Path to hdf5 dataset
    --demo           Demo key to playback (default: demo_1)
    --all-demos      Playback all demos in the dataset
    --osc-kp         OSC controller Kp (default: 500)
    --osc-input-type Controller input type: "delta" | "absolute" (default: "absolute")
    --video-path     Optional output mp4 path

Example:
    cd docs/route_b_validation
    MUJOCO_GL=egl uv run python playback_eef_pose.py \
        --dataset ../../third_party/robomimic/datasets/can/yq/image_v15.hdf5 \
        --demo demo_1 --osc-kp 500 --osc-input-type absolute
"""

import argparse
import json
import os
import h5py
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robosuite.utils.transform_utils as T
import robosuite.utils.control_utils as CU


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def quat_angular_distance(q1_xyzw, q2_xyzw):
    """Angular distance (rad) between two quaternions [x,y,z,w]."""
    m1 = T.quat2mat(q1_xyzw)
    m2 = T.quat2mat(q2_xyzw)
    return np.linalg.norm(CU.orientation_error(m2, m1))


def sync_arm_controllers(env):
    """Refresh controller references after reset_to writes a flattened sim state."""
    for ctrl in env.env.robots[0].part_controllers.values():
        if hasattr(ctrl, "update"):
            ctrl.update(force=True)
        if hasattr(ctrl, "reset_goal"):
            ctrl.reset_goal()


def build_eef_actions(grp, num_steps):
    """Pre-construct a (num_steps, 7) array of OSC absolute-mode actions.

    Returns:
        np.ndarray shape (T, 7): [pos_x, pos_y, pos_z, rotvec_x, rotvec_y, rotvec_z, grip]
    """
    eef_pos = grp["next_obs/robot0_eef_pos"][:]
    eef_quat_site = grp["next_obs/robot0_eef_quat_site"][:]  # [x,y,z,w]
    grip = grp["actions"][:, 6]

    actions = np.zeros((num_steps, 7), dtype=np.float64)
    for i in range(num_steps):
        q = eef_quat_site[i]
        rotvec = T.quat2axisangle(q.copy())
        actions[i] = np.concatenate([eef_pos[i], rotvec, [grip[i]]])
    return actions


def sorted_demo_keys(data_grp):
    return sorted(data_grp.keys(), key=lambda x: int(x.split("_")[-1]))


def task_success(env):
    return bool(env.is_success().get("task", False))


def initial_state_from_demo(grp, states, is_robosuite_env):
    initial_state = dict(states=states[0])
    if is_robosuite_env:
        initial_state["model"] = grp.attrs["model_file"]
        initial_state["ep_meta"] = grp.attrs.get("ep_meta", None)
    return initial_state


def render_video_frame(env, camera_names, height, width):
    frames = [
        env.render(mode="rgb_array", height=height, width=width, camera_name=camera_name)
        for camera_name in camera_names
    ]
    return np.concatenate(frames, axis=1)


def replay_demo(
    env,
    demo_name,
    grp,
    is_robosuite_env,
    verbose=False,
    check_state=True,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    video_height=512,
    video_width=512,
):
    states = grp["states"][:]
    num_steps = states.shape[0]
    actions = build_eef_actions(grp, num_steps)

    env.reset_to(initial_state_from_demo(grp, states, is_robosuite_env))
    sync_arm_controllers(env)

    state_errs = []
    pos_errs = []
    ori_errs = []
    any_success = task_success(env)

    video_count = 0
    for i in range(num_steps):
        obs, _, _, _ = env.step(actions[i])
        any_success = any_success or task_success(env)

        if video_writer is not None and video_count % video_skip == 0:
            video_writer.append_data(render_video_frame(env, camera_names, video_height, video_width))
        video_count += 1

        if check_state and i < num_steps - 1:
            replay_state = env.get_state()["states"]
            state_errs.append(float(np.linalg.norm(states[i + 1] - replay_state)))

        pos_err = np.linalg.norm(obs["robot0_eef_pos"] - grp["next_obs/robot0_eef_pos"][i])
        pos_errs.append(float(pos_err))

        if "robot0_eef_quat_site" in obs:
            target_q = grp["next_obs/robot0_eef_quat_site"][i]     # [x,y,z,w]
            achieved_q = obs["robot0_eef_quat_site"]               # [x,y,z,w]
            ori_err = quat_angular_distance(achieved_q, target_q)
            ori_errs.append(float(ori_err * 180 / np.pi))

        detail = verbose and (i < 10 or i % 50 == 0)
        if detail:
            print(f"  step {i:>4}: state_err={state_errs[-1] if state_errs else 0:.4f}  "
                  + (f"pos_err={pos_errs[-1]*100 if pos_errs else 0:.1f}cm  "
                     f"ori_err={ori_errs[-1] if ori_errs else 0:.1f}deg"))

    final_success = task_success(env)
    return {
        "demo": demo_name,
        "steps": int(num_steps),
        "any_success": bool(any_success),
        "final_success": bool(final_success),
        "state_err_mean": float(np.mean(state_errs)) if state_errs else None,
        "state_err_max": float(np.max(state_errs)) if state_errs else None,
        "pos_err_mean_cm": float(np.mean(pos_errs) * 100),
        "pos_err_max_cm": float(np.max(pos_errs) * 100),
        "pos_err_end_cm": float(pos_errs[-1] * 100),
        "ori_err_mean_deg": float(np.mean(ori_errs)) if ori_errs else None,
        "ori_err_max_deg": float(np.max(ori_errs)) if ori_errs else None,
        "ori_err_end_deg": float(ori_errs[-1]) if ori_errs else None,
    }


def summarize_results(results):
    n = len(results)
    return {
        "num_demos": n,
        "num_steps": int(sum(r["steps"] for r in results)),
        "any_success_count": int(sum(r["any_success"] for r in results)),
        "any_success_rate": float(np.mean([r["any_success"] for r in results])) if n else 0.0,
        "final_success_count": int(sum(r["final_success"] for r in results)),
        "final_success_rate": float(np.mean([r["final_success"] for r in results])) if n else 0.0,
        "pos_err_mean_cm": float(np.mean([r["pos_err_mean_cm"] for r in results])) if n else 0.0,
        "pos_err_max_cm": float(np.max([r["pos_err_max_cm"] for r in results])) if n else 0.0,
        "pos_err_end_mean_cm": float(np.mean([r["pos_err_end_cm"] for r in results])) if n else 0.0,
        "ori_err_mean_deg": float(np.mean([r["ori_err_mean_deg"] for r in results])) if n else 0.0,
        "ori_err_max_deg": float(np.max([r["ori_err_max_deg"] for r in results])) if n else 0.0,
        "ori_err_end_mean_deg": float(np.mean([r["ori_err_end_deg"] for r in results])) if n else 0.0,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to hdf5 dataset")
    parser.add_argument("--demo", type=str, default="demo_1", help="Demo key to playback")
    parser.add_argument("--all-demos", action="store_true", help="Playback all demos in the dataset")
    parser.add_argument("--n-demos", type=int, default=None, help="Optional cap on number of demos")
    parser.add_argument("--osc-kp", type=float, default=500.0, help="OSC controller Kp")
    parser.add_argument("--osc-input-type", type=str, default="absolute",
                        choices=["delta", "absolute"], help="OSC input_type")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path for per-demo results")
    parser.add_argument("--video-path", type=str, default=None, help="Optional output video path")
    parser.add_argument("--video-skip", type=int, default=5, help="Render every N env steps")
    parser.add_argument("--render-image-names", type=str, nargs="+", default=None,
                        help="Camera names to render. Defaults to agentview.")
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--video-width", type=int, default=512)
    args = parser.parse_args()
    write_video = args.video_path is not None
    camera_names = args.render_image_names or ["agentview"]

    # ---- 1. Init ObsUtils (dummy spec, observations unused in playback) ----
    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs=dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=[]))
    )

    # ---- 2. Load env metadata and override OSC controller params ----
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env_meta = json.loads(json.dumps(env_meta))  # deep-copy so we can mutate safely

    for bp in env_meta["env_kwargs"]["controller_configs"]["body_parts"].values():
        if bp.get("type") == "OSC_POSE":
            bp["kp"] = args.osc_kp
            bp["input_type"] = args.osc_input_type
            if args.osc_input_type == "absolute":
                bp["input_ref_frame"] = "world"

    # ---- 3. Create environment ----
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=write_video,
    )

    # ---- 4. Open dataset, load demo data ----
    f = h5py.File(args.dataset, "r")
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    if args.all_demos:
        demo_names = sorted_demo_keys(f["data"])
    else:
        demo_names = [args.demo]
    if args.n_demos is not None:
        demo_names = demo_names[:args.n_demos]

    print(f"Demos: {len(demo_names)}")
    print(f"OSC  : kp={args.osc_kp}, input_type={args.osc_input_type}")
    if write_video:
        os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
        print(f"Video: {args.video_path}  cameras={camera_names}  skip={args.video_skip}")

    # ---- 5. Playback open-loop ----
    results = []
    video_writer = imageio.get_writer(args.video_path, fps=20) if write_video else None
    try:
        for demo_i, demo_name in enumerate(demo_names, start=1):
            verbose = not args.all_demos
            result = replay_demo(
                env=env,
                demo_name=demo_name,
                grp=f[f"data/{demo_name}"],
                is_robosuite_env=is_robosuite_env,
                verbose=verbose,
                check_state=not args.all_demos,
                video_writer=video_writer,
                video_skip=args.video_skip,
                camera_names=camera_names,
                video_height=args.video_height,
                video_width=args.video_width,
            )
            results.append(result)
            print(
                f"[{demo_i:>3}/{len(demo_names)}] {demo_name}: "
                f"any_success={int(result['any_success'])} "
                f"final_success={int(result['final_success'])} "
                f"pos_mean={result['pos_err_mean_cm']:.2f}cm "
                f"pos_max={result['pos_err_max_cm']:.2f}cm "
                f"ori_mean={result['ori_err_mean_deg']:.2f}deg"
            )
    finally:
        if video_writer is not None:
            video_writer.close()

    # ---- 6. Summary ----
    summary = summarize_results(results)
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Demos            : {summary['num_demos']} ({summary['num_steps']} steps)")
    print(
        f"  Any success      : {summary['any_success_count']}/{summary['num_demos']} "
        f"({summary['any_success_rate'] * 100:.1f}%)"
    )
    print(
        f"  Final success    : {summary['final_success_count']}/{summary['num_demos']} "
        f"({summary['final_success_rate'] * 100:.1f}%)"
    )
    print(f"  Pos error (cm)   : mean={summary['pos_err_mean_cm']:.2f}  max={summary['pos_err_max_cm']:.2f}  end_mean={summary['pos_err_end_mean_cm']:.2f}")
    print(f"  Ori error (deg)  : mean={summary['ori_err_mean_deg']:.2f}  max={summary['ori_err_max_deg']:.2f}  end_mean={summary['ori_err_end_mean_deg']:.2f}")

    if args.output_json is not None:
        with open(args.output_json, "w") as fp:
            json.dump({"summary": summary, "results": results}, fp, indent=2)
        print(f"  Wrote results    : {args.output_json}")

    f.close()
    env.env.close()


if __name__ == "__main__":
    main()
