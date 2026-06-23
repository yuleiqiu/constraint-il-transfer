"""
Benchmark pointcloud computation overhead before/after static caching optimization.

Measures:
  1. render_obstacle_mask() time
  2. depth_mask_to_world_pointcloud() time
  3. Total per-call build_pointcloud_context_fields() time
  4. Simulated 400-step rollout: original (recompute every step) vs cached (compute once)
"""
import argparse
import time
import sys
import numpy as np

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvType
from robomimic.scripts.run_obstacle_guided_agent import build_pointcloud_context_fields
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils

OBS_MODALITIES = {
    "low_dim": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
        "object",
    ],
    "rgb": ["agentview_image"],
    "depth": ["agentview_depth"],
}

ENV_NAME = "PickPlaceBreadCan"
HORIZON = 400
WARMUP = 3
BENCHMARK_RUNS = 5


def make_env():
    ObsUtils.initialize_obs_modality_mapping_from_dict(OBS_MODALITIES)
    env = EnvUtils.create_env(
        env_type=EnvType.ROBOSUITE_TYPE,
        env_name=ENV_NAME,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=True,
        robots="Panda",
    )
    EnvUtils.set_env_specific_obs_processing(env=env)
    return env


def dummy_args():
    return argparse.Namespace(
        pc_depth_obs_key="agentview_depth",
        pc_camera_name="agentview",
        target_object_name="Can",
        obstacle_names=None,
        pc_voxel_size=0.005,
        pc_max_points=1024,
        pc_workspace_crop=True,
        pc_debug_visualization=False,
    )


def time_render_mask(env, args):
    depth = np.zeros((128, 128, 1), dtype=np.float32)
    t0 = time.perf_counter()
    ObstacleGuidanceUtils.render_obstacle_mask(
        env=env,
        camera_name=args.pc_camera_name,
        height=128,
        width=128,
        target_object_name=args.target_object_name,
        obstacle_names=args.obstacle_names,
    )
    return time.perf_counter() - t0


def time_full_step(env, obs, args, step_i):
    t0 = time.perf_counter()
    build_pointcloud_context_fields(env=env, obs=obs, args=args, step_i=step_i)
    return time.perf_counter() - t0


def run_benchmark():
    print("=" * 60)
    print("Pointcloud Overhead Benchmark")
    print("=" * 60)

    print("\n[1] Creating environment ...", flush=True)
    env = make_env()
    args = dummy_args()

    print("[2] Resetting environment ...", flush=True)
    obs = env.reset()
    state = env.get_state()
    obs = env.reset_to(state)

    print("[3] Checking required obs keys ...", flush=True)
    depth_key = args.pc_depth_obs_key
    if depth_key not in obs:
        print(f"  ERROR: depth key '{depth_key}' not in observations!")
        print(f"  Available keys: {list(obs.keys())}")
        del env
        return
    depth_val = np.asarray(obs[depth_key])
    print(f"  depth shape: {depth_val.shape}, dtype: {depth_val.dtype}")
    print(f"  depth range: [{depth_val.min():.4f}, {depth_val.max():.4f}]")

    print(f"\n[4] Warmup ({WARMUP} runs) ...", flush=True)
    for _ in range(WARMUP):
        build_pointcloud_context_fields(env=env, obs=obs, args=args, step_i=0)

    print(f"\n[5] Benchmark render_obstacle_mask ({BENCHMARK_RUNS} runs) ...", flush=True)
    mask_times = []
    for r in range(BENCHMARK_RUNS):
        t = time_render_mask(env, args)
        mask_times.append(t)
        print(f"  run {r+1}: {t*1000:.2f} ms", flush=True)

    print(f"\n[6] Benchmark full step ({BENCHMARK_RUNS} runs) ...", flush=True)
    full_times = []
    point_counts = []
    for r in range(BENCHMARK_RUNS):
        t = time_full_step(env, obs, args, step_i=0)
        full_times.append(t)
        print(f"  run {r+1}: {t*1000:.2f} ms", flush=True)

    raw_full_time = time_full_step(env, obs, args, step_i=0)
    mask_time = time_render_mask(env, args)
    backproject_time = raw_full_time - mask_time

    print(f"\n[7] Simulated rollout overhead (horizon={HORIZON}) ...", flush=True)

    time_per_step = np.mean(full_times)
    time_per_mask = np.mean(mask_times)

    original_total = time_per_step * HORIZON
    cached_total = time_per_step * 1.0

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  render_obstacle_mask()      :  {time_per_mask*1000:7.2f} ms  ({time_per_mask/time_per_step*100:5.1f}% of total)")
    print(f"  depth→world backprojection  :  {backproject_time*1000:7.2f} ms  ({backproject_time/time_per_step*100:5.1f}% of total)")
    print(f"  per-step total              :  {time_per_step*1000:7.2f} ms  (100.0%)")
    print(f"  ----")
    print(f"  Original ({HORIZON} recomputes)   : {original_total*1000:7.1f} ms")
    print(f"  Cached   (1 recompute)       :  {cached_total*1000:7.1f} ms")
    print(f"  Time saved                   : {(original_total - cached_total)*1000:7.1f} ms ({100*(1 - cached_total/original_total):.1f}%)")
    print(f"{'='*60}")

    del env


if __name__ == "__main__":
    run_benchmark()
