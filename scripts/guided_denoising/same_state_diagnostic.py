"""Capture fixed states and sweep delta-EEF guided-denoising scales.

The default capture mode retains chunk-start states whose unguided predicted
trajectory activates the deployment cost and whose executed prefix contacts a
non-target object before any target contact.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import common  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def write_state(group, state_id, payload):
    item = group.create_group(state_id)
    for key in ("env", "seed", "episode", "chunk", "start_step", "noise_seed"):
        item.attrs[key] = payload[key]
    item.create_dataset("sim_state", data=payload["state"]["states"])
    item.create_dataset("model_xml", data=payload["state"]["model"], dtype=STRING_DTYPE)
    item.create_dataset(
        "ep_meta",
        data=payload["state"].get("ep_meta", ""),
        dtype=STRING_DTYPE,
    )
    obs_group = item.create_group("obs")
    for key, value in payload["obs"].items():
        kwargs = {"compression": "gzip"} if np.asarray(value).ndim >= 3 else {}
        obs_group.create_dataset(key, data=value, **kwargs)
    item.create_dataset("obstacle_centers", data=payload["centers"])
    item.create_dataset("obstacle_radii", data=payload["radii"])
    item.create_dataset(
        "obstacle_names",
        data=np.asarray(payload["names"], dtype=object),
        dtype=STRING_DTYPE,
    )


def read_state(item):
    def text_value(name):
        value = item[name][()]
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    ep_meta = text_value("ep_meta")
    return {
        "id": item.name.rsplit("/", 1)[-1],
        "env": str(item.attrs["env"]),
        "seed": int(item.attrs["seed"]),
        "episode": int(item.attrs["episode"]),
        "chunk": int(item.attrs["chunk"]),
        "start_step": int(item.attrs["start_step"]),
        "noise_seed": int(item.attrs["noise_seed"]),
        "state": {
            "states": item["sim_state"][()],
            "model": text_value("model_xml"),
            **({"ep_meta": ep_meta} if ep_meta else {}),
        },
        "obs": {key: dataset[()] for key, dataset in item["obs"].items()},
    }


def capture_states(policy, ckpt, args, states_path):
    counts = {env_name: 0 for env_name in args.envs}
    with h5py.File(states_path, "w") as output:
        states_group = output.create_group("states")
        for env_name in args.envs:
            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=ckpt,
                env_name=env_name,
                render=False,
                render_offscreen=False,
                verbose=False,
            )
            for seed in args.capture_seeds:
                if counts[env_name] >= args.states_per_env:
                    break
                for episode in range(args.max_episodes_per_seed):
                    if counts[env_name] >= args.states_per_env:
                        break
                    common.seed_everything(seed * 1000 + episode)
                    policy.start_episode()
                    obs = env.reset()
                    records = common.active_object_records(common.raw_env_from_wrapper(env))
                    maps = common.contact_maps(
                        common.raw_env_from_wrapper(env), records, args.target_object_name
                    )
                    target_seen = False
                    step = 0
                    chunk_index = 0
                    done = False
                    while step < args.horizon and not done:
                        start_step = step
                        state_before = env.get_state()
                        obs_before = common.policy_observation(obs, policy)
                        raw_env = common.raw_env_from_wrapper(env)
                        centers, radii, names = common.oracle_obstacle_geometry(
                            raw_env, records, args.target_object_name
                        )
                        noise_seed = seed * 1_000_000 + episode * 1000 + chunk_index
                        context = common.make_guidance_context(
                            policy,
                            obs_before,
                            centers,
                            radii,
                            scale=0.0,
                            clearance_margin=args.clearance_margin,
                        )
                        _, raw_chunk, diagnostics = common.sample_action_chunk(
                            policy, obs_before, context, noise_seed=noise_seed
                        )
                        active_prediction = any(
                            item["active_penetration_count"] > 0 for item in diagnostics
                        )

                        collision_in_prefix = False
                        target_before_prefix = target_seen
                        for action in raw_chunk:
                            if step >= args.horizon:
                                break
                            obs, _, done, _ = env.step(action)
                            target_contacts, obstacle_contacts, _ = common.contact_snapshot(
                                raw_env, maps
                            )
                            target_seen = target_seen or target_contacts > 0
                            collision_in_prefix = collision_in_prefix or obstacle_contacts > 0
                            step += 1
                            if done or common.task_success(env):
                                done = True
                                break

                        eligible = active_prediction
                        if args.candidate_mode == "collision":
                            eligible = (
                                eligible
                                and collision_in_prefix
                                and not target_before_prefix
                            )
                        if eligible:
                            state_id = "{}_state_{:03d}".format(
                                env_name, counts[env_name]
                            )
                            write_state(
                                states_group,
                                state_id,
                                {
                                    "env": env_name,
                                    "seed": seed,
                                    "episode": episode,
                                    "chunk": chunk_index,
                                    "start_step": start_step,
                                    "noise_seed": noise_seed,
                                    "state": state_before,
                                    "obs": obs_before,
                                    "centers": centers,
                                    "radii": radii,
                                    "names": names,
                                },
                            )
                            counts[env_name] += 1
                            print("captured {}".format(state_id), flush=True)
                            if counts[env_name] >= args.states_per_env:
                                break
                        chunk_index += 1
    return counts


def sweep_states(policy, ckpt, args, states_path, results_path):
    envs = {}
    rows = []
    try:
        with h5py.File(states_path, "r") as states, results_path.open("w") as output:
            for state_id in sorted(states["states"]):
                saved = read_state(states["states"][state_id])
                env_name = saved["env"]
                if env_name not in envs:
                    envs[env_name], _ = FileUtils.env_from_checkpoint(
                        ckpt_dict=ckpt,
                        env_name=env_name,
                        render=False,
                        render_offscreen=False,
                        verbose=False,
                    )
                env = envs[env_name]
                for scale in args.scales:
                    policy.start_episode()
                    env.reset_to(saved["state"])
                    raw_env = common.raw_env_from_wrapper(env)
                    records = common.active_object_records(raw_env)
                    maps = common.contact_maps(raw_env, records, args.target_object_name)
                    centers, radii, names = common.oracle_obstacle_geometry(
                        raw_env, records, args.target_object_name
                    )
                    before_target = common.target_object_position(
                        raw_env, records, args.target_object_name
                    )
                    before_progress = common.staged_task_progress(raw_env)
                    start_position = common.current_eef_position(saved["obs"])
                    context = common.make_guidance_context(
                        policy,
                        saved["obs"],
                        centers,
                        radii,
                        scale=scale,
                        clearance_margin=args.clearance_margin,
                    )
                    normalized, raw_chunk, diagnostics = common.sample_action_chunk(
                        policy,
                        saved["obs"],
                        context,
                        noise_seed=saved["noise_seed"],
                    )
                    predicted = common.predict_eef_positions(start_position, raw_chunk)
                    low, high = common.action_limits(env, raw_chunk.shape[-1])
                    clipped = np.clip(raw_chunk, low, high)
                    clip_delta = np.abs(clipped - raw_chunk)

                    actual = []
                    actual_clearance = []
                    collision_any = False
                    target_contact_any = False
                    success = common.task_success(env)
                    for action in raw_chunk:
                        next_obs, _, done, _ = env.step(action)
                        position = common.current_eef_position(next_obs)
                        actual.append(position)
                        actual_clearance.append(
                            common.minimum_clearance(position, centers, radii)
                        )
                        target_contacts, obstacle_contacts, _ = common.contact_snapshot(
                            raw_env, maps
                        )
                        target_contact_any = target_contact_any or target_contacts > 0
                        collision_any = collision_any or obstacle_contacts > 0
                        success = success or common.task_success(env)
                        if done:
                            break

                    actual = np.asarray(actual, dtype=np.float32)
                    predicted_executed = predicted[: len(actual)]
                    reconstruction_error = (
                        np.linalg.norm(predicted_executed - actual, axis=1)
                        if len(actual)
                        else np.zeros((0,), dtype=np.float32)
                    )
                    after_target = common.target_object_position(
                        raw_env, records, args.target_object_name
                    )
                    row = {
                        "state_id": state_id,
                        "env": env_name,
                        "scale": float(scale),
                        "noise_seed": saved["noise_seed"],
                        "obstacle_names": names,
                        "predicted_min_clearance_m": min(
                            common.minimum_clearance(point, centers, radii)
                            for point in predicted
                        ),
                        "predicted_safety_clearance_m": min(
                            common.minimum_clearance(
                                point, centers, radii, args.clearance_margin
                            )
                            for point in predicted
                        ),
                        "actual_min_clearance_m": min(actual_clearance)
                        if actual_clearance
                        else None,
                        "collision_any": collision_any,
                        "target_contact_any": target_contact_any,
                        "success": success,
                        "action_clip_count": int(np.count_nonzero(clip_delta > 1e-5)),
                        "action_clip_max_abs": float(np.max(clip_delta)),
                        "normalized_action_l2_norm": float(np.linalg.norm(normalized)),
                        "reconstruction_error_mean_cm": float(
                            np.mean(reconstruction_error) * 100.0
                        )
                        if len(reconstruction_error)
                        else None,
                        "reconstruction_error_max_cm": float(
                            np.max(reconstruction_error) * 100.0
                        )
                        if len(reconstruction_error)
                        else None,
                        "target_displacement_m": float(
                            np.linalg.norm(after_target - before_target)
                        ),
                        "task_progress_before": before_progress,
                        "task_progress_after": common.staged_task_progress(raw_env),
                        "guidance_trigger_steps": int(
                            sum(item["active_penetration_count"] > 0 for item in diagnostics)
                        ),
                        "guidance_update_norm_max": float(
                            max(item["normalized_applied_update_norm"] for item in diagnostics)
                        ),
                        "physical_waypoint_displacement_max_m": float(
                            max(item["max_waypoint_displacement_m"] for item in diagnostics)
                        ),
                        "reverse_steps": diagnostics,
                        "raw_action_chunk": raw_chunk,
                        "predicted_eef_positions": predicted,
                        "actual_eef_positions": actual,
                    }
                    rows.append(row)
                    output.write(json.dumps(common.json_safe(row)) + "\n")
                    output.flush()
                    print(
                        "swept {} scale={} pred_clearance={:.4f} actual_clearance={}".format(
                            state_id,
                            scale,
                            row["predicted_min_clearance_m"],
                            row["actual_min_clearance_m"],
                        ),
                        flush=True,
                    )
    finally:
        envs.clear()
    return rows


def write_summary(path, rows, counts, args):
    baseline_by_state = {
        row["state_id"]: row for row in rows if row["scale"] == 0.0
    }
    by_scale = []
    for scale in args.scales:
        selected = [row for row in rows if row["scale"] == float(scale)]
        if not selected:
            continue
        raw_action_changes = [
            np.linalg.norm(
                np.asarray(row["raw_action_chunk"])
                - np.asarray(baseline_by_state[row["state_id"]]["raw_action_chunk"])
            )
            for row in selected
        ]
        summary = {
            "scale": float(scale),
            "states": len(selected),
            "mean_predicted_min_clearance_m": float(
                np.mean([row["predicted_min_clearance_m"] for row in selected])
            ),
            "mean_actual_min_clearance_m": float(
                np.mean(
                    [
                        row["actual_min_clearance_m"]
                        for row in selected
                        if row["actual_min_clearance_m"] is not None
                    ]
                )
            ),
            "collision_rate": float(
                np.mean([row["collision_any"] for row in selected])
            ),
            "action_clip_count": int(
                sum(row["action_clip_count"] for row in selected)
            ),
            "max_guidance_update_norm": float(
                max(row["guidance_update_norm_max"] for row in selected)
            ),
            "max_physical_waypoint_displacement_m": float(
                max(row["physical_waypoint_displacement_max_m"] for row in selected)
            ),
            "mean_raw_action_l2_change_from_scale_zero": float(
                np.mean(raw_action_changes)
            ),
        }
        by_scale.append(summary)

    payload = {
        "captured_states": counts,
        "candidate_mode": args.candidate_mode,
        "clearance_margin": args.clearance_margin,
        "scales": args.scales,
        "summary_by_scale": by_scale,
        "selected_scale": None,
        "selection_note": (
            "Record the largest scale that improves predicted and executed clearance "
            "without implausible waypoint displacement or action clipping before the paired pilot."
        ),
    }
    path.write_text(json.dumps(common.json_safe(payload), indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=common.DEFAULT_AGENT)
    parser.add_argument("--envs", nargs="+", default=common.DEFAULT_ENVS)
    parser.add_argument("--capture-seeds", nargs="+", type=int, default=[600, 601, 602])
    parser.add_argument("--states-per-env", type=int, default=5)
    parser.add_argument("--max-episodes-per-seed", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.001, 0.003, 0.01, 0.03])
    parser.add_argument("--clearance-margin", type=float, default=0.02)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--candidate-mode", choices=("collision", "active"), default="collision")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/guided_denoising/same_state"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError("{} already exists; pass --overwrite".format(args.out_dir))
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt = common.load_guided_policy_and_checkpoint(args.agent, device)
    states_path = args.out_dir / "states.hdf5"
    counts = capture_states(policy, ckpt, args, states_path)
    rows = sweep_states(
        policy,
        ckpt,
        args,
        states_path,
        args.out_dir / "sweep_results.jsonl",
    )
    write_summary(args.out_dir / "summary.json", rows, counts, args)
    manifest = {
        "agent": str(args.agent),
        "envs": args.envs,
        "capture_seeds": args.capture_seeds,
        "states_per_env": args.states_per_env,
        "horizon": args.horizon,
        "candidate_mode": args.candidate_mode,
        "scales": args.scales,
        "clearance_margin": args.clearance_margin,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("wrote {}".format(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
