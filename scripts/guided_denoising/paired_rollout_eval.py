"""Run a small matched baseline versus guided rollout comparison."""

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import common  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


def state_hash(state):
    digest = hashlib.sha256()
    digest.update(np.asarray(state["states"]).tobytes())
    digest.update(state["model"].encode("utf-8"))
    digest.update(state.get("ep_meta", "").encode("utf-8"))
    return digest.hexdigest()


def rollout_condition(
    policy,
    env,
    initial_state,
    condition,
    guidance_scale,
    policy_seed,
    args,
):
    policy.start_episode()
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    obs = env.reset_to(initial_state)
    torch.manual_seed(policy_seed)
    raw_env = common.raw_env_from_wrapper(env)
    records = common.active_object_records(raw_env)
    maps = common.contact_maps(raw_env, records, args.target_object_name)
    target_start = common.target_object_position(raw_env, records, args.target_object_name)
    low, high = common.action_limits(env, policy.policy.ac_dim)

    step = 0
    chunks = 0
    success = common.task_success(env)
    collision_any = False
    collision_names = set()
    target_contact_any = False
    clip_count = 0
    clip_max = 0.0
    trigger_steps = 0
    reverse_steps = 0
    update_norms = []
    reconstruction_errors = []
    min_clearance = float("inf")

    while step < args.horizon and not success:
        centers, radii, _ = common.oracle_obstacle_geometry(
            raw_env, records, args.target_object_name
        )
        scale = 0.0 if condition == "baseline" else guidance_scale
        context = common.make_guidance_context(
            policy,
            obs,
            centers,
            radii,
            scale=scale,
            clearance_margin=args.clearance_margin,
        )
        start_position = common.current_eef_position(obs)
        _, raw_chunk, diagnostics = common.sample_action_chunk(
            policy, obs, context, noise_seed=None
        )
        predicted = common.predict_eef_positions(start_position, raw_chunk)
        clip_delta = np.abs(np.clip(raw_chunk, low, high) - raw_chunk)
        clip_count += int(np.count_nonzero(clip_delta > 1e-5))
        clip_max = max(clip_max, float(np.max(clip_delta)))
        trigger_steps += sum(item["active_penetration_count"] > 0 for item in diagnostics)
        reverse_steps += len(diagnostics)
        update_norms.extend(item["normalized_applied_update_norm"] for item in diagnostics)

        actual = []
        for action in raw_chunk:
            if step >= args.horizon:
                break
            obs, _, done, _ = env.step(action)
            position = common.current_eef_position(obs)
            actual.append(position)
            min_clearance = min(
                min_clearance,
                common.minimum_clearance(position, centers, radii),
            )
            target_contacts, obstacle_contacts, names = common.contact_snapshot(raw_env, maps)
            target_contact_any = target_contact_any or target_contacts > 0
            collision_any = collision_any or obstacle_contacts > 0
            collision_names.update(names)
            success = success or common.task_success(env)
            step += 1
            if done or (args.terminate_on_success and success):
                break

        actual = np.asarray(actual, dtype=np.float32)
        if len(actual):
            reconstruction_errors.extend(
                np.linalg.norm(predicted[: len(actual)] - actual, axis=1).tolist()
            )
        chunks += 1
        if len(actual) == 0 or done or (args.terminate_on_success and success):
            break

    target_end = common.target_object_position(raw_env, records, args.target_object_name)
    safe_success = success and not collision_any
    success_with_collision = success and collision_any
    collision_failure = (not success) and collision_any
    ncr = (not success) and (not collision_any)
    return {
        "condition": condition,
        "success": success,
        "horizon": step,
        "chunks": chunks,
        "collision_any": collision_any,
        "collision_names": sorted(collision_names),
        "target_contact_any": target_contact_any,
        "safe_success": safe_success,
        "success_with_collision": success_with_collision,
        "collision_failure": collision_failure,
        "ncr": ncr,
        "min_eef_obstacle_clearance_m": min_clearance,
        "target_displacement_m": float(np.linalg.norm(target_end - target_start)),
        "action_clip_count": clip_count,
        "action_clip_max_abs": clip_max,
        "guidance_trigger_rate": trigger_steps / reverse_steps if reverse_steps else 0.0,
        "guidance_update_norm_mean": float(np.mean(update_norms)) if update_norms else 0.0,
        "guidance_update_norm_max": float(np.max(update_norms)) if update_norms else 0.0,
        "trajectory_reconstruction_error_mean_cm": float(
            np.mean(reconstruction_errors) * 100.0
        )
        if reconstruction_errors
        else None,
        "trajectory_reconstruction_error_max_cm": float(
            np.max(reconstruction_errors) * 100.0
        )
        if reconstruction_errors
        else None,
    }


def aggregate(rows):
    def mean_present(selected, key):
        values = [row[key] for row in selected if row[key] is not None]
        return float(np.mean(values)) if values else None

    def max_present(selected, key):
        values = [row[key] for row in selected if row[key] is not None]
        return float(np.max(values)) if values else None

    result = []
    keys = sorted({(row["env"], row["condition"]) for row in rows})
    for env_name, condition in keys:
        selected = [
            row for row in rows if row["env"] == env_name and row["condition"] == condition
        ]
        result.append(
            {
                "env": env_name,
                "condition": condition,
                "episodes": len(selected),
                "task_sr": float(np.mean([row["success"] for row in selected])),
                "safe_sr": float(np.mean([row["safe_success"] for row in selected])),
                "cr": float(np.mean([row["collision_any"] for row in selected])),
                "ncr": float(np.mean([row["ncr"] for row in selected])),
                "safe_success": int(sum(row["safe_success"] for row in selected)),
                "success_with_collision": int(
                    sum(row["success_with_collision"] for row in selected)
                ),
                "collision_failure": int(sum(row["collision_failure"] for row in selected)),
                "collision_free_non_completion": int(sum(row["ncr"] for row in selected)),
                "guidance_trigger_rate": float(
                    np.mean([row["guidance_trigger_rate"] for row in selected])
                ),
                "guidance_update_norm_mean": float(
                    np.mean([row["guidance_update_norm_mean"] for row in selected])
                ),
                "guidance_update_norm_max": float(
                    np.max([row["guidance_update_norm_max"] for row in selected])
                ),
                "action_clip_count": int(sum(row["action_clip_count"] for row in selected)),
                "trajectory_reconstruction_error_mean_cm": mean_present(
                    selected, "trajectory_reconstruction_error_mean_cm"
                ),
                "trajectory_reconstruction_error_max_cm": max_present(
                    selected, "trajectory_reconstruction_error_max_cm"
                ),
            }
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=common.DEFAULT_AGENT)
    parser.add_argument("--envs", nargs="+", default=common.DEFAULT_ENVS)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--clearance-margin", type=float, default=0.02)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument(
        "--terminate-on-success", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/guided_denoising/paired_pilot"),
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
    rows = []
    jsonl_path = args.out_dir / "episode_metrics.jsonl"
    with jsonl_path.open("w") as output:
        for env_name in args.envs:
            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=ckpt,
                env_name=env_name,
                render=False,
                render_offscreen=False,
                verbose=False,
            )
            for seed in args.seeds:
                for episode in range(args.n_rollouts):
                    layout_seed = seed * 1000 + episode
                    policy_seed = seed * 1_000_000 + episode
                    random.seed(layout_seed)
                    np.random.seed(layout_seed)
                    env.reset()
                    initial_state = env.get_state()
                    initial_hash = state_hash(initial_state)
                    for condition in ("baseline", "guided"):
                        metrics = rollout_condition(
                            policy,
                            env,
                            initial_state,
                            condition,
                            args.guidance_scale,
                            policy_seed,
                            args,
                        )
                        row = {
                            "env": env_name,
                            "seed": seed,
                            "episode": episode,
                            "layout_seed": layout_seed,
                            "policy_seed": policy_seed,
                            "initial_state_hash": initial_hash,
                            **metrics,
                        }
                        rows.append(row)
                        output.write(json.dumps(common.json_safe(row)) + "\n")
                        output.flush()
                        print(
                            "{} seed={} episode={} {} success={} collision={}".format(
                                env_name,
                                seed,
                                episode,
                                condition,
                                row["success"],
                                row["collision_any"],
                            ),
                            flush=True,
                        )

    summary = aggregate(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(common.json_safe(summary), indent=2) + "\n"
    )
    manifest = {
        "agent": str(args.agent),
        "envs": args.envs,
        "seeds": args.seeds,
        "n_rollouts": args.n_rollouts,
        "horizon": args.horizon,
        "guidance_scale": args.guidance_scale,
        "clearance_margin": args.clearance_margin,
        "terminate_on_success": args.terminate_on_success,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("wrote {}".format(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
