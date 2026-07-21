"""Run paired baseline / sample / refine / sample-refine rollout pilots."""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import common  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


CONDITIONS = {
    "baseline": (False, False),
    "sample_only": (True, False),
    "refine_only": (False, True),
    "sample_refine": (True, True),
}


def warmup_policy(policy, env, args):
    """Exclude one-time CUDA and convolution setup from reported inference time."""

    policy.start_episode()
    obs = env.reset()
    context = common.make_context(
        policy,
        env,
        common.policy_observation(obs, policy),
        candidate_count=1,
        margin=args.margin,
        refine_timestep=args.refine_timestep,
        guidance_scale=0.0,
        candidate_seed=0,
        refinement_seed=1,
        sample_enabled=False,
        refinement_enabled=False,
        target_object_name=args.target_object_name,
        eef_radius=args.eef_radius,
    )
    common.sample_action_chunk(policy, common.policy_observation(obs, policy), context)


def rollout_condition(policy, env, initial_state, condition, policy_seed, args):
    sample_enabled, refinement_enabled = CONDITIONS[condition]
    common.seed_everything(policy_seed)
    policy.start_episode()
    obs = env.reset_to(initial_state)
    raw_env = common.raw_env_from_wrapper(env)
    records = common.active_object_records(raw_env)
    maps = common.contact_maps(raw_env, records, args.target_object_name)
    target_start = common.object_position(
        raw_env, common.target_record(raw_env, records, args.target_object_name)
    )
    low, high = common.action_limits(env, policy.policy.ac_dim)

    step = chunks = 0
    success = common.task_success(env)
    collisions = set()
    target_contact_any = False
    action_clip_count = 0
    inference_ms = []
    selected_changed = positive_safe_coverage = 0
    refine_triggered = refine_accepted = refine_fallback = 0
    initial_regime = None
    pregrasp_regimes = []
    done = False

    while step < args.horizon and not success:
        candidate_seed = policy_seed + chunks * 2
        context = common.make_context(
            policy,
            env,
            obs,
            candidate_count=args.candidate_count,
            margin=args.margin,
            refine_timestep=args.refine_timestep,
            guidance_scale=args.guidance_scale,
            candidate_seed=candidate_seed,
            refinement_seed=candidate_seed + 1,
            sample_enabled=sample_enabled,
            refinement_enabled=refinement_enabled,
            target_object_name=args.target_object_name,
            eef_radius=args.eef_radius,
        )
        _, raw_chunk, diagnostics = common.sample_action_chunk(
            policy, common.policy_observation(obs, policy), context
        )
        candidate_progress = diagnostics["candidate_progress_m"]
        candidate_clearance = diagnostics["candidate_clearance_m"]
        if candidate_progress:
            chunk_regime = common.regime(candidate_progress[0], candidate_clearance[0])
            if initial_regime is None:
                initial_regime = chunk_regime
            if not context.grasped:
                pregrasp_regimes.append(chunk_regime)
            positive_safe_coverage += int(
                any(p > 0 and c >= 0 for p, c in zip(candidate_progress, candidate_clearance))
            )
        selected_changed += int(diagnostics["selected_index"] != 0)
        inference_ms.append(diagnostics["inference_time_ms"])
        refine_triggered += int(diagnostics["refinement_triggered"])
        refine_accepted += int(diagnostics["refinement_accepted"])
        refine_fallback += int(
            diagnostics["refinement_triggered"] and not diagnostics["refinement_accepted"]
        )
        action_clip_count += int(np.count_nonzero(np.abs(np.clip(raw_chunk, low, high) - raw_chunk) > 1e-5))

        progressed = False
        done = False
        for action in raw_chunk:
            if step >= args.horizon:
                break
            obs, _, done, _ = env.step(action)
            target_contacts, obstacle_names = common.contact_snapshot(raw_env, maps)
            target_contact_any = target_contact_any or target_contacts > 0
            collisions.update(obstacle_names)
            success = success or common.task_success(env)
            step += 1
            progressed = True
            if done or (args.terminate_on_success and success):
                break
        chunks += 1
        if not progressed or done or (args.terminate_on_success and success):
            break

    target_end = common.object_position(
        raw_env, common.target_record(raw_env, records, args.target_object_name)
    )
    collision_any = bool(collisions)
    return {
        "condition": condition,
        "success": success,
        "safe_success": success and not collision_any,
        "success_with_collision": success and collision_any,
        "collision_failure": (not success) and collision_any,
        "ncr": (not success) and (not collision_any),
        "collision_any": collision_any,
        "collision_names": sorted(collisions),
        "target_contact_any": target_contact_any,
        "target_displacement_m": float(np.linalg.norm(target_end - target_start)),
        "horizon": step,
        "chunks": chunks,
        "action_clip_count": action_clip_count,
        "inference_time_ms_mean": float(np.mean(inference_ms)) if inference_ms else None,
        "selected_nonzero_rate": selected_changed / chunks if chunks else 0.0,
        "positive_safe_candidate_coverage": positive_safe_coverage / chunks if chunks else 0.0,
        "refine_trigger_count": refine_triggered,
        "refine_accept_count": refine_accepted,
        "refine_fallback_count": refine_fallback,
        "refine_accept_rate": refine_accepted / refine_triggered if refine_triggered else None,
        "initial_p0_regime": initial_regime,
        "pregrasp_regime_counts": {
            name: pregrasp_regimes.count(name)
            for name in (
                "target_directed_obstructed",
                "non_target_directed",
                "target_directed_clear",
            )
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=common.DEFAULT_AGENT)
    parser.add_argument("--envs", nargs="+", default=["PickPlaceBreadCerealCan"])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--refine-timestep", type=int, default=40)
    parser.add_argument("--eef-radius", type=float, default=0.0)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--terminate-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sample_refine/paired_pilot"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError("{} exists; pass --overwrite".format(args.out_dir))
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt = common.load_policy(args.agent, device)
    output_path = args.out_dir / "episode_metrics.jsonl"
    with output_path.open("w") as output:
        for env_name in args.envs:
            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=ckpt, env_name=env_name, render=False, render_offscreen=False, verbose=False
            )
            try:
                warmup_policy(policy, env, args)
                for seed in args.seeds:
                    for episode in range(args.n_rollouts):
                        layout_seed = seed * 1000 + episode
                        random.seed(layout_seed)
                        np.random.seed(layout_seed)
                        env.reset()
                        initial_state = env.get_state()
                        for condition in CONDITIONS:
                            metrics = rollout_condition(
                                policy, env, initial_state, condition, seed * 1_000_000 + episode, args
                            )
                            row = {
                                "env": env_name,
                                "seed": seed,
                                "episode": episode,
                                "layout_seed": layout_seed,
                                "initial_state_hash": common.state_hash(initial_state),
                                **metrics,
                            }
                            output.write(json.dumps(common.json_safe(row)) + "\n")
                            output.flush()
                            print(env_name, seed, episode, condition, metrics["success"], metrics["collision_any"], flush=True)
            finally:
                common.raw_env_from_wrapper(env).close()
    manifest = vars(args).copy()
    manifest["agent"] = str(manifest["agent"])
    manifest["out_dir"] = str(manifest["out_dir"])
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote {}".format(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
