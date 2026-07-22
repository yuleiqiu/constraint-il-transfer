"""Short-prefix sweep and fixed-gate ranking for sample-refine parameters.

Consumes the HDF5 fixed states captured by the earlier same-state diagnostic,
but loads and evaluates the sample-refine policy independently.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import common  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


def text_value(item, name):
    value = item[name][()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_state(item):
    ep_meta = text_value(item, "ep_meta")
    return {
        "id": item.name.rsplit("/", 1)[-1],
        "env": str(item.attrs["env"]),
        "noise_seed": int(item.attrs["noise_seed"]),
        "state": {
            "states": item["sim_state"][()],
            "model": text_value(item, "model_xml"),
            **({"ep_meta": ep_meta} if ep_meta else {}),
        },
        "obs": {key: dataset[()] for key, dataset in item["obs"].items()},
    }


def execute_prefix(env, raw_chunk, maps):
    raw_env = common.raw_env_from_wrapper(env)
    collided = False
    positions = []
    for action in raw_chunk:
        obs, _, done, _ = env.step(action)
        positions.append(common.current_eef_position(obs))
        _, names = common.contact_snapshot(raw_env, maps)
        collided = collided or bool(names)
        if done:
            break
    return collided, np.asarray(positions, np.float32)


def sample_from_saved(policy, env, saved, args, margin, scale, sample_enabled, refine_enabled):
    obs = saved["obs"]
    context = common.make_context(
        policy,
        env,
        obs,
        candidate_count=4,
        margin=margin,
        refine_timestep=40,
        guidance_scale=scale,
        candidate_seed=saved["noise_seed"],
        refinement_seed=saved["noise_seed"] + 1,
        sample_enabled=sample_enabled,
        refinement_enabled=refine_enabled,
        target_object_name=args.target_object_name,
        eef_radius=args.eef_radius,
    )
    _, raw, diagnostics = common.sample_action_chunk(policy, obs, context)
    return raw, diagnostics


def summarize(rows, margins, scales):
    summary = []
    for margin in margins:
        for scale in scales:
            selected = [r for r in rows if r["margin"] == margin and r["scale"] == scale]
            positive_baseline = [r for r in selected if r["baseline_progress_m"] > 0]
            intent_rate = (
                float(np.mean([r["final_progress_m"] > 0 for r in positive_baseline]))
                if positive_baseline else 1.0
            )
            clip_count = int(sum(r["action_clipping"] for r in selected))
            max_displacement = max((r["waypoint_displacement_m"] for r in selected), default=0.0)
            conversions = int(sum(r["baseline_collision"] and not r["condition_collision"] for r in selected))
            regressions = int(sum((not r["baseline_collision"]) and r["condition_collision"] for r in selected))
            clearance_improvements = [
                r["final_clearance_m"] - r["baseline_clearance_m"]
                for r in selected
                if r["final_clearance_m"] is not None and r["baseline_clearance_m"] is not None
            ]
            eligible = clip_count == 0 and max_displacement <= 0.02 and intent_rate >= 0.90
            summary.append(
                {
                    "margin": margin,
                    "scale": scale,
                    "states": len(selected),
                    "action_clip_states": clip_count,
                    "max_waypoint_displacement_m": max_displacement,
                    "task_intent_preservation_rate": intent_rate,
                    "actual_safety_conversions": conversions,
                    "actual_safety_regressions": regressions,
                    "median_clearance_improvement_m": float(np.median(clearance_improvements)) if clearance_improvements else None,
                    "eligible": eligible,
                }
            )
    eligible = [row for row in summary if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -row["actual_safety_conversions"],
            -round(row["median_clearance_improvement_m"], 6)
            if row["median_clearance_improvement_m"] is not None
            else float("inf"),
            row["margin"],
            row["scale"],
        )
    )
    return summary, eligible[0] if eligible else None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=common.DEFAULT_AGENT)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--envs", nargs="+", default=["PickPlaceBreadCerealCan"])
    parser.add_argument("--margins", nargs="+", type=float, default=[0.02, 0.03, 0.04])
    parser.add_argument("--scales", nargs="+", type=float, default=[0.001, 0.003, 0.01])
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument("--eef-radius", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sample_refine/fixed_state_sweep"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError("{} exists; pass --overwrite".format(args.out_dir))
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    policy, ckpt = common.load_policy(args.agent, TorchUtils.get_torch_device(try_to_use_cuda=True))
    envs = {}
    rows = []
    try:
        with h5py.File(args.states, "r") as states, (args.out_dir / "state_metrics.jsonl").open("w") as output:
            for state_id in sorted(states["states"]):
                saved = read_state(states["states"][state_id])
                if saved["env"] not in args.envs:
                    continue
                if saved["env"] not in envs:
                    envs[saved["env"]], _ = FileUtils.env_from_checkpoint(
                        ckpt_dict=ckpt, env_name=saved["env"], render=False, render_offscreen=False, verbose=False
                    )
                env = envs[saved["env"]]
                policy.start_episode()
                env.reset_to(saved["state"])
                records = common.active_object_records(common.raw_env_from_wrapper(env))
                maps = common.contact_maps(common.raw_env_from_wrapper(env), records, args.target_object_name)
                baseline_raw, _ = sample_from_saved(
                    policy, env, saved, args, margin=args.margins[0], scale=0.0, sample_enabled=False, refine_enabled=False
                )
                env.reset_to(saved["state"])
                baseline_collision, _ = execute_prefix(env, baseline_raw, maps)
                for margin in args.margins:
                    policy.start_episode()
                    env.reset_to(saved["state"])
                    _, baseline_diag = sample_from_saved(
                        policy, env, saved, args, margin=margin, scale=0.0, sample_enabled=False, refine_enabled=False
                    )
                    baseline_progress = baseline_diag["final_progress_m"]
                    baseline_clearance = baseline_diag["final_clearance_m"]
                    for scale in args.scales:
                        policy.start_episode()
                        env.reset_to(saved["state"])
                        raw, diagnostics = sample_from_saved(
                            policy, env, saved, args, margin, scale, True, True
                        )
                        env.reset_to(saved["state"])
                        condition_collision, _ = execute_prefix(env, raw, maps)
                        start = common.current_eef_position(saved["obs"])
                        baseline_waypoints = start[None] + np.cumsum(baseline_raw[:, :3], axis=0)
                        condition_waypoints = start[None] + np.cumsum(raw[:, :3], axis=0)
                        total_waypoint_displacement = float(
                            np.max(np.linalg.norm(condition_waypoints[:, :2] - baseline_waypoints[:, :2], axis=1))
                        )
                        low, high = common.action_limits(env, raw.shape[-1])
                        requires_clip = bool(
                            np.any(np.abs(np.clip(raw, low, high) - raw) > 1e-5)
                        )
                        row = {
                            "state_id": saved["id"],
                            "env": saved["env"],
                            "margin": margin,
                            "scale": scale,
                            "baseline_collision": baseline_collision,
                            "condition_collision": condition_collision,
                            "baseline_progress_m": baseline_progress,
                            "final_progress_m": diagnostics["final_progress_m"],
                            "baseline_clearance_m": baseline_clearance,
                            "final_clearance_m": diagnostics["final_clearance_m"],
                            "action_clipping": requires_clip,
                            "refinement_proposal_clipping": diagnostics["action_clipping"],
                            "waypoint_displacement_m": max(
                                diagnostics["masked_xy_waypoint_update_max_m"],
                                total_waypoint_displacement,
                            ),
                            "selected_index": diagnostics["selected_index"],
                            "refinement_accepted": diagnostics["refinement_accepted"],
                            "fallback_reason": diagnostics["fallback_reason"],
                        }
                        rows.append(row)
                        output.write(json.dumps(common.json_safe(row)) + "\n")
                        output.flush()
    finally:
        for env in envs.values():
            common.raw_env_from_wrapper(env).close()

    summary, selected = summarize(rows, args.margins, args.scales)
    result = {
        "fixed": {"candidate_count": 4, "refine_timestep": 40},
        "gates": {"action_clips": 0, "max_waypoint_displacement_m": 0.02, "min_task_intent_preservation_rate": 0.90},
        "ranking": [
            "actual_safety_conversions",
            "median_clearance_improvement_m_rounded_to_1um",
            "smaller_margin",
            "smaller_scale",
        ],
        "configs": summary,
        "selected": selected,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(common.json_safe(result), indent=2) + "\n")
    print(json.dumps(common.json_safe(result["selected"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
