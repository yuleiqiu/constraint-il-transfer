"""Collect obstacle-free broad-route Tabletop point-cloud demonstrations."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import wide_support_teacher as TEACHER  # noqa: E402

DIRECT_COLLECT_PATH = SCRIPT_DIR.parent / "2026-08-14_tabletop_guidance" / "collect_demos.py"
_SPEC = importlib.util.spec_from_file_location("tabletop_direct_collect", DIRECT_COLLECT_PATH)
DIRECT_COLLECT = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = DIRECT_COLLECT
_SPEC.loader.exec_module(DIRECT_COLLECT)


def route_support(data, names):
    peaks = np.asarray([float(data[name].attrs["achieved_route_signed_peak_m"]) for name in names])
    midpoints = np.asarray(
        [float(data[name].attrs["commanded_route_midpoint_offset_m"]) for name in names]
    )
    proxy_radius = 0.06
    obstacle_offsets = (-0.0175, 0.0175)
    result = {
        "demos": len(names),
        "upper_wide": int(np.count_nonzero(peaks > 0.065)),
        "lower_wide": int(np.count_nonzero(peaks < -0.065)),
        "near_direct": int(np.count_nonzero(np.abs(peaks) < 0.04)),
        "signed_peak_min_m": float(peaks.min()),
        "signed_peak_max_m": float(peaks.max()),
        "midpoint_mean_m": float(midpoints.mean()),
        "counterfactual_proxy": {},
    }
    for offset in obstacle_offsets:
        safe = np.abs(midpoints - offset) > proxy_radius
        result["counterfactual_proxy"][f"obstacle_offset_{offset:+.4f}"] = {
            "safe": int(safe.sum()),
            "collision": int((~safe).sum()),
        }
    result["gates"] = {
        "upper_wide_15": result["upper_wide"] >= 15,
        "lower_wide_15": result["lower_wide"] >= 15,
        "near_direct_15": result["near_direct"] >= 15,
        "symmetric_mean_2cm": abs(result["midpoint_mean_m"]) <= 0.02,
        "proxy_safe_and_collision_20": all(
            item["safe"] >= 20 and item["collision"] >= 20
            for item in result["counterfactual_proxy"].values()
        ),
    }
    result["passed"] = all(result["gates"].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.episodes < 2:
        raise ValueError("--episodes must be at least 2")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    env = DIRECT_COLLECT.make_env(args.seed)
    config = TEACHER.WideSupportTeacherConfig()
    teacher = TEACHER.WideSupportPickPlaceTeacher(config)
    env_meta = env.serialize()
    demo_names = []
    total = 0
    try:
        with h5py.File(partial, "w") as output_file:
            data = output_file.create_group("data")
            data.attrs["env_args"] = json.dumps(env_meta, indent=4)
            data.attrs["action_key"] = DIRECT_COLLECT.ACTION_KEY
            data.attrs["action_contract"] = (
                "next achieved absolute world EEF xyz + site axis-angle + gripper command"
            )
            data.attrs["teacher"] = (
                "obstacle-free endpoint-conditioned RBF-GP lateral transport routes"
            )
            data.attrs["route_config"] = json.dumps(
                {
                    "points": config.route_points,
                    "length_scale": config.route_length_scale,
                    "kernel_amplitude_m2": config.route_kernel_amplitude_m2,
                    "max_lateral_m": config.route_max_lateral_m,
                    "obstacle_used_for_generation": False,
                },
                sort_keys=True,
            )
            for episode in range(args.episodes):
                episode_seed = args.seed + episode
                buffer = DIRECT_COLLECT.EpisodeBuffer()
                result = teacher.rollout(env, seed=episode_seed, transition_callback=buffer.append)
                if not result.success:
                    raise RuntimeError(
                        f"teacher failed at episode {episode} seed {episode_seed}; refusing a partial dataset"
                    )
                demo_name = f"demo_{episode}"
                model_xml = env.env.sim.model.get_xml()
                DIRECT_COLLECT.write_episode(
                    data, demo_name, buffer, result, model_xml=model_xml, seed=episode_seed
                )
                group = data[demo_name]
                group.attrs["commanded_route_signed_peak_m"] = result.commanded_route_signed_peak_m
                group.attrs["commanded_route_midpoint_offset_m"] = result.commanded_route_midpoint_offset_m
                group.attrs["achieved_route_signed_peak_m"] = result.achieved_route_signed_peak_m
                group.attrs["route_draw_attempts"] = result.route_draw_attempts
                demo_names.append(demo_name)
                total += len(buffer)
                print(
                    json.dumps(
                        {
                            "demo": demo_name,
                            "seed": episode_seed,
                            "steps": len(buffer),
                            "success": result.success,
                            "signed_peak_m": result.achieved_route_signed_peak_m,
                            "midpoint_m": result.commanded_route_midpoint_offset_m,
                        }
                    ),
                    flush=True,
                )
            data.attrs["total"] = total
            train, valid = DIRECT_COLLECT.write_masks(
                output_file, demo_names, args.validation_fraction, args.seed + 991
            )
            data.attrs["train_demos"] = len(train)
            data.attrs["valid_demos"] = len(valid)
            support = route_support(data, train)
            data.attrs["train_route_support"] = json.dumps(support, sort_keys=True)
            if not support["passed"]:
                raise RuntimeError(f"route-support gate failed: {support['gates']}")
    finally:
        env.env.close()
    if output.exists():
        output.unlink()
    partial.replace(output)
    print(
        json.dumps(
            {
                "wrote": str(output),
                "demos": len(demo_names),
                "transitions": total,
                "train": len(train),
                "valid": len(valid),
                "train_route_support": support,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

