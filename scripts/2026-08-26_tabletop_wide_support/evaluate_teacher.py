"""Evaluate task success and route coverage of the wide-support teacher."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import robosuite as suite


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import wide_support_teacher as TEACHER  # noqa: E402


def make_env(seed):
    return suite.make(
        env_name="PickPlaceCanTabletop",
        robots="Panda",
        gripper_types="RethinkGripper",
        controller_configs=TEACHER.absolute_osc_controller_config(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=600,
        ignore_done=False,
        seed=seed,
    )


def summarize(records):
    peaks = np.asarray([item["achieved_route_signed_peak_m"] for item in records])
    successes = np.asarray([item["success"] for item in records], dtype=bool)
    gates = {
        "all_success": bool(np.all(successes)),
        "upper_wide_3": int(np.count_nonzero(peaks > 0.065)) >= 3,
        "lower_wide_3": int(np.count_nonzero(peaks < -0.065)) >= 3,
        "near_direct_3": int(np.count_nonzero(np.abs(peaks) < 0.04)) >= 3,
        "horizon": int(max(item["steps"] for item in records)) < 600,
    }
    return {
        "episodes": len(records),
        "successes": int(successes.sum()),
        "mean_steps": float(np.mean([item["steps"] for item in records])),
        "max_steps": int(max(item["steps"] for item in records)),
        "achieved_signed_peak_min_m": float(peaks.min()),
        "achieved_signed_peak_max_m": float(peaks.max()),
        "upper_wide": int(np.count_nonzero(peaks > 0.065)),
        "lower_wide": int(np.count_nonzero(peaks < -0.065)),
        "near_direct": int(np.count_nonzero(np.abs(peaks) < 0.04)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=16000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    env = make_env(args.seed)
    teacher = TEACHER.WideSupportPickPlaceTeacher()
    records = []
    try:
        for episode in range(args.episodes):
            result = teacher.rollout(env, seed=args.seed + episode)
            record = {
                "episode": episode,
                "seed": args.seed + episode,
                "success": result.success,
                "grasped": result.grasped,
                "steps": result.steps,
                "can_start": result.can_start.tolist(),
                "goal_xy": result.goal_xy.tolist(),
                "max_transport_line_deviation_m": result.max_transport_line_deviation_m,
                "commanded_route_signed_peak_m": result.commanded_route_signed_peak_m,
                "commanded_route_midpoint_offset_m": result.commanded_route_midpoint_offset_m,
                "achieved_route_signed_peak_m": result.achieved_route_signed_peak_m,
                "route_draw_attempts": result.route_draw_attempts,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        env.close()
    payload = {"summary": summarize(records), "episodes": records}
    print(json.dumps(payload["summary"], indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    if not payload["summary"]["passed"]:
        raise RuntimeError("wide-support teacher gate failed")


if __name__ == "__main__":
    main()

