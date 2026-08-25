"""Evaluate the obstacle-free direct PickPlaceCanTabletop teacher."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import robosuite as suite


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import waypoint_teacher as TEACHER  # noqa: E402


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=14000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    env = make_env(args.seed)
    teacher = TEACHER.DirectPickPlaceTeacher()
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
            }
            records.append(record)
            print(json.dumps(record))
    finally:
        env.close()

    summary = {
        "episodes": len(records),
        "successes": sum(item["success"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])),
        "grasp_rate": float(np.mean([item["grasped"] for item in records])),
        "mean_steps": float(np.mean([item["steps"] for item in records])),
        "max_steps": int(np.max([item["steps"] for item in records])),
        "maximum_transport_line_deviation_m": float(
            np.max([item["max_transport_line_deviation_m"] for item in records])
        ),
    }
    payload = {"summary": summary, "episodes": records}
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
