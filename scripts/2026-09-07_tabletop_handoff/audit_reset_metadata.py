"""Check task metadata against restored static body poses without policy inference."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import robomimic.utils.file_utils as F
import robomimic.utils.obs_utils as O

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    m = json.loads((ROOT / "outputs/tabletop_wide_support/raised_milk_paired_seed828_n50/baseline/manifest.json").read_text())
    ckpt = F.load_dict_from_checkpoint(m["checkpoint"])
    config, _ = F.config_from_checkpoint(ckpt_dict=ckpt)
    O.initialize_obs_utils_with_config(config)
    env, _ = F.env_from_checkpoint(ckpt_dict=ckpt, env_name=m["environment"], render_offscreen=True)
    raw = env.base_env
    bank = pickle.load(open(m["state_bank"], "rb"))
    rows = []
    try:
        for episode in (8, 43):
            for reset_seed in (101, 202, 303):
                raw.rng = np.random.default_rng(reset_seed)
                state = bank["states"][f"PickPlaceCanTabletopRaisedMilk:828:{episode}"]["state"]
                env.reset_to(state)
                goal = np.asarray(raw.goal_xy).copy()
                physical = raw.sim.model.body_pos[raw.goal_tray_body_id, :2].copy()
                obstacle = raw.sim.model.body_pos[raw.obstacle_body_id, :2].copy()
                row = {"episode": episode, "reset_seed": reset_seed,
                       "restored_dynamic_state_max_abs": float(np.max(np.abs(raw.sim.get_state().flatten() - state["states"]))),
                       "metadata_goal_xy": goal.tolist(), "physical_goal_xy": physical.tolist(),
                       "goal_error_m": float(np.linalg.norm(goal - physical)),
                       "metadata_obstacle_xy": np.asarray(raw.obstacle_xy).tolist(),
                       "physical_obstacle_xy": obstacle.tolist(),
                       "obstacle_error_m": float(np.linalg.norm(raw.obstacle_xy - obstacle))}
                # Synthetic settled target positions test the goal predicate only.
                # These are not rollout successes or teacher evidence.
                qpos = raw.sim.data.get_joint_qpos(raw.can.joints[0]).copy()
                tests = []
                half = raw.goal_tray_inner_size / 2.
                offsets = [(0., 0.)] + [(sign * factor * half[0], 0.)
                           for factor in (.8, 1.2) for sign in (-1, 1)] + [
                           (0., sign * factor * half[1]) for factor in (.8, 1.2) for sign in (-1, 1)]
                for dx, dy in offsets:
                    qpos[:3] = [physical[0] + dx, physical[1] + dy, raw.table_offset[2] + .05]
                    raw.sim.data.set_joint_qpos(raw.can.joints[0], qpos)
                    raw.sim.forward()
                    raw.goal_xy = goal
                    before = bool(raw._can_in_goal())
                    raw.goal_xy = physical
                    after = bool(raw._can_in_goal())
                    tests.append({"offset_xy": [dx, dy], "before_sync": before, "after_sync": after})
                row["synthetic_goal_predicate_tests"] = tests
                rows.append(row)
    finally:
        raw.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
