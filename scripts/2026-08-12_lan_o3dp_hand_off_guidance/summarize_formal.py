"""Verify and summarize the frozen four-environment hand-off 2x2 evaluation."""

import argparse
import json
import math
from pathlib import Path


ENVIRONMENTS = {
    "d0": "PickPlaceCan",
    "d1": "PickPlaceBreadCan",
    "d2": "PickPlaceBreadCerealCan",
    "d3": "PickPlaceBreadCerealMilkCan",
}
PHYSICS = ("full_physics", "hand_off")
CONDITIONS = ("baseline", "guided")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def exact_mcnemar(discordant_a, discordant_b):
    n = int(discordant_a + discordant_b)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(discordant_a, discordant_b) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def outcome_summary(rows):
    n = len(rows)
    collision_rows = [row for row in rows if row["collision_any"]]
    guidance_before_collision = sum(
        row.get("first_guidance_active_step") is not None
        and row.get("first_collision_step") is not None
        and row["first_guidance_active_step"] < row["first_collision_step"]
        for row in collision_rows
    )
    return {
        "episodes": n,
        "task_sr": sum(row["success"] for row in rows) / n,
        "safe_sr": sum(row["safe_success"] for row in rows) / n,
        "cr": sum(row["collision_any"] for row in rows) / n,
        "ncr": sum(row["ncr"] for row in rows) / n,
        "hand_cr": sum(row.get("hand_obstacle_collision_any", False) for row in rows) / n,
        "finger_cr": sum(row.get("finger_obstacle_collision_any", False) for row in rows) / n,
        "arm_cr": sum(row.get("arm_obstacle_collision_any", False) for row in rows) / n,
        "target_cr": sum(row.get("target_obstacle_collision_any", False) for row in rows) / n,
        "safe_success": sum(row["safe_success"] for row in rows),
        "success_with_collision": sum(row["success_with_collision"] for row in rows),
        "collision_failure": sum(row["collision_failure"] for row in rows),
        "collision_free_non_completion": sum(row["ncr"] for row in rows),
        "horizon_truncated": sum(row["horizon_truncated"] for row in rows),
        "guidance_trigger_rate_mean": sum(row["guidance_trigger_rate"] for row in rows) / n,
        "guidance_update_norm_mean": sum(row["guidance_update_norm_mean"] for row in rows) / n,
        "guidance_update_norm_max": max(row["guidance_update_norm_max"] for row in rows),
        "guidance_active_episodes": sum(
            row.get("first_guidance_active_step") is not None for row in rows
        ),
        "guidance_before_collision": guidance_before_collision,
        "obstacle_cloud_missing_chunks": sum(
            row.get("obstacle_cloud_missing_chunks", 0) for row in rows
        ),
        "obstacle_cloud_missing_episodes": sum(
            row.get("obstacle_cloud_missing_chunks", 0) > 0 for row in rows
        ),
    }


def transition_summary(before, after):
    keys = sorted(set(before) & set(after))
    if set(before) != set(after):
        raise ValueError("Paired result keys differ")

    def transitions(field):
        zero_to_one_keys = [
            key for key in keys if not before[key][field] and after[key][field]
        ]
        one_to_zero_keys = [
            key for key in keys if before[key][field] and not after[key][field]
        ]
        zero_to_one = len(zero_to_one_keys)
        one_to_zero = len(one_to_zero_keys)
        return {
            "zero_to_one": zero_to_one,
            "one_to_zero": one_to_zero,
            "zero_to_one_keys": zero_to_one_keys,
            "one_to_zero_keys": one_to_zero_keys,
            "exact_mcnemar_p": exact_mcnemar(zero_to_one, one_to_zero),
        }

    return {
        "pairs": len(keys),
        "task": transitions("success"),
        "safe_success": transitions("safe_success"),
        "collision": transitions("collision_any"),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists() and not args.overwrite:
        raise FileExistsError("Output already exists: {}".format(args.out_dir))
    args.out_dir.mkdir(parents=True, exist_ok=args.overwrite)
    indexed = {}
    manifests = {}
    for label, env_name in ENVIRONMENTS.items():
        for physics in PHYSICS:
            result_dir = args.root / "formal_{}_{}_seed812_n50".format(label, physics)
            rows = read_rows(result_dir / "episode_metrics.jsonl")
            manifest = json.loads((result_dir / "manifest.json").read_text())
            if manifest["envs"] != [env_name] or manifest["n_rollouts"] != 50:
                raise ValueError("Unexpected manifest contract in {}".format(result_dir))
            expected_hand = physics == "full_physics"
            if manifest["hand_collision_enabled"] != expected_hand:
                raise ValueError("Hand physics mismatch in {}".format(result_dir))
            if (
                manifest["xy_only"] is not True
                or manifest["obstacle_cost_type"] != "huber_hinge"
                or manifest["obstacle_cost_smoothing"] != 0.01
                or manifest["guidance_scales"] != [0.01]
                or manifest["safety_distances"] != [0.04]
                or manifest["horizon"] != 600
            ):
                raise ValueError("Frozen guidance contract mismatch in {}".format(result_dir))
            manifests[(label, physics)] = manifest
            for condition in CONDITIONS:
                selected = [row for row in rows if row["condition"] == condition]
                if len(selected) != 50:
                    raise ValueError("Expected 50 {} rows in {}".format(condition, result_dir))
                for row in selected:
                    audit = row.get("hand_collision_audit", {})
                    if (
                        row.get("hand_collision_enabled") != expected_hand
                        or audit.get("enabled") != expected_hand
                        or not (0 < row.get("horizon", 0) <= 600)
                        or row.get("obstacle_cost_type") != "huber_hinge"
                    ):
                        raise ValueError(
                            "Per-rollout physics or guidance contract mismatch in {}".format(
                                result_dir
                            )
                        )
                    hand_masks = audit.get("hand_masks", {})
                    if not hand_masks:
                        raise ValueError("Missing hand-mask audit in {}".format(result_dir))
                    if expected_hand:
                        if any(mask == [0, 0] for mask in hand_masks.values()):
                            raise ValueError("Full-physics hand mask is disabled")
                    else:
                        if any(mask != [0, 0] for mask in hand_masks.values()):
                            raise ValueError("Hand-off mask is not disabled")
                        if row.get("hand_obstacle_collision_any", False):
                            raise ValueError("Hand collision leaked into hand-off rollout")
                    if condition == "guided":
                        if (
                            row.get("guidance_scale") != 0.01
                            or row.get("safety_distance") != 0.04
                            or row.get("obstacle_cost_smoothing") != 0.01
                        ):
                            raise ValueError("Guided row does not match frozen parameters")
                    elif (
                        row.get("guidance_scale") is not None
                        or row.get("safety_distance") is not None
                        or row.get("obstacle_cost_smoothing") is not None
                    ):
                        raise ValueError("Baseline row unexpectedly contains guidance")
                indexed[(label, physics, condition)] = {
                    (row["seed"], row["episode"]): row for row in selected
                }

        reference = indexed[(label, "full_physics", "baseline")]
        for physics in PHYSICS:
            for condition in CONDITIONS:
                candidate = indexed[(label, physics, condition)]
                if set(candidate) != set(reference):
                    raise ValueError("Pair keys differ for {} {} {}".format(label, physics, condition))
                for key in reference:
                    if (
                        candidate[key]["initial_state_hash"]
                        != reference[key]["initial_state_hash"]
                        or candidate[key]["policy_seed"] != reference[key]["policy_seed"]
                    ):
                        raise ValueError("State or policy seed mismatch for {} {}".format(label, key))
                    if (
                        candidate[key]["hand_collision_audit"].get("finger_masks")
                        != reference[key]["hand_collision_audit"].get("finger_masks")
                    ):
                        raise ValueError("Finger contact masks changed for {} {}".format(label, key))

    summary = {}
    transitions = {}
    for label in ENVIRONMENTS:
        summary[label] = {}
        transitions[label] = {}
        for physics in PHYSICS:
            summary[label][physics] = {}
            for condition in CONDITIONS:
                rows = list(indexed[(label, physics, condition)].values())
                summary[label][physics][condition] = outcome_summary(rows)
            transitions[label]["guidance_with_{}".format(physics)] = transition_summary(
                indexed[(label, physics, "baseline")],
                indexed[(label, physics, "guided")],
            )
        for condition in CONDITIONS:
            transitions[label]["hand_off_with_{}".format(condition)] = transition_summary(
                indexed[(label, "full_physics", condition)],
                indexed[(label, "hand_off", condition)],
            )

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "paired_transitions.json").write_text(
        json.dumps(transitions, indent=2) + "\n"
    )
    lines = [
        "# LAN-O3DP hand-off guidance: formal 2x2 result",
        "",
        "## Result",
        "",
        "Disabling only the Panda hand collision shell is an effective physics "
        "relaxation in the hard environments, but it does **not** unlock a "
        "material guidance benefit. In D2, hand-off changes baseline Task SR "
        "from 25/50 to 31/50 while Safe SR stays 17/50; the extra completions "
        "therefore remain collision-contaminated. In D3 it changes baseline "
        "Task/Safe SR from 36/50 and 17/50 to 39/50 and 24/50.",
        "",
        "With hand collision disabled, guidance changes D2 Task/Safe SR from "
        "31/50 and 17/50 to 31/50 and 18/50, and D3 from 39/50 and 24/50 "
        "to 40/50 and 25/50. Every paired guidance comparison has exact "
        "McNemar `p >= 0.5`; the observed net changes are one episode at most.",
        "",
        "The contact split explains the limit. Under full physics, guidance "
        "does not change hand CR at all (D2 12/50 to 12/50; D3 19/50 to "
        "19/50). Under hand-off, residual finger CR changes from 16/50 to "
        "18/50 in D2 and stays 17/50 in D3. Target-object CR falls, but the "
        "XY point-EEF cost still does not model the swept finger geometry or "
        "produce a persistent bypass route.",
        "",
        "Decision: retain hand-off as a useful mechanism diagnostic, not as "
        "evidence that the current 3-D guidance path succeeds. Do not scale a "
        "larger blind `rho` / `Q` sweep of this same XY point cost. Also treat "
        "hand-off safety numbers as modified-physics results: the hand shell "
        "cannot make any physical contact, including with the target or table.",
        "",
        "## Protocol and integrity",
        "",
        "Frozen policy: epoch-50 corrected high-resolution LAN-O3DP sample "
        "checkpoint (`...paper256_hires_continuous_success_0.7.pth`). Its "
        "observation is `agent_pos` plus a 256-point world-frame point cloud "
        "containing only Can and VisualCan; Bread, Cereal, and Milk are "
        "excluded from the policy input and supplied only to guidance through "
        "the oracle obstacle cloud.",
        "",
        "Frozen protocol: seed 812, 50 paired states per environment, H600, XY "
        "Huberized hinge, `Q=0.04 m`, `epsilon=0.01 m`, `rho=0.01`.",
        "The seed-705 D2 calibration bank is excluded. All 800 condition "
        "rollouts passed state-hash, policy-seed, per-row parameter, and "
        "collision-mask checks. Only `gripper0_right_hand_collision` is "
        "disabled in hand-off; all finger mesh and pad masks are byte-for-byte "
        "unchanged from full physics.",
        "",
        "## Aggregate outcomes",
        "",
        "| Env | Physics | Policy | Task SR | Safe SR | CR | NCR | Hand CR | Finger CR | Arm CR | Target CR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ENVIRONMENTS:
        for physics in PHYSICS:
            for condition in CONDITIONS:
                item = summary[label][physics][condition]
                lines.append(
                    "| {} | {} | {} | {}/50 | {}/50 | {}/50 | {}/50 | {}/50 | {}/50 | {}/50 | {}/50 |".format(
                        label.upper(),
                        physics,
                        condition,
                        round(50 * item["task_sr"]),
                        round(50 * item["safe_sr"]),
                        round(50 * item["cr"]),
                        round(50 * item["ncr"]),
                        round(50 * item["hand_cr"]),
                        round(50 * item["finger_cr"]),
                        round(50 * item["arm_cr"]),
                        round(50 * item["target_cr"]),
                    )
                )
    lines.extend(
        [
            "",
            "## Hard-environment four-way partition",
            "",
            "| Env | Physics | Policy | Safe success | Success + collision | Collision failure | NCR |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for label in ("d2", "d3"):
        for physics in PHYSICS:
            for condition in CONDITIONS:
                item = summary[label][physics][condition]
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} |".format(
                        label.upper(),
                        physics,
                        condition,
                        item["safe_success"],
                        item["success_with_collision"],
                        item["collision_failure"],
                        item["collision_free_non_completion"],
                    )
                )
    lines.extend(
        [
            "",
            "## Guidance mechanism activity",
            "",
            "| Env | Physics | Active episodes | Mean trigger rate | Max update norm | Active before collision | Missing-cloud chunks / episodes |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("d1", "d2", "d3"):
        for physics in PHYSICS:
            item = summary[label][physics]["guided"]
            lines.append(
                "| {} | {} | {}/50 | {:.1%} | {:.3e} | {} | {} / {} |".format(
                    label.upper(),
                    physics,
                    item["guidance_active_episodes"],
                    item["guidance_trigger_rate_mean"],
                    item["guidance_update_norm_max"],
                    item["guidance_before_collision"],
                    item["obstacle_cloud_missing_chunks"],
                    item["obstacle_cloud_missing_episodes"],
                )
            )
    lines.extend(["", "## Paired transitions", ""])
    for label in ENVIRONMENTS:
        lines.append("### {}".format(label.upper()))
        lines.append("")
        for comparison, item in transitions[label].items():
            task = item["task"]
            collision = item["collision"]
            safe = item["safe_success"]
            lines.append(
                "- `{}`: task 0->1 / 1->0 = `{}/{}`; collision 1->0 / 0->1 = "
                "`{}/{}`; safe 0->1 / 1->0 = `{}/{}`; exact p(task/safe/CR) "
                "= `{:.4g}/{:.4g}/{:.4g}`.".format(
                    comparison,
                    task["zero_to_one"],
                    task["one_to_zero"],
                    collision["one_to_zero"],
                    collision["zero_to_one"],
                    safe["zero_to_one"],
                    safe["one_to_zero"],
                    task["exact_mcnemar_p"],
                    safe["exact_mcnemar_p"],
                    collision["exact_mcnemar_p"],
                )
            )
        lines.append("")
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("Wrote {}".format(args.out_dir))


if __name__ == "__main__":
    main()
