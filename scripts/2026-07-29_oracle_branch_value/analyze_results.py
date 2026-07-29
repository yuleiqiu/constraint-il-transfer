"""Aggregate branch outcomes and write the canonical pilot report."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from branch_value import select_oracle_branch
from simulator_branch_eval import local_selector_branch


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def outcome_counts(branches):
    return dict(Counter(branch["outcome"] for branch in branches))


def analyze(states, teacher_payload):
    source = []
    corrective = []
    local = []
    oracle_any = []
    oracle_eligible = []
    per_state = []
    for state in states:
        branches = state["branches"]
        source_branch = next(
            branch
            for branch in branches
            if branch["candidate_type"] == "source"
        )
        corrective_branches = [
            branch
            for branch in branches
            if branch["candidate_type"] == "corrective"
        ]
        local_branch = local_selector_branch(branches)
        any_branch = select_oracle_branch(branches)
        eligible_branch = select_oracle_branch(
            branches, eligible_only=True
        )
        source.append(source_branch)
        corrective.extend(corrective_branches)
        local.append(local_branch)
        oracle_any.append(any_branch)
        oracle_eligible.append(eligible_branch)
        per_state.append(
            {
                "state_id": state["state_id"],
                "source_outcome": source_branch["outcome"],
                "geometry_eligible": int(
                    sum(
                        branch["geometry_eligible"]
                        for branch in corrective_branches
                    )
                ),
                "immediate_collision_free": int(
                    sum(
                        not branch["candidate_collision_any"]
                        for branch in corrective_branches
                    )
                ),
                "corrective_target_contacts": int(
                    sum(
                        branch["target_contact_any"]
                        for branch in corrective_branches
                    )
                ),
                "corrective_successes": int(
                    sum(branch["success"] for branch in corrective_branches)
                ),
                "local_outcome": local_branch["outcome"],
                "oracle_any_outcome": any_branch["outcome"],
                "oracle_eligible_outcome": eligible_branch["outcome"],
                "capture_reproduction_max_abs": state["state_geometry"][
                    "source_capture_reproduction_max_abs"
                ],
            }
        )

    teacher_summary = (
        teacher_payload["summary"] if teacher_payload is not None else None
    )
    return {
        "states": len(states),
        "source_branches": len(source),
        "corrective_branches": len(corrective),
        "source_outcomes": outcome_counts(source),
        "corrective_outcomes": outcome_counts(corrective),
        "local_selector_outcomes": outcome_counts(local),
        "oracle_any_outcomes": outcome_counts(oracle_any),
        "oracle_geometry_eligible_outcomes": outcome_counts(oracle_eligible),
        "corrective_task_successes": int(
            sum(branch["success"] for branch in corrective)
        ),
        "corrective_target_contacts": int(
            sum(branch["target_contact_any"] for branch in corrective)
        ),
        "corrective_safe_successes": int(
            sum(
                branch["success"] and not branch["collision_any"]
                for branch in corrective
            )
        ),
        "corrective_immediate_collision_free": int(
            sum(not branch["candidate_collision_any"] for branch in corrective)
        ),
        "corrective_full_collision_free": int(
            sum(not branch["collision_any"] for branch in corrective)
        ),
        "geometry_eligible_branches": int(
            sum(branch["geometry_eligible"] for branch in corrective)
        ),
        "action_clip_count": int(
            sum(branch["action_clip_count"] for branch in source + corrective)
        ),
        "maximum_action_clip_abs": float(
            max(branch["action_clip_max_abs"] for branch in source + corrective)
        ),
        "maximum_source_capture_reproduction_abs": float(
            max(
                row["capture_reproduction_max_abs"]
                for row in per_state
            )
        ),
        "mean_branch_wall_time_s": float(
            np.mean([branch["wall_time_s"] for branch in source + corrective])
        ),
        "teacher_control": teacher_summary,
        "per_state": per_state,
    }


def write_report(path, result):
    teacher = result["teacher_control"]
    lines = [
        "# Oracle Branch-Value Pilot",
        "",
        "Status: stopped at the preregistered three-state pilot gate.",
        "",
        "## Question",
        "",
        "Does the existing SDP policy produce at least one one-chunk corrective "
        "candidate whose full frozen-source continuation succeeds, so that a "
        "learned value/world model would have a useful branch to select?",
        "",
        "## Protocol",
        "",
        "- first three held-out one-chunk-lead collision states;",
        "- one source plus 32 SDP chunks per state;",
        "- exact saved simulator state and policy observation history restored "
        "for every branch;",
        "- execute the candidate's eight actions, then run the frozen source "
        "until the original 400-step episode horizon;",
        "- identical source continuation noise seeds across branches;",
        "- record full task success, contacts, collision, staged rewards, and "
        "8/32/64/128-step checkpoints.",
        "",
        "The pilot advances to all 20 states only if at least one corrective "
        "branch completes the task. This is an oracle existence gate, not a "
        "statistical performance estimate.",
        "",
        "## Result",
        "",
        "| quantity | result |",
        "|---|---:|",
        "| saved states | {} |".format(result["states"]),
        "| source / corrective branches | {} / {} |".format(
            result["source_branches"], result["corrective_branches"]
        ),
        "| corrective Task successes | {} / {} |".format(
            result["corrective_task_successes"],
            result["corrective_branches"],
        ),
        "| corrective Can-contact branches | {} / {} |".format(
            result["corrective_target_contacts"],
            result["corrective_branches"],
        ),
        "| corrective immediate collision-free branches | {} / {} |".format(
            result["corrective_immediate_collision_free"],
            result["corrective_branches"],
        ),
        "| corrective full collision-free branches | {} / {} |".format(
            result["corrective_full_collision_free"],
            result["corrective_branches"],
        ),
        "| geometry-eligible branches | {} / {} |".format(
            result["geometry_eligible_branches"],
            result["corrective_branches"],
        ),
        "| action clipping | {} |".format(result["action_clip_count"]),
        "",
        "Corrective outcomes: `{}`.".format(result["corrective_outcomes"]),
        "Even a hindsight oracle over all 32 candidates obtains `{}`; restricting "
        "it to the source plus geometry-eligible candidates obtains `{}`.".format(
            result["oracle_any_outcomes"],
            result["oracle_geometry_eligible_outcomes"],
        ),
        "",
        "## Per-state structure",
        "",
        "| state | eligible | immediate safe | Can contact | success | local | "
        "oracle any |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in result["per_state"]:
        lines.append(
            "| {state_id} | {geometry_eligible} | "
            "{immediate_collision_free} | {corrective_target_contacts} | "
            "{corrective_successes} | {local_outcome} | "
            "{oracle_any_outcome} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The first state demonstrates the strongest local/global mismatch: "
            "all 32 corrective chunks avoid immediate contact and 30 satisfy "
            "the geometry filter, but all full continuations are NCR. In the "
            "other two states, all 64 corrective continuations are collision "
            "failures and none contacts Can.",
            "",
        ]
    )
    if teacher is not None:
        lines.extend(
            [
                "## Explicit waypoint control",
                "",
                "The existing discrete-yaw waypoint teacher was rerun on the "
                "same three simulator states. It obtains {successes}/3 Task "
                "success, {contacts}/3 Can contact, and {collisions}/3 "
                "distractor collision, using 121--144 controlled steps before "
                "handoff. Thus these states are solvable; the SDP one-chunk "
                "candidate set is the failed component.".format(
                    successes=teacher["teacher_then_policy"]["successes"],
                    contacts=teacher["teacher_then_policy"]["target_contacts"],
                    collisions=teacher["teacher_then_policy"]["collisions"],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "Do not expand this diagnostic to 20 states and do not train a world "
            "model whose only role is to rank the current one-chunk SDP "
            "candidates. On the pilot, there is no successful future for such "
            "a model to select.",
            "",
            "This does not reject long-horizon model-based planning that "
            "generates new multi-step behavior. It rejects the narrower and "
            "cheaper hypothesis that selection alone is the remaining problem. "
            "The positive waypoint control indicates that the next useful model "
            "should represent an explicit multi-step avoidance/grasp route or "
            "residual expert, with the source frozen and a defined handoff.",
            "",
            "## Validation",
            "",
            "- maximum source-vs-captured action reproduction difference: "
            "{:.3e};".format(
                result["maximum_source_capture_reproduction_abs"]
            ),
            "- action clips: {};".format(result["action_clip_count"]),
            "- mean full-branch runtime: {:.2f} s.".format(
                result["mean_branch_wall_time_s"]
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states",
        type=Path,
        default=Path(
            "outputs/oracle_branch_value/pilot_3state/state_results.jsonl"
        ),
    )
    parser.add_argument(
        "--teacher-control",
        type=Path,
        default=Path(
            "outputs/oracle_branch_value/pilot_3state/teacher_control.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/oracle_branch_value/pilot_3state"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    states = load_jsonl(args.states)
    teacher_payload = (
        json.loads(args.teacher_control.read_text())
        if args.teacher_control.is_file()
        else None
    )
    result = analyze(states, teacher_payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_report(args.out_dir / "result_report.md", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
