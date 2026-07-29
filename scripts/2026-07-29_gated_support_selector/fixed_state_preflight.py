"""Preflight the gated selector on the 20 held-out pre-collision states."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_DIR = SCRIPT_DIR.parent / "guided_denoising"
MANIFOLD_DIR = SCRIPT_DIR.parent / "2026-07-27_policy_manifold_diagnostic"
for directory in (SCRIPT_DIR, GUIDED_DIR, MANIFOLD_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common as guided_common  # noqa: E402
from run_fixed_state import DEFAULT_STATE_BANKS, read_state  # noqa: E402
from runtime import choose_action_chunk  # noqa: E402
from selector import SelectorConfig  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


DEFAULT_CORRECTIVE_AGENT = Path(
    "outputs/robomimic/train/dp_can_delta_pose_osc_targeted_sdp/"
    "20260729173254/models/model_epoch_12.pth"
)


def summarize(rows):
    triggered = [row for row in rows if row["triggered"]]
    selected = [row for row in rows if row["mode"] == "corrective"]
    return {
        "states": len(rows),
        "triggered_states": len(triggered),
        "trigger_rate": len(triggered) / len(rows) if rows else None,
        "states_with_eligible_corrective_candidate": len(selected),
        "selection_rate_all_states": len(selected) / len(rows) if rows else None,
        "selection_rate_triggered_states": (
            len(selected) / len(triggered) if triggered else None
        ),
        "total_eligible_candidates": int(
            sum(row["eligible_candidate_count"] for row in rows)
        ),
        "median_selected_clearance_m": (
            float(np.median([row["selected_clearance_m"] for row in selected]))
            if selected
            else None
        ),
        "median_selected_progress_m": (
            float(np.median([row["selected_progress_m"] for row in selected]))
            if selected
            else None
        ),
        "median_selected_normalized_rms_deviation": (
            float(
                np.median(
                    [row["selected_normalized_rms_deviation"] for row in selected]
                )
            )
            if selected
            else None
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-agent", type=Path, default=guided_common.DEFAULT_AGENT)
    parser.add_argument(
        "--corrective-agent", type=Path, default=DEFAULT_CORRECTIVE_AGENT
    )
    parser.add_argument("--states", nargs="+", type=Path, default=DEFAULT_STATE_BANKS)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--trigger-clearance", type=float, default=0.01)
    parser.add_argument("--minimum-start-clearance", type=float, default=0.01)
    parser.add_argument("--candidate-clearance", type=float, default=0.01)
    parser.add_argument("--progress-tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--progress-rule",
        choices=("nonretreat", "source_relative"),
        default="nonretreat",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/gated_support_selector/fixed_state_preflight"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError("{} exists; pass --overwrite".format(args.out_dir))
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    config = SelectorConfig(
        trigger_clearance_m=args.trigger_clearance,
        minimum_start_clearance_m=args.minimum_start_clearance,
        candidate_clearance_m=args.candidate_clearance,
        progress_tolerance_m=args.progress_tolerance,
        progress_rule=args.progress_rule,
    )

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    source_policy, source_ckpt = guided_common.load_guided_policy_and_checkpoint(
        args.source_agent, device
    )
    corrective_policy, _ = guided_common.load_guided_policy_and_checkpoint(
        args.corrective_agent, device
    )
    if (
        source_policy.policy.global_config.all_obs_keys
        != corrective_policy.policy.global_config.all_obs_keys
    ):
        raise ValueError("Source and corrective policies require different observations")
    envs = {}
    rows = []
    try:
        for states_path in args.states:
            with h5py.File(states_path, "r") as input_file:
                bank_name = states_path.parent.name
                for key in sorted(input_file["states"]):
                    saved = read_state(input_file["states"][key], bank_name)
                    if saved["env"] not in envs:
                        envs[saved["env"]], _ = FileUtils.env_from_checkpoint(
                            ckpt_dict=source_ckpt,
                            env_name=saved["env"],
                            render=False,
                            render_offscreen=False,
                            verbose=False,
                        )
                    env = envs[saved["env"]]
                    source_policy.start_episode()
                    corrective_policy.start_episode()
                    env.reset_to(saved["state"])
                    raw_env = guided_common.raw_env_from_wrapper(env)
                    records = guided_common.active_object_records(raw_env)
                    candidate_seeds = [saved["noise_seed"]]
                    candidate_seeds.extend(
                        saved["noise_seed"] + 100_000 + index
                        for index in range(args.candidate_count - 1)
                    )
                    _, diagnostic = choose_action_chunk(
                        common=guided_common,
                        source_policy=source_policy,
                        corrective_policy=corrective_policy,
                        obs=saved["obs"],
                        raw_env=raw_env,
                        object_records=records,
                        target_object_name="Can",
                        nominal_seed=saved["noise_seed"],
                        candidate_seeds=candidate_seeds,
                        target_contact_seen=False,
                        obstacle_contact_seen=False,
                        config=config,
                    )
                    compact = {
                        key: value
                        for key, value in diagnostic.items()
                        if key != "candidate_metrics"
                    }
                    compact.update(
                        {
                            "state_id": saved["state_id"],
                            "env": saved["env"],
                            "seed": saved["seed"],
                            "episode": saved["episode"],
                            "chunk": saved["chunk"],
                            "start_step": saved["start_step"],
                        }
                    )
                    rows.append(compact)
                    print(
                        "{} trigger={} mode={} eligible={}".format(
                            saved["state_id"],
                            diagnostic["triggered"],
                            diagnostic["mode"],
                            diagnostic["eligible_candidate_count"],
                        ),
                        flush=True,
                    )
    finally:
        for env in envs.values():
            close = getattr(guided_common.raw_env_from_wrapper(env), "close", None)
            if close is not None:
                close()

    summary = summarize(rows)
    with (args.out_dir / "state_metrics.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(guided_common.json_safe(row)) + "\n")
    manifest = {
        "source_agent": str(args.source_agent),
        "corrective_agent": str(args.corrective_agent),
        "state_banks": [str(path) for path in args.states],
        "candidate_count": args.candidate_count,
        "selector": config.to_dict(),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "summary.json").write_text(
        json.dumps(guided_common.json_safe(summary), indent=2) + "\n"
    )
    print(json.dumps(guided_common.json_safe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
