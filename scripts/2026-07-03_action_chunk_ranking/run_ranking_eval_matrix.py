"""Launch controlled rollout comparisons for action-chunk ranking."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARALLEL_SCRIPT = ROOT / "third_party" / "robomimic" / "robomimic" / "scripts" / "run_obstacle_guided_agent_parallel.py"


def condition_args(name, args):
    common = [
        "--guidance_geometry_source", "oracle_center",
        "--guidance_mode", args.guidance_mode,
        "--xy_clearance", str(args.xy_clearance),
        "--z_clearance", str(args.z_clearance),
        "--guidance_horizon", str(args.guidance_horizon),
        "--target_object_name", args.target_object_name,
    ]
    if args.obstacle_names:
        common += ["--obstacle_names"] + list(args.obstacle_names)

    if name == "no_guidance":
        return common + ["--selection_mode", "none", "--trajectory_backend", "cumsum", "--guidance_scale", "0.0"]
    if name == "cumsum_ranking":
        extra = [
            "--selection_mode", "ranking",
            "--trajectory_backend", "cumsum",
            "--ranking_num_candidates", str(args.ranking_num_candidates),
            "--guidance_scale", "0.0",
        ]
        return extra + common + (["--ranking_only_if_first_unsafe"] if args.ranking_only_if_first_unsafe else [])
    if name == "forward_model_ranking":
        extra = [
            "--selection_mode", "ranking",
            "--trajectory_backend", "forward_model",
            "--forward_model_path", args.forward_model_path,
            "--ranking_num_candidates", str(args.ranking_num_candidates),
            "--guidance_scale", "0.0",
        ]
        return extra + common + (["--ranking_only_if_first_unsafe"] if args.ranking_only_if_first_unsafe else [])
    if name == "forward_model_gradient":
        return common + [
            "--selection_mode", "gradient",
            "--trajectory_backend", "forward_model",
            "--forward_model_path", args.forward_model_path,
            "--guidance_scale", str(args.gradient_guidance_scale),
            "--guidance_schedule", args.guidance_schedule,
            "--guidance_start_step_pct", str(args.guidance_start_step_pct),
        ]
    raise ValueError("Unknown condition '{}'".format(name))


def run_command(cmd, dry_run=False):
    print(shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/model_epoch_140_image_v15_can_mask_success_1.0.pth",
    )
    parser.add_argument("--forward_model_path", type=str, default="outputs/forward_model/osc_eef_forward_image_v15/model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/robomimic/eval/action_chunk_ranking")
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=["PickPlaceBreadCerealCan", "PickPlaceBreadCerealMilkCan"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[700, 701, 702])
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["no_guidance", "cumsum_ranking", "forward_model_ranking", "forward_model_gradient"],
    )
    parser.add_argument("--n_rollouts", type=int, default=20)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--ranking_num_candidates", type=int, default=16)
    parser.add_argument("--ranking_only_if_first_unsafe", action="store_true")
    parser.add_argument("--guidance_mode", type=str, choices=["xy", "xyz_cylinder"], default="xy")
    parser.add_argument("--xy_clearance", type=float, default=0.02)
    parser.add_argument("--z_clearance", type=float, default=0.03)
    parser.add_argument("--guidance_horizon", type=int, default=8)
    parser.add_argument("--gradient_guidance_scale", type=float, default=0.005)
    parser.add_argument("--guidance_schedule", type=str, default="late")
    parser.add_argument("--guidance_start_step_pct", type=float, default=0.7)
    parser.add_argument("--target_object_name", type=str, default="Can")
    parser.add_argument("--obstacle_names", type=str, nargs="*", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main(args):
    output_dir = Path(args.output_dir)
    for env_name in args.envs:
        for condition in args.conditions:
            for seed in args.seeds:
                stats_path = output_dir / env_name / condition / "seed_{}.json".format(seed)
                cmd = [
                    sys.executable,
                    str(PARALLEL_SCRIPT),
                    "--agent", args.agent,
                    "--env", env_name,
                    "--n_rollouts", str(args.n_rollouts),
                    "--n_workers", str(args.n_workers),
                    "--seed", str(seed),
                    "--stats_path", str(stats_path),
                ]
                if args.horizon is not None:
                    cmd += ["--horizon", str(args.horizon)]
                cmd += condition_args(condition, args)
                run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main(parse_args())
