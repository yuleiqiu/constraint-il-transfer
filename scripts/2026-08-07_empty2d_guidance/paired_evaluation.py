"""Calibrate and test Empty2D point guidance with fixed paired seeds."""

import argparse
import json
from pathlib import Path

import torch

from empty2d_guidance.evaluation import load_guidable_policy, rollout, write_jsonl
from empty2d_guidance.metrics import paired_summary, summarize


RHO_VALUES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
Q_VALUES = (0.02, 0.04, 0.08, 0.12)
CALIBRATION_SEEDS = list(range(421_000, 421_020))
TEST_SEEDS = list(range(422_000, 422_100))


def evaluate_grid(checkpoint, environment, device):
    baseline_policy, _ = load_guidable_policy(checkpoint, device)
    guided_policy, _ = load_guidable_policy(checkpoint, device)
    baseline = [
        rollout(
            baseline_policy,
            environment_name=environment,
            policy_seed=seed,
            guidance_scale=0.0,
            safety_distance=Q_VALUES[0],
        )
        for seed in CALIBRATION_SEEDS
    ]
    candidates = []
    raw = list(baseline)
    for rho in RHO_VALUES:
        for q_value in Q_VALUES:
            guided = [
                rollout(
                    guided_policy,
                    environment_name=environment,
                    policy_seed=seed,
                    guidance_scale=rho,
                    safety_distance=q_value,
                )
                for seed in CALIBRATION_SEEDS
            ]
            raw.extend(guided)
            paired = paired_summary(baseline, guided)
            candidates.append({"rho": rho, "q": q_value, **paired})
    selected = min(
        candidates,
        key=lambda row: (
            -row["guided"]["safe_sr"],
            -row["guided"]["task_sr"],
            row["guided"]["cr"],
            row["rho"],
            row["q"],
        ),
    )
    return raw, candidates, {"rho": selected["rho"], "q": selected["q"]}


def held_out_test(checkpoint, environment, parameters, device):
    baseline_policy, _ = load_guidable_policy(checkpoint, device)
    guided_policy, _ = load_guidable_policy(checkpoint, device)
    baseline, guided = [], []
    for seed in TEST_SEEDS:
        baseline.append(
            rollout(
                baseline_policy,
                environment_name=environment,
                policy_seed=seed,
                guidance_scale=0.0,
                safety_distance=parameters["q"],
                record_chunks=True,
            )
        )
        guided.append(
            rollout(
                guided_policy,
                environment_name=environment,
                policy_seed=seed,
                guidance_scale=parameters["rho"],
                safety_distance=parameters["q"],
                record_chunks=True,
            )
        )
    paired = paired_summary(baseline, guided)
    paired["cr_percentage_point_drop"] = 100 * (
        paired["baseline"]["cr"] - paired["guided"]["cr"]
    )
    paired["gate"] = {
        "net_rescues_at_least_10": paired["net_rescues"] >= 10,
        "cr_drop_at_least_10pp": paired["cr_percentage_point_drop"] >= 10.0 - 1e-9,
        "exact_mcnemar_below_0.05": paired["exact_mcnemar_p"] < 0.05,
    }
    paired["passed"] = all(paired["gate"].values())
    return baseline, guided, paired


def conclusion(results):
    passed = [name for name, result in results.items() if result["passed"]]
    if len(passed) == 2:
        return "Guidance robustly works in both idealized point geometries."
    if len(passed) == 1:
        return f"Shape-dependent partial pass: {passed[0]} only."
    return "Current guidance is not validated even in the simplified environments."


def write_report(path, seed, calibration, tests, final_conclusion):
    lines = [
        "# Empty2D guidance paired evaluation",
        "",
        f"Training seed: {seed}",
        "",
        "| Geometry | rho | Q | Base Task SR | Guided Task SR | Base CR | Guided CR | F→S | S→F | exact p | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for geometry in ("circle", "square"):
        result = tests[geometry]
        parameters = calibration[geometry]
        lines.append(
            "| {g} | {rho:g} | {q:g} | {bt:.2f} | {gt:.2f} | {bc:.2f} | {gc:.2f} | {fs} | {sf} | {p:.4g} | {passed} |".format(
                g=geometry,
                rho=parameters["rho"],
                q=parameters["q"],
                bt=result["baseline"]["task_sr"],
                gt=result["guided"]["task_sr"],
                bc=result["baseline"]["cr"],
                gc=result["guided"]["cr"],
                fs=result["failure_to_success"],
                sf=result["success_to_failure"],
                p=result["exact_mcnemar_p"],
                passed="yes" if result["passed"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            final_conclusion,
            "",
            "Even a two-geometry pass establishes only that the implementation and predicted-clean gradient work in idealized point geometry. It supports, but does not prove, the hypothesis that D2 failed because an EEF-point cost mismatched swept gripper/object geometry; it cannot establish complex geometry as the unique cause.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--source-screen", type=Path, default=Path("outputs/empty2d_guidance/source_screen.json"))
    parser.add_argument("--locked-parameters", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--environment",
        choices=("circle", "square", "both"),
        default="both",
        help="Run one geometry for two-GPU sharding, or both sequentially.",
    )
    parser.add_argument(
        "--device",
        help="Explicit torch device such as cuda:0 or cuda:1 (default: cuda if available).",
    )
    args = parser.parse_args()
    screen = json.loads(args.source_screen.read_text())
    if not screen["passed"]:
        raise RuntimeError("Source gate has not passed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    environments = (
        ("circle", "square") if args.environment == "both" else (args.environment,)
    )
    calibration = {}
    if args.locked_parameters is None:
        if args.training_seed != 42:
            raise ValueError("Only seed 42 may select calibration parameters")
        for environment in environments:
            raw, candidates, selected = evaluate_grid(args.checkpoint, environment, device)
            write_jsonl(args.out_dir / f"calibration_{environment}.jsonl", raw)
            (args.out_dir / f"calibration_{environment}_summary.json").write_text(
                json.dumps({"candidates": candidates, "selected": selected}, indent=2, allow_nan=True) + "\n"
            )
            calibration[environment] = selected
        parameter_path = (
            args.out_dir / "locked_parameters.json"
            if args.environment == "both"
            else args.out_dir / f"locked_parameters_{args.environment}.json"
        )
        parameter_path.write_text(json.dumps(calibration, indent=2) + "\n")
    else:
        calibration = json.loads(args.locked_parameters.read_text())
        missing = set(environments) - set(calibration)
        if missing:
            raise ValueError(f"Locked parameters missing geometries: {sorted(missing)}")

    tests = {}
    for environment in environments:
        baseline, guided, result = held_out_test(
            args.checkpoint,
            environment,
            calibration[environment],
            device,
        )
        write_jsonl(args.out_dir / f"test_{environment}_baseline.jsonl", baseline)
        write_jsonl(args.out_dir / f"test_{environment}_guided.jsonl", guided)
        tests[environment] = result
    final_conclusion = (
        conclusion(tests)
        if args.environment == "both"
        else f"{args.environment} {'passed' if tests[args.environment]['passed'] else 'failed'} its mechanism gate."
    )
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "training_seed": args.training_seed,
        "calibration_seeds": CALIBRATION_SEEDS,
        "test_seeds": TEST_SEEDS,
        "parameters": calibration,
        "tests": tests,
        "conclusion": final_conclusion,
    }
    result_path = (
        args.out_dir / "result.json"
        if args.environment == "both"
        else args.out_dir / f"result_{args.environment}.json"
    )
    result_path.write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    if args.environment == "both":
        write_report(args.out_dir / "report.md", args.training_seed, calibration, tests, final_conclusion)
    print(json.dumps(manifest, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
