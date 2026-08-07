"""Screen fixed Empty2D source checkpoints on D0 and enforce the source gate."""

import argparse
import json
import re
from pathlib import Path

import torch

from empty2d_guidance.evaluation import load_guidable_policy, rollout
from empty2d_guidance.metrics import summarize


EPOCH_PATTERN = re.compile(r"model_epoch_(100|200|300|400|500)(?:_|\.pth)")


def validation_losses(log_path: Path) -> dict[int, float]:
    if not log_path.exists():
        return {}
    lines = log_path.read_text(errors="replace").splitlines()
    result = {}
    for index, line in enumerate(lines):
        match = re.fullmatch(r"Validation Epoch (\d+)", line.strip())
        if not match:
            continue
        payload = []
        depth = 0
        started = False
        for candidate in lines[index + 1 :]:
            if "{" in candidate:
                started = True
            if started:
                payload.append(candidate)
                depth += candidate.count("{") - candidate.count("}")
                if depth == 0:
                    break
        try:
            parsed = json.loads("\n".join(payload))
        except (ValueError, json.JSONDecodeError):
            continue
        if "Loss" in parsed:
            result[int(match.group(1))] = float(parsed["Loss"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/empty2d_guidance/source_screen.json"))
    parser.add_argument(
        "--device",
        help="Explicit torch device such as cuda:1 (default: cuda if available).",
    )
    args = parser.parse_args()
    checkpoints = []
    for path in (args.run_dir / "models").glob("model_epoch_*.pth"):
        match = EPOCH_PATTERN.search(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if [epoch for epoch, _ in checkpoints] != [100, 200, 300, 400, 500]:
        raise RuntimeError("Expected exactly the five formal checkpoint epochs")
    losses = validation_losses(args.run_dir / "logs/log.txt")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    seeds = list(range(420_000, 420_050))
    for epoch, checkpoint in checkpoints:
        policy, _ = load_guidable_policy(checkpoint, device)
        rollouts = [
            rollout(
                policy,
                environment_name="empty",
                policy_seed=seed,
                guidance_scale=0.0,
                safety_distance=0.02,
            )
            for seed in seeds
        ]
        metrics = summarize(rollouts)
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint.resolve()),
                "validation_loss": losses.get(epoch, float("inf")),
                **metrics,
            }
        )
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected = min(
        rows,
        key=lambda row: (
            -row["task_sr"],
            row["ncr"],
            row["validation_loss"],
            row["epoch"],
        ),
    )
    source_gate = {
        "task_sr_at_least_0.95": selected["task_sr"] >= 0.95,
        "cr_zero": selected["cr"] == 0.0,
        "clip_rate_at_most_0.01": selected["mean_clip_rate"] <= 0.01,
    }
    output = {
        "protocol": {"environment": "empty", "policy_seeds": seeds},
        "checkpoints": rows,
        "selected": selected,
        "source_gate": source_gate,
        "passed": all(source_gate.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n")
    print(json.dumps(output, indent=2, allow_nan=True))
    if not output["passed"]:
        raise SystemExit("Source gate failed; guidance calibration is forbidden")


if __name__ == "__main__":
    main()
