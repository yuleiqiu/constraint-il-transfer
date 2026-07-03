"""
Plot phase-2 replay results for delta_eef -> OSC adapters.

Run from repo root:
    uv run python docs/route_b_validation/plot_delta_eef_to_osc_adapter_replay.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "outputs/route_b_validation/delta_eef_to_osc_adapter/replay_results.json"
DEFAULT_FIGURE_DIR = REPO_ROOT / "outputs/route_b_validation/delta_eef_to_osc_adapter/figures"

PLAN_LABELS = {
    "plan_A_original_osc": "Plan A: original OSC",
    "plan_B1_raw_delta_eef": "B1: raw delta EEF",
    "adapter_scalar_scale": "Adapter: scalar",
    "adapter_diagonal_scale": "Adapter: diagonal",
    "adapter_full_linear": "Adapter: full linear",
    "adapter_mlp_state_conditioned": "Adapter: MLP state",
}

PLAN_COLORS = {
    "plan_A_original_osc": "#2ca02c",
    "plan_B1_raw_delta_eef": "#d62728",
    "adapter_scalar_scale": "#1f77b4",
    "adapter_diagonal_scale": "#9467bd",
    "adapter_full_linear": "#ff7f0e",
    "adapter_mlp_state_conditioned": "#17becf",
}


def pad_stack(sequences):
    max_len = max(len(seq) for seq in sequences)
    out = np.full((len(sequences), max_len), np.nan)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = seq
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    with args.results.open() as f:
        data = json.load(f)

    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = data["results"]
    summary = data["summary"]

    plt.figure(figsize=(10, 6))
    for plan, demos in results.items():
        arr = pad_stack([demo["err_to_target_per_step_cm"] for demo in demos])
        mean = np.nanmean(arr, axis=0)
        p25 = np.nanpercentile(arr, 25, axis=0)
        p75 = np.nanpercentile(arr, 75, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=PLAN_LABELS.get(plan, plan), color=PLAN_COLORS.get(plan))
        plt.fill_between(x, p25, p75, color=PLAN_COLORS.get(plan), alpha=0.15)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("EEF error to next_obs target (cm, log scale)")
    plt.title("Delta EEF -> OSC Adapter Replay Error")
    plt.legend()
    plt.tight_layout()
    path = figure_dir / "err_to_target_per_step.png"
    plt.savefig(path, dpi=200)
    plt.close()

    plans = list(summary.keys())
    values = [summary[p]["end_err_to_orig_cm"] for p in plans]
    colors = [PLAN_COLORS.get(p, "#333333") for p in plans]
    labels = [PLAN_LABELS.get(p, p) for p in plans]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(plans)), values, color=colors)
    plt.xticks(np.arange(len(plans)), labels, rotation=25, ha="right")
    plt.ylabel("mean end error to recorded EEF (cm)")
    plt.title("Open-Loop Replay End Error")
    plt.tight_layout()
    path_bar = figure_dir / "end_err_to_orig_bar.png"
    plt.savefig(path_bar, dpi=200)
    plt.close()

    print(f"Wrote {path}")
    print(f"Wrote {path_bar}")


if __name__ == "__main__":
    main()
