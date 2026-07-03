"""
Fit simple offline adapters from real EEF delta to original OSC command.

This is phase 1 of the delta_eef -> OSC adapter diagnosis. It does not create
robosuite environments or run replay. It only answers whether the dataset's
actual EEF displacement can predict the original OSC command well enough to
justify a later open-loop replay test.

Run from repo root:
    uv run python docs/route_b_validation/fit_delta_eef_to_osc_adapter.py
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/route_b_validation/delta_eef_to_osc_adapter"


def _decode_demo_names(raw_names):
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names]


def load_split_arrays(dataset_path, split):
    """Load delta_eef_action and original actions for a split."""
    with h5py.File(dataset_path, "r") as f:
        if "mask" in f and split in f["mask"]:
            demo_names = _decode_demo_names(f["mask"][split][:])
        else:
            demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))

        delta_eef = []
        osc_actions = []
        for demo_name in demo_names:
            grp = f[f"data/{demo_name}"]
            delta_eef.append(grp["delta_eef_action"][:].astype(np.float64))
            osc_actions.append(grp["actions"][:].astype(np.float64))

    return {
        "split": split,
        "demo_names": demo_names,
        "delta_eef": np.concatenate(delta_eef, axis=0),
        "osc_actions": np.concatenate(osc_actions, axis=0),
    }


def fit_scalar_scale(x, y):
    denom = float(np.sum(x * x))
    scale = 0.0 if denom <= 1e-12 else float(np.sum(x * y) / denom)
    return {"kind": "scalar_scale", "scale": scale}


def fit_diagonal_scale(x, y):
    denom = np.sum(x * x, axis=0)
    numer = np.sum(x * y, axis=0)
    scale = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
    return {"kind": "diagonal_scale", "scale": scale.tolist()}


def fit_full_linear(x, y):
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    weights = coef[:3].T
    bias = coef[3]
    return {"kind": "full_linear", "weights": weights.tolist(), "bias": bias.tolist()}


def apply_adapter(params, delta_eef):
    """Convert full 7D delta_eef_action into full 7D OSC command."""
    x = delta_eef[:, :3]
    if params["kind"] == "scalar_scale":
        y = x * float(params["scale"])
    elif params["kind"] == "diagonal_scale":
        y = x * np.asarray(params["scale"], dtype=np.float64)
    elif params["kind"] == "full_linear":
        weights = np.asarray(params["weights"], dtype=np.float64)
        bias = np.asarray(params["bias"], dtype=np.float64)
        y = x @ weights.T + bias
    else:
        raise ValueError(f"Unknown adapter kind: {params['kind']}")

    out = delta_eef.copy()
    out[:, :3] = np.clip(y, -1.0, 1.0)
    return out


def vector_metrics(pred, target, source_delta=None):
    diff = pred - target
    pred_xyz = pred[:, :3]
    target_xyz = target[:, :3]
    pred_norm = np.linalg.norm(pred_xyz, axis=1)
    target_norm = np.linalg.norm(target_xyz, axis=1)
    dot = np.sum(pred_xyz * target_xyz, axis=1)
    cosine = dot / (pred_norm * target_norm + 1e-12)
    metrics = {
        "n_samples": int(pred.shape[0]),
        "mse_xyz": float(np.mean(diff[:, :3] ** 2)),
        "mae_xyz": float(np.mean(np.abs(diff[:, :3]))),
        "rmse_xyz": float(np.sqrt(np.mean(diff[:, :3] ** 2))),
        "mse_full": float(np.mean(diff ** 2)),
        "mae_full": float(np.mean(np.abs(diff))),
        "cosine_xyz_mean": float(np.mean(cosine)),
        "cosine_xyz_median": float(np.median(cosine)),
        "pred_xyz_norm_median": float(np.median(pred_norm)),
        "target_xyz_norm_median": float(np.median(target_norm)),
        "pred_xyz_norm_q90": float(np.quantile(pred_norm, 0.9)),
        "target_xyz_norm_q90": float(np.quantile(target_norm, 0.9)),
        "clip_rate_xyz": float(np.mean(np.abs(pred_xyz) >= 1.0 - 1e-9)),
        "clip_rate_any_xyz": float(np.mean(np.any(np.abs(pred_xyz) >= 1.0 - 1e-9, axis=1))),
        "per_axis_mae_xyz": np.mean(np.abs(diff[:, :3]), axis=0).tolist(),
        "per_axis_rmse_xyz": np.sqrt(np.mean(diff[:, :3] ** 2, axis=0)).tolist(),
    }
    if source_delta is not None:
        source_norm = np.linalg.norm(source_delta[:, :3], axis=1)
        ratio = target_norm / (source_norm + 1e-12)
        finite = np.isfinite(ratio)
        metrics["target_over_delta_norm_ratio_median"] = float(np.median(ratio[finite]))
        metrics["target_over_delta_norm_ratio_q10"] = float(np.quantile(ratio[finite], 0.1))
        metrics["target_over_delta_norm_ratio_q90"] = float(np.quantile(ratio[finite], 0.9))
    return metrics


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_summary(output_dir, dataset_path, params_by_name, metrics):
    best_name = min(metrics["valid"], key=lambda name: metrics["valid"][name]["mse_xyz"])
    lines = [
        "# Delta EEF to OSC Adapter: Phase 1",
        "",
        f"Dataset: `{dataset_path}`",
        "",
        "This phase only fits offline adapters. No environment replay was run.",
        "",
        "## Best Offline Adapter",
        "",
        f"Best by valid `mse_xyz`: `{best_name}`",
        "",
        "## Valid Metrics",
        "",
        "| adapter | mse_xyz | mae_xyz | cosine_median | clip_any_xyz | pred_norm_med | target_norm_med |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, vals in metrics["valid"].items():
        lines.append(
            "| {name} | {mse:.6g} | {mae:.6g} | {cos:.4f} | {clip:.4f} | {pn:.4f} | {tn:.4f} |".format(
                name=name,
                mse=vals["mse_xyz"],
                mae=vals["mae_xyz"],
                cos=vals["cosine_xyz_median"],
                clip=vals["clip_rate_any_xyz"],
                pn=vals["pred_xyz_norm_median"],
                tn=vals["target_xyz_norm_median"],
            )
        )
    lines.extend([
        "",
        "## Fitted Parameters",
        "",
        "```json",
        json.dumps(make_jsonable(params_by_name), indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "These metrics only test whether `delta_eef_action[:3]` predicts the dataset's original OSC `actions[:3]`.",
        "The adapter is not validated for use until the next phase performs open-loop replay through robosuite.",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split_arrays(dataset_path, "train")
    valid = load_split_arrays(dataset_path, "valid")

    x_train = train["delta_eef"][:, :3]
    y_train = train["osc_actions"][:, :3]

    params_by_name = {
        "scalar_scale": fit_scalar_scale(x_train, y_train),
        "diagonal_scale": fit_diagonal_scale(x_train, y_train),
        "full_linear": fit_full_linear(x_train, y_train),
    }

    metrics = {
        "dataset": str(dataset_path),
        "splits": {
            "train": {"n_demos": len(train["demo_names"]), "n_samples": int(train["delta_eef"].shape[0])},
            "valid": {"n_demos": len(valid["demo_names"]), "n_samples": int(valid["delta_eef"].shape[0])},
        },
        "train": {},
        "valid": {},
    }

    for split_name, split_data in (("train", train), ("valid", valid)):
        for adapter_name, params in params_by_name.items():
            pred = apply_adapter(params, split_data["delta_eef"])
            metrics[split_name][adapter_name] = vector_metrics(
                pred=pred,
                target=split_data["osc_actions"],
                source_delta=split_data["delta_eef"],
            )

    best_name = min(metrics["valid"], key=lambda name: metrics["valid"][name]["mse_xyz"])
    best_params = params_by_name[best_name]

    (output_dir / "adapter_params.json").write_text(
        json.dumps(
            {
                "best_adapter": best_name,
                "best_params": make_jsonable(best_params),
                "all_params": make_jsonable(params_by_name),
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "metrics.json").write_text(json.dumps(make_jsonable(metrics), indent=2) + "\n")
    write_summary(output_dir, dataset_path, params_by_name, metrics)

    print(f"Wrote adapter params to {output_dir / 'adapter_params.json'}")
    print(f"Wrote metrics to {output_dir / 'metrics.json'}")
    print(f"Wrote summary to {output_dir / 'summary.md'}")
    print("\nValid metrics:")
    for name, vals in metrics["valid"].items():
        print(
            f"  {name:<15} mse_xyz={vals['mse_xyz']:.6g} "
            f"mae_xyz={vals['mae_xyz']:.6g} "
            f"cos_med={vals['cosine_xyz_median']:.4f} "
            f"clip_any={vals['clip_rate_any_xyz']:.4f}"
        )
    print(f"\nBest adapter by valid mse_xyz: {best_name}")


if __name__ == "__main__":
    main()
