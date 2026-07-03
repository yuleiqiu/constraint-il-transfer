"""
Train a state-conditioned MLP adapter from real EEF delta to OSC command.

The adapter learns:
    [eef_pos, eef_quat, joint_pos, joint_vel, delta_eef_xyz] -> osc_action_xyz

Rotation and gripper are intentionally not learned in this phase, since
delta_eef_action[:, 3:6] and delta_eef_action[:, 6] already match the original
rotation and gripper command fields.

Run from repo root:
    uv run python docs/route_b_validation/train_delta_eef_to_osc_mlp_adapter.py
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fit_delta_eef_to_osc_adapter import vector_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/route_b_validation/delta_eef_to_osc_adapter"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "mlp_adapter.pth"

FEATURE_KEYS = (
    "obs/robot0_eef_pos",
    "obs/robot0_eef_quat",
    "obs/robot0_joint_pos",
    "obs/robot0_joint_vel",
)


class DeltaEEFToOSCMLP(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def decode_demo_names(raw_names):
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names]


def get_split_demos(h5_file, split):
    if "mask" in h5_file and split in h5_file["mask"]:
        return decode_demo_names(h5_file["mask"][split][:])
    return sorted(h5_file["data"].keys(), key=lambda x: int(x.split("_")[-1]))


def load_split(dataset_path, split):
    features = []
    targets = []
    delta_eef = []
    with h5py.File(dataset_path, "r") as f:
        demo_names = get_split_demos(f, split)
        for demo_name in demo_names:
            grp = f[f"data/{demo_name}"]
            parts = [grp[key][:].astype(np.float32) for key in FEATURE_KEYS]
            de = grp["delta_eef_action"][:, :3].astype(np.float32)
            action = grp["actions"][:, :3].astype(np.float32)
            features.append(np.concatenate(parts + [de], axis=1))
            targets.append(action)
            delta_eef.append(grp["delta_eef_action"][:].astype(np.float32))
    return {
        "split": split,
        "demo_names": demo_names,
        "features": np.concatenate(features, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "delta_eef": np.concatenate(delta_eef, axis=0),
    }


def normalize(x, mean, std):
    return (x - mean) / std


def predict_osc_actions(model, features, delta_eef, stats, device, batch_size=8192):
    model.eval()
    x = normalize(features, stats["input_mean"], stats["input_std"])
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).float().to(device)
            pred_norm = model(xb).cpu().numpy()
            preds.append(pred_norm)
    pred_xyz = np.concatenate(preds, axis=0) * stats["target_std"] + stats["target_mean"]
    out = delta_eef.astype(np.float64).copy()
    out[:, :3] = np.clip(pred_xyz, -1.0, 1.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(dataset_path, "train")
    valid = load_split(dataset_path, "valid")

    input_mean = train["features"].mean(axis=0, keepdims=True)
    input_std = train["features"].std(axis=0, keepdims=True)
    input_std = np.maximum(input_std, 1e-6)
    target_mean = train["targets"].mean(axis=0, keepdims=True)
    target_std = train["targets"].std(axis=0, keepdims=True)
    target_std = np.maximum(target_std, 1e-6)

    stats = {
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
    }

    x_train = normalize(train["features"], stats["input_mean"], stats["input_std"]).astype(np.float32)
    y_train = normalize(train["targets"], stats["target_mean"], stats["target_std"]).astype(np.float32)
    x_valid = normalize(valid["features"], stats["input_mean"], stats["input_std"]).astype(np.float32)
    y_valid = normalize(valid["targets"], stats["target_mean"], stats["target_std"]).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeltaEEFToOSCMLP(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    x_valid_t = torch.from_numpy(x_valid).to(device)
    y_valid_t = torch.from_numpy(y_valid).to(device)

    best = {
        "epoch": -1,
        "valid_loss": float("inf"),
        "state_dict": None,
    }
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(x_valid_t), y_valid_t).item())
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

        if valid_loss < best["valid_loss"]:
            best = {
                "epoch": epoch,
                "valid_loss": valid_loss,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:04d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")

        if stale >= args.patience:
            print(f"early stopping at epoch {epoch} (best epoch {best['epoch']})")
            break

    model.load_state_dict(best["state_dict"])

    train_pred = predict_osc_actions(model, train["features"], train["delta_eef"], stats, device)
    valid_pred = predict_osc_actions(model, valid["features"], valid["delta_eef"], stats, device)

    train_target = train["delta_eef"].astype(np.float64).copy()
    train_target[:, :3] = train["targets"]
    valid_target = valid["delta_eef"].astype(np.float64).copy()
    valid_target[:, :3] = valid["targets"]

    metrics = {
        "dataset": str(dataset_path),
        "device": str(device),
        "splits": {
            "train": {"n_demos": len(train["demo_names"]), "n_samples": int(train["features"].shape[0])},
            "valid": {"n_demos": len(valid["demo_names"]), "n_samples": int(valid["features"].shape[0])},
        },
        "config": {
            "feature_keys": FEATURE_KEYS,
            "input_dim": int(x_train.shape[1]),
            "hidden_dim": args.hidden_dim,
            "epochs_requested": args.epochs,
            "best_epoch": best["epoch"],
            "best_valid_loss_normalized": best["valid_loss"],
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
        },
        "train": vector_metrics(train_pred, train_target, train["delta_eef"]),
        "valid": vector_metrics(valid_pred, valid_target, valid["delta_eef"]),
        "history": history,
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "DeltaEEFToOSCMLP",
        "feature_keys": FEATURE_KEYS,
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": args.hidden_dim,
        "output_dim": 3,
        "input_mean": stats["input_mean"],
        "input_std": stats["input_std"],
        "target_mean": stats["target_mean"],
        "target_std": stats["target_std"],
        "metrics": metrics,
    }
    torch.save(checkpoint, model_path)

    metrics_path = output_dir / "mlp_adapter_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    summary_path = output_dir / "mlp_adapter_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Delta EEF to OSC MLP Adapter",
                "",
                f"Dataset: `{dataset_path}`",
                f"Model: `{model_path}`",
                f"Device: `{device}`",
                "",
                "## Offline Metrics",
                "",
                "| split | mse_xyz | mae_xyz | cosine_median | clip_any_xyz |",
                "|---|---:|---:|---:|---:|",
                "| train | {mse:.6g} | {mae:.6g} | {cos:.4f} | {clip:.4f} |".format(
                    mse=metrics["train"]["mse_xyz"],
                    mae=metrics["train"]["mae_xyz"],
                    cos=metrics["train"]["cosine_xyz_median"],
                    clip=metrics["train"]["clip_rate_any_xyz"],
                ),
                "| valid | {mse:.6g} | {mae:.6g} | {cos:.4f} | {clip:.4f} |".format(
                    mse=metrics["valid"]["mse_xyz"],
                    mae=metrics["valid"]["mae_xyz"],
                    cos=metrics["valid"]["cosine_xyz_median"],
                    clip=metrics["valid"]["clip_rate_any_xyz"],
                ),
                "",
                "Offline metrics are not sufficient; this model still requires open-loop replay validation.",
            ]
        )
        + "\n"
    )

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved summary to {summary_path}")
    print(
        "valid: mse_xyz={mse:.6g} mae_xyz={mae:.6g} cos_med={cos:.4f} clip_any={clip:.4f}".format(
            mse=metrics["valid"]["mse_xyz"],
            mae=metrics["valid"]["mae_xyz"],
            cos=metrics["valid"]["cosine_xyz_median"],
            clip=metrics["valid"]["clip_rate_any_xyz"],
        )
    )


if __name__ == "__main__":
    main()
