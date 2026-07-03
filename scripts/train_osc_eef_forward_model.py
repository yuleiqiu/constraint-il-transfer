"""
Train a differentiable forward model for OSC command -> EEF trajectory.

Run from repo root:
    uv run python scripts/train_osc_eef_forward_model.py \
        --config configs/forward_model/osc_eef_forward_image_v15.json
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/forward_model/osc_eef_forward_image_v15.json"


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def decode_demo_names(raw_names):
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names]


def get_split_demos(h5_file, split):
    if "mask" in h5_file and split in h5_file["mask"]:
        return decode_demo_names(h5_file["mask"][split][:])
    return sorted(h5_file["data"].keys(), key=lambda x: int(x.split("_")[-1]))


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def resolve_horizon(config):
    if config.get("horizon") != "auto":
        return int(config["horizon"])
    dp_config = load_json(resolve_path(config["dp_config"]))
    return int(dp_config["algo"]["horizon"]["prediction_horizon"])


def load_split(dataset_path, split, horizon, state_keys, action_key, target_key):
    states = []
    actions = []
    targets = []
    demo_names_out = []

    with h5py.File(dataset_path, "r") as f:
        demo_names = get_split_demos(f, split)
        for demo_name in demo_names:
            grp = f[f"data/{demo_name}"]
            n_steps = grp[action_key].shape[0]
            if n_steps < horizon:
                continue

            state_parts = [grp[key][:].astype(np.float32) for key in state_keys]
            state_arr = np.concatenate(state_parts, axis=1)
            action_arr = grp[action_key][:].astype(np.float32)
            eef_pos = grp["obs/robot0_eef_pos"][:].astype(np.float32)
            next_eef_pos = grp[target_key][:].astype(np.float32)

            for start in range(0, n_steps - horizon + 1):
                states.append(state_arr[start])
                actions.append(action_arr[start : start + horizon].reshape(-1))
                targets.append(next_eef_pos[start : start + horizon] - eef_pos[start])

            demo_names_out.append(demo_name)

    return {
        "split": split,
        "demo_names": demo_names_out,
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
    }


def normalize(x, mean, std):
    return (x - mean) / std


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class OSCForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, horizon, state_embed_dim=128, action_embed_dim=256, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.horizon = int(horizon)
        self.state_net = nn.Sequential(
            MLPBlock(state_dim, 64, dropout),
            MLPBlock(64, state_embed_dim, dropout),
        )
        self.action_net = nn.Sequential(
            MLPBlock(action_dim, action_embed_dim, dropout),
            MLPBlock(action_embed_dim, action_embed_dim, dropout),
        )
        fusion_dim = state_embed_dim + action_embed_dim
        self.fusion_net = nn.Sequential(
            MLPBlock(fusion_dim, hidden_dim, dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout),
            MLPBlock(hidden_dim, 256, dropout),
            nn.Linear(256, self.horizon * 3),
        )

    def forward(self, state, action):
        state_emb = self.state_net(state)
        action_emb = self.action_net(action)
        out = self.fusion_net(torch.cat([state_emb, action_emb], dim=-1))
        return out.view(out.shape[0], self.horizon, 3)


def prediction_metrics(pred, target):
    err = pred - target
    l2 = np.linalg.norm(err, axis=-1)
    per_step_rmse = np.sqrt(np.mean(np.sum(err**2, axis=-1), axis=0))
    return {
        "n_samples": int(pred.shape[0]),
        "traj_rmse_cm": float(np.sqrt(np.mean(np.sum(err**2, axis=-1))) * 100.0),
        "traj_mae_cm": float(np.mean(np.abs(err)) * 100.0),
        "terminal_error_mean_cm": float(np.mean(l2[:, -1]) * 100.0),
        "terminal_error_median_cm": float(np.median(l2[:, -1]) * 100.0),
        "terminal_error_p90_cm": float(np.quantile(l2[:, -1], 0.9) * 100.0),
        "per_step_rmse_cm": [float(x * 100.0) for x in per_step_rmse],
    }


def fit_cumsum_baselines(train_split, horizon):
    actions = train_split["actions"].reshape(-1, horizon, 7)[:, :, :3].reshape(-1, 3)
    target_traj = train_split["targets"]
    target_deltas = np.concatenate(
        [target_traj[:, :1, :], target_traj[:, 1:, :] - target_traj[:, :-1, :]],
        axis=1,
    )
    targets = target_deltas.reshape(-1, 3)

    denom = float(np.sum(actions * actions))
    scalar = 0.0 if denom <= 1e-12 else float(np.sum(actions * targets) / denom)

    diag_denom = np.sum(actions * actions, axis=0)
    diag_numer = np.sum(actions * targets, axis=0)
    diagonal = np.divide(diag_numer, diag_denom, out=np.zeros_like(diag_numer), where=diag_denom > 1e-12)

    x_aug = np.concatenate([actions, np.ones((actions.shape[0], 1), dtype=actions.dtype)], axis=1)
    coef, *_ = np.linalg.lstsq(x_aug, targets, rcond=None)
    full_weights = coef[:3].T
    full_bias = coef[3]

    return {
        "scalar": scalar,
        "diagonal": diagonal,
        "full_linear_weights": full_weights,
        "full_linear_bias": full_bias,
    }


def baseline_predictions(split, horizon, fitted_params):
    actions = split["actions"].reshape(-1, horizon, 7)
    target = split["targets"]
    hold = np.zeros_like(target)
    cumsum = np.cumsum(actions[:, :, :3] * 0.05, axis=1)
    fitted_scalar = np.cumsum(actions[:, :, :3] * fitted_params["scalar"], axis=1)
    fitted_diagonal = np.cumsum(actions[:, :, :3] * fitted_params["diagonal"], axis=1)
    fitted_full_linear = np.cumsum(
        actions[:, :, :3] @ fitted_params["full_linear_weights"].T + fitted_params["full_linear_bias"],
        axis=1,
    )
    return {
        "hold": hold,
        "cumsum_action_scale_0p05": cumsum,
        "cumsum_fitted_scalar": fitted_scalar,
        "cumsum_fitted_diagonal": fitted_diagonal,
        "cumsum_fitted_full_linear": fitted_full_linear,
    }


def evaluate_model(model, split, stats, device, batch_size=8192):
    model.eval()
    state = normalize(split["states"], stats["state_mean"], stats["state_std"]).astype(np.float32)
    action = normalize(split["actions"], stats["action_mean"], stats["action_std"]).astype(np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, state.shape[0], batch_size):
            state_t = torch.from_numpy(state[start : start + batch_size]).to(device)
            action_t = torch.from_numpy(action[start : start + batch_size]).to(device)
            pred = model(state_t, action_t).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0) * stats["target_std"] + stats["target_mean"]


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_summary(output_dir, config, metrics):
    lines = [
        "# OSC EEF Forward Model",
        "",
        f"Dataset: `{metrics['dataset']}`",
        f"DP config: `{metrics['dp_config']}`",
        f"Resolved horizon: `{metrics['resolved_horizon']}`",
        "",
        "## Validation Metrics",
        "",
        "| predictor | traj RMSE cm | terminal mean cm | terminal median cm | terminal p90 cm |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, vals in metrics["valid"].items():
        lines.append(
            "| {name} | {rmse:.3f} | {term:.3f} | {med:.3f} | {p90:.3f} |".format(
                name=name,
                rmse=vals["traj_rmse_cm"],
                term=vals["terminal_error_mean_cm"],
                med=vals["terminal_error_median_cm"],
                p90=vals["terminal_error_p90_cm"],
            )
        )

    model_rmse = metrics["valid"]["model"]["traj_rmse_cm"]
    model_terminal = metrics["valid"]["model"]["terminal_error_mean_cm"]
    cumsum_rmse = metrics["valid"]["cumsum_action_scale_0p05"]["traj_rmse_cm"]
    cumsum_terminal = metrics["valid"]["cumsum_action_scale_0p05"]["terminal_error_mean_cm"]
    baseline_items = {k: v for k, v in metrics["valid"].items() if k != "model"}
    best_rmse_baseline_name, best_rmse_baseline = min(
        baseline_items.items(), key=lambda item: item[1]["traj_rmse_cm"]
    )
    best_terminal_baseline_name, best_terminal_baseline = min(
        baseline_items.items(), key=lambda item: item[1]["terminal_error_mean_cm"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Model trajectory RMSE improvement over old `action * 0.05` cumsum: `{cumsum_rmse / model_rmse:.2f}x`.",
            f"- Model terminal error improvement over old `action * 0.05` cumsum: `{cumsum_terminal / model_terminal:.2f}x`.",
            f"- Best fitted cumsum baseline by trajectory RMSE: `{best_rmse_baseline_name}` at `{best_rmse_baseline['traj_rmse_cm']:.3f} cm`; model improves this by `{best_rmse_baseline['traj_rmse_cm'] / model_rmse:.2f}x`.",
            f"- Best fitted cumsum baseline by terminal error: `{best_terminal_baseline_name}` at `{best_terminal_baseline['terminal_error_mean_cm']:.3f} cm`; model improves this by `{best_terminal_baseline['terminal_error_mean_cm'] / model_terminal:.2f}x`.",
            "- This model is only a guidance surrogate. It does not replace the OSC controller or the diffusion policy.",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(make_jsonable(config), indent=2),
            "```",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_json(config_path)
    dataset_path = resolve_path(config["dataset"])
    dp_config_path = resolve_path(config["dp_config"])
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["train"].get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    horizon = resolve_horizon(config)
    state_keys = config["state_keys"]
    action_key = config["action_key"]
    target_key = config["target_key"]

    print(f"dataset: {dataset_path}")
    print(f"dp_config: {dp_config_path}")
    print(f"resolved_horizon: {horizon}")
    print("loading splits...")
    train = load_split(dataset_path, "train", horizon, state_keys, action_key, target_key)
    valid = load_split(dataset_path, "valid", horizon, state_keys, action_key, target_key)
    print(f"train samples: {train['states'].shape[0]} demos: {len(train['demo_names'])}")
    print(f"valid samples: {valid['states'].shape[0]} demos: {len(valid['demo_names'])}")

    stats = {
        "state_mean": train["states"].mean(axis=0, keepdims=True),
        "state_std": np.maximum(train["states"].std(axis=0, keepdims=True), 1e-6),
        "action_mean": train["actions"].mean(axis=0, keepdims=True),
        "action_std": np.maximum(train["actions"].std(axis=0, keepdims=True), 1e-6),
        "target_mean": train["targets"].mean(axis=(0, 1), keepdims=True),
        "target_std": np.maximum(train["targets"].std(axis=(0, 1), keepdims=True), 1e-6),
    }

    x_state = normalize(train["states"], stats["state_mean"], stats["state_std"]).astype(np.float32)
    x_action = normalize(train["actions"], stats["action_mean"], stats["action_std"]).astype(np.float32)
    y = normalize(train["targets"], stats["target_mean"], stats["target_std"]).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_state), torch.from_numpy(x_action), torch.from_numpy(y)),
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=int(config["train"].get("num_workers", 0)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = OSCForwardModel(
        state_dim=train["states"].shape[1],
        action_dim=train["actions"].shape[1],
        horizon=horizon,
        state_embed_dim=int(model_cfg["state_embed_dim"]),
        action_embed_dim=int(model_cfg["action_embed_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    terminal_weight = float(config["train"].get("terminal_weight", 2.0))

    best = {"epoch": -1, "valid_loss": float("inf"), "state_dict": None}
    history = []
    stale = 0
    patience = int(config["train"].get("patience", 30))

    valid_state = normalize(valid["states"], stats["state_mean"], stats["state_std"]).astype(np.float32)
    valid_action = normalize(valid["actions"], stats["action_mean"], stats["action_std"]).astype(np.float32)
    valid_target = normalize(valid["targets"], stats["target_mean"], stats["target_std"]).astype(np.float32)
    valid_state_t = torch.from_numpy(valid_state).to(device)
    valid_action_t = torch.from_numpy(valid_action).to(device)
    valid_target_t = torch.from_numpy(valid_target).to(device)

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        train_losses = []
        for state_b, action_b, target_b in train_loader:
            state_b = state_b.to(device)
            action_b = action_b.to(device)
            target_b = target_b.to(device)
            pred = model(state_b, action_b)
            loss = loss_fn(pred, target_b) + terminal_weight * loss_fn(pred[:, -1], target_b[:, -1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            pred_valid = model(valid_state_t, valid_action_t)
            valid_loss = float(
                loss_fn(pred_valid, valid_target_t).item()
                + terminal_weight * loss_fn(pred_valid[:, -1], valid_target_t[:, -1]).item()
            )
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
        if stale >= patience:
            print(f"early stopping at epoch {epoch} (best epoch {best['epoch']})")
            break

    model.load_state_dict(best["state_dict"])
    stats_np = {k: v.astype(np.float32) for k, v in stats.items()}

    print("evaluating...")
    train_pred = evaluate_model(model, train, stats_np, device)
    valid_pred = evaluate_model(model, valid, stats_np, device)

    fitted_baseline_params = fit_cumsum_baselines(train, horizon)

    metrics = {
        "dataset": str(dataset_path),
        "dp_config": str(dp_config_path),
        "config_path": str(config_path),
        "resolved_horizon": horizon,
        "baseline_params": {
            "scalar": fitted_baseline_params["scalar"],
            "diagonal": fitted_baseline_params["diagonal"],
            "full_linear_weights": fitted_baseline_params["full_linear_weights"],
            "full_linear_bias": fitted_baseline_params["full_linear_bias"],
        },
        "splits": {
            "train": {"n_demos": len(train["demo_names"]), "n_samples": int(train["states"].shape[0])},
            "valid": {"n_demos": len(valid["demo_names"]), "n_samples": int(valid["states"].shape[0])},
        },
        "model_config": {
            "state_dim": int(train["states"].shape[1]),
            "action_dim": int(train["actions"].shape[1]),
            "output_dim": int(horizon * 3),
            **model_cfg,
        },
        "training": {
            **config["train"],
            "device": str(device),
            "best_epoch": best["epoch"],
            "best_valid_loss": best["valid_loss"],
        },
        "train": {"model": prediction_metrics(train_pred, train["targets"])},
        "valid": {"model": prediction_metrics(valid_pred, valid["targets"])},
        "history": history,
    }

    for split_name, split_data in (("train", train), ("valid", valid)):
        for baseline_name, pred in baseline_predictions(split_data, horizon, fitted_baseline_params).items():
            metrics[split_name][baseline_name] = prediction_metrics(pred, split_data["targets"])

    resolved_config = {
        **config,
        "resolved_horizon": horizon,
        "resolved_dataset": str(dataset_path),
        "resolved_dp_config": str(dp_config_path),
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "OSCForwardModel",
        "config": resolved_config,
        "stats": stats_np,
        "metrics": metrics,
    }

    torch.save(checkpoint, output_dir / "model.pth")
    (output_dir / "config.json").write_text(json.dumps(make_jsonable(resolved_config), indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(make_jsonable(metrics), indent=2) + "\n")
    write_summary(output_dir, resolved_config, metrics)

    print("\nVALID SUMMARY")
    for name, vals in metrics["valid"].items():
        print(
            f"{name:<24} traj_rmse={vals['traj_rmse_cm']:.3f}cm "
            f"terminal_mean={vals['terminal_error_mean_cm']:.3f}cm "
            f"terminal_p90={vals['terminal_error_p90_cm']:.3f}cm"
        )
    print(f"\nwrote model to {output_dir / 'model.pth'}")
    print(f"wrote metrics to {output_dir / 'metrics.json'}")
    print(f"wrote summary to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
