"""
Open-loop replay validation for delta_eef -> OSC adapters.

This is phase 2 of the adapter diagnosis:
    delta_eef_action -> adapter -> OSC command -> env.step(...)

It compares the adapter against:
    - Plan A: original OSC actions (replay upper bound)
    - Plan B-1: raw delta_eef_action sent to OSC (known failure)

Run from repo root:
    MUJOCO_GL=egl uv run python docs/route_b_validation/replay_delta_eef_to_osc_adapter.py
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET = REPO_ROOT / "third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5"
DEFAULT_ADAPTER_PARAMS = REPO_ROOT / "outputs/route_b_validation/delta_eef_to_osc_adapter/adapter_params.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/route_b_validation/delta_eef_to_osc_adapter"
DEFAULT_MLP_ADAPTER = DEFAULT_OUTPUT_DIR / "mlp_adapter.pth"

sys.path.insert(0, str(SCRIPT_DIR))
from fit_delta_eef_to_osc_adapter import apply_adapter  # noqa: E402


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


def load_mlp_adapter(path):
    ckpt = torch.load(path, map_location="cpu")
    model = DeltaEEFToOSCMLP(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        output_dim=int(ckpt["output_dim"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "input_mean": np.asarray(ckpt["input_mean"], dtype=np.float32),
        "input_std": np.asarray(ckpt["input_std"], dtype=np.float32),
        "target_mean": np.asarray(ckpt["target_mean"], dtype=np.float32),
        "target_std": np.asarray(ckpt["target_std"], dtype=np.float32),
    }


def apply_mlp_adapter(adapter, obs, delta_eef_action):
    feature = np.concatenate(
        [
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_joint_pos"],
            obs["robot0_joint_vel"],
            delta_eef_action[:3],
        ],
        axis=0,
    ).astype(np.float32)[None]
    x = (feature - adapter["input_mean"]) / adapter["input_std"]
    with torch.no_grad():
        pred_norm = adapter["model"](torch.from_numpy(x)).numpy()
    pred_xyz = pred_norm * adapter["target_std"] + adapter["target_mean"]
    out = delta_eef_action.copy()
    out[:3] = np.clip(pred_xyz[0], -1.0, 1.0)
    return out


def decode_demo_names(raw_names):
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names]


def get_demo_names(h5_file, split, n_demos):
    if "mask" in h5_file and split in h5_file["mask"]:
        demo_names = decode_demo_names(h5_file["mask"][split][:])
    else:
        demo_names = sorted(h5_file["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    return demo_names[:n_demos]


def make_env(env_meta):
    return EnvUtils.create_env_from_metadata(
        env_meta=dict(
            type=env_meta["type"],
            env_name=env_meta["env_name"],
            env_version=env_meta.get("env_version"),
            env_kwargs=json.loads(json.dumps(env_meta["env_kwargs"])),
        ),
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )


def reset_controller_refs(env):
    """Match compare_all.py: reset_to changes sim state after controller refs are initialized."""
    for ctrl in env.env.robots[0].part_controllers.values():
        if hasattr(ctrl, "update"):
            ctrl.update(force=True)
        if hasattr(ctrl, "reset_goal"):
            ctrl.reset_goal()
        if hasattr(ctrl, "user_sensitivity"):
            ctrl.user_sensitivity = 1.0


def replay(env, grp, action_fn, label):
    states = grp["states"][:]
    obs = env.reset_to(
        {
            "states": states[0],
            "model": grp.attrs["model_file"],
            "ep_meta": grp.attrs.get("ep_meta", None),
        }
    )
    reset_controller_refs(env)

    n_steps = grp["actions"].shape[0]
    desired_dpos_list = []
    actual_dpos_list = []
    err_to_target_list = []
    err_to_orig_list = []
    replay_traj = [obs["robot0_eef_pos"].copy()]

    for t in range(n_steps):
        action = action_fn(t, grp, obs)
        before = obs["robot0_eef_pos"].copy()
        err_to_orig_list.append(np.linalg.norm(before - grp["obs/robot0_eef_pos"][t]))

        obs, _, _, _ = env.step(action)
        after = obs["robot0_eef_pos"].copy()
        replay_traj.append(after)

        desired_dpos_list.append(action[:3] * 0.05)
        actual_dpos_list.append(after - before)
        err_to_target_list.append(np.linalg.norm(after - grp["next_obs/robot0_eef_pos"][t]))

    replay_traj = np.asarray(replay_traj)
    desired_mags = np.asarray([np.linalg.norm(x) for x in desired_dpos_list])
    actual_mags = np.asarray([np.linalg.norm(x) for x in actual_dpos_list])
    nontrivial = desired_mags > 0.005
    if np.any(nontrivial):
        ratios = actual_mags[nontrivial] / desired_mags[nontrivial]
        tracking_median = float(np.median(ratios))
        tracking_p25 = float(np.percentile(ratios, 25))
        tracking_p75 = float(np.percentile(ratios, 75))
    else:
        tracking_median = tracking_p25 = tracking_p75 = float("nan")

    return {
        "label": label,
        "n_steps": int(n_steps),
        "desired_dpos_mag_mean_cm": float(np.mean(desired_mags) * 100),
        "actual_dpos_mag_mean_cm": float(np.mean(actual_mags) * 100),
        "tracking_median": tracking_median,
        "tracking_p25": tracking_p25,
        "tracking_p75": tracking_p75,
        "end_err_to_target_cm": float(np.linalg.norm(replay_traj[-1] - grp["next_obs/robot0_eef_pos"][-1]) * 100),
        "err_to_target_mean_cm": float(np.mean(err_to_target_list) * 100),
        "err_to_target_max_cm": float(np.max(err_to_target_list) * 100),
        "end_err_to_orig_cm": float(np.linalg.norm(replay_traj[-1] - grp["obs/robot0_eef_pos"][-1]) * 100),
        "err_to_orig_per_step_cm": [float(x * 100) for x in err_to_orig_list],
        "err_to_target_per_step_cm": [float(x * 100) for x in err_to_target_list],
        "replay_traj": replay_traj.tolist(),
        "data_traj": grp["obs/robot0_eef_pos"][:].tolist(),
    }


def summarize(results_by_plan):
    summary = {}
    for plan, results in results_by_plan.items():
        summary[plan] = {
            "n_demos": len(results),
            "desired_dpos_mag_mean_cm": float(np.mean([r["desired_dpos_mag_mean_cm"] for r in results])),
            "actual_dpos_mag_mean_cm": float(np.mean([r["actual_dpos_mag_mean_cm"] for r in results])),
            "tracking_median": float(np.nanmean([r["tracking_median"] for r in results])),
            "err_to_target_mean_cm": float(np.mean([r["err_to_target_mean_cm"] for r in results])),
            "end_err_to_orig_cm": float(np.mean([r["end_err_to_orig_cm"] for r in results])),
            "end_err_to_target_cm": float(np.mean([r["end_err_to_target_cm"] for r in results])),
        }
    return summary


def write_summary_md(output_dir, dataset, split, demo_names, summary, adapter_names):
    lines = [
        "# Delta EEF to OSC Adapter: Phase 2 Replay",
        "",
        f"Dataset: `{dataset}`",
        f"Split: `{split}`",
        f"Demos: `{', '.join(demo_names)}`",
        "",
        "## Replay Summary",
        "",
        "| plan | desired cm | actual cm | tracking | mean target err cm | end orig err cm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for plan, vals in summary.items():
        lines.append(
            "| {plan} | {desired:.3f} | {actual:.3f} | {track:.3f} | {target:.3f} | {end:.3f} |".format(
                plan=plan,
                desired=vals["desired_dpos_mag_mean_cm"],
                actual=vals["actual_dpos_mag_mean_cm"],
                track=vals["tracking_median"],
                target=vals["err_to_target_mean_cm"],
                end=vals["end_err_to_orig_cm"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- `plan_A_original_osc` is the replay upper bound: original dataset action through original OSC controller.",
            "- `plan_B1_raw_delta_eef` is the known failure: actual EEF delta sent directly as an OSC command.",
            "- Adapter plans transform `delta_eef_action[:3]` into OSC `actions[:3]`; rotation and gripper are copied from `delta_eef_action`.",
            "",
            "Adapter plans tested:",
            "",
        ]
    )
    for name in adapter_names:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "A usable adapter should be much closer to Plan A than to Plan B-1 in open-loop replay.",
        ]
    )
    (output_dir / "replay_summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--adapter-params", type=Path, default=DEFAULT_ADAPTER_PARAMS)
    parser.add_argument("--mlp-adapter", type=Path, default=DEFAULT_MLP_ADAPTER)
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--n-demos", type=int, default=5)
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    adapter_params_path = args.adapter_params.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))

    with adapter_params_path.open() as f:
        adapter_bundle = json.load(f)
    adapters = adapter_bundle["all_params"]
    mlp_adapter = None
    if not args.no_mlp and args.mlp_adapter.exists():
        mlp_adapter = load_mlp_adapter(args.mlp_adapter.resolve())
        print(f"Loaded MLP adapter from {args.mlp_adapter.resolve()}")

    results_by_plan = {
        "plan_A_original_osc": [],
        "plan_B1_raw_delta_eef": [],
    }
    for adapter_name in adapters:
        results_by_plan[f"adapter_{adapter_name}"] = []
    if mlp_adapter is not None:
        results_by_plan["adapter_mlp_state_conditioned"] = []

    with h5py.File(dataset_path, "r") as f:
        demo_names = get_demo_names(f, args.split, args.n_demos)
        for plan_name in results_by_plan:
            print(f"\n=== {plan_name} ===")
            env = make_env(env_meta)
            try:
                for demo_name in demo_names:
                    grp = f[f"data/{demo_name}"]
                    if plan_name == "plan_A_original_osc":
                        action_fn = lambda t, g, o: g["actions"][t]
                    elif plan_name == "plan_B1_raw_delta_eef":
                        action_fn = lambda t, g, o: g["delta_eef_action"][t]
                    elif plan_name == "adapter_mlp_state_conditioned":
                        action_fn = lambda t, g, o: apply_mlp_adapter(mlp_adapter, o, g["delta_eef_action"][t])
                    else:
                        adapter_name = plan_name.removeprefix("adapter_")
                        params = adapters[adapter_name]

                        def action_fn(t, g, o, _params=params):
                            return apply_adapter(_params, g["delta_eef_action"][t : t + 1])[0]

                    result = replay(env, grp, action_fn, f"{plan_name}:{demo_name}")
                    results_by_plan[plan_name].append(result)
                    print(
                        f"  {demo_name}: desired={result['desired_dpos_mag_mean_cm']:.3f}cm "
                        f"actual={result['actual_dpos_mag_mean_cm']:.3f}cm "
                        f"track={result['tracking_median']:.3f} "
                        f"target_err={result['err_to_target_mean_cm']:.3f}cm "
                        f"end_orig={result['end_err_to_orig_cm']:.3f}cm"
                    )
            finally:
                env.env.close()

    summary = summarize(results_by_plan)
    out = {
        "dataset": str(dataset_path),
        "adapter_params": str(adapter_params_path),
        "split": args.split,
        "demo_names": demo_names,
        "summary": summary,
        "results": results_by_plan,
    }
    (output_dir / "replay_results.json").write_text(json.dumps(out, indent=2) + "\n")
    adapter_plan_names = list(adapters.keys())
    if mlp_adapter is not None:
        adapter_plan_names.append("mlp_state_conditioned")
    write_summary_md(output_dir, dataset_path, args.split, demo_names, summary, adapter_plan_names)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for plan, vals in summary.items():
        print(
            f"{plan:<28} desired={vals['desired_dpos_mag_mean_cm']:.3f}cm "
            f"actual={vals['actual_dpos_mag_mean_cm']:.3f}cm "
            f"track={vals['tracking_median']:.3f} "
            f"target_err={vals['err_to_target_mean_cm']:.3f}cm "
            f"end_orig={vals['end_err_to_orig_cm']:.3f}cm"
        )
    print(f"\nWrote replay results to {output_dir / 'replay_results.json'}")
    print(f"Wrote replay summary to {output_dir / 'replay_summary.md'}")


if __name__ == "__main__":
    main()
