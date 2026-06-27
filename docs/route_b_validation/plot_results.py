"""Plot per-step trajectory error curves for the 4 plans."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

PLAN_LABELS = {
    "plan_A_delta_action_osc": "Plan A: OSC delta action (baseline)",
    "plan_B1_delta_eef_osc": "Plan B-1: delta_eef_action as OSC delta",
    "plan_B2_absolute_osc": "Plan B-2: next_eef_pos as OSC absolute target",
    "plan_C_cumulative_ik": "Plan C: cumulative next_eef_pos delta in IK",
}

PLAN_COLORS = {
    "plan_A_delta_action_osc": "#2ca02c",
    "plan_B1_delta_eef_osc": "#ff7f0e",
    "plan_B2_absolute_osc": "#d62728",
    "plan_C_cumulative_ik": "#9467bd",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(RESULTS_PATH) as f:
    data = json.load(f)

# === Figure 1: err_to_orig per step, 4 plans as 4 subplots (one line per demo) ===
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
plan_order = ["plan_A_delta_action_osc", "plan_B1_delta_eef_osc",
              "plan_B2_absolute_osc", "plan_C_cumulative_ik"]
for ax, plan_key in zip(axes.flat, plan_order):
    results = data[plan_key]
    for r in results:
        errs = np.array(r["err_to_orig_per_step_cm"])
        steps = np.arange(len(errs))
        ax.plot(steps, errs, alpha=0.7, linewidth=0.8,
                label=r["label"].split(":")[-1])
    ax.set_title(PLAN_LABELS[plan_key], fontsize=10, loc="left")
    ax.set_xlabel("step t")
    ax.set_ylabel("|replay_EEF(t) - data_EEF(t)| [cm]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(bottom=0)

fig.suptitle("Per-step EEF tracking error (5 demos per plan)\n"
             "Y = |replay start-of-step pos - data start-of-step pos|, lower is better",
             fontsize=12)
fig.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "err_per_step_4plans.png")
fig.savefig(out1, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")

# === Figure 2: same data, all 4 plans overlaid with shaded range (min-max across demos) ===
fig, ax = plt.subplots(1, 1, figsize=(11, 6))
for plan_key in plan_order:
    results = data[plan_key]
    # Align all demos to the same length by truncation
    min_len = min(len(r["err_to_orig_per_step_cm"]) for r in results)
    err_matrix = np.array([r["err_to_orig_per_step_cm"][:min_len] for r in results])
    steps = np.arange(min_len)
    mean_err = err_matrix.mean(axis=0)
    min_err = err_matrix.min(axis=0)
    max_err = err_matrix.max(axis=0)
    color = PLAN_COLORS[plan_key]
    ax.plot(steps, mean_err, color=color, linewidth=1.5,
            label=PLAN_LABELS[plan_key])
    ax.fill_between(steps, min_err, max_err, color=color, alpha=0.15)
ax.set_xlabel("step t (control steps @ 20 Hz)")
ax.set_ylabel("|replay_EEF(t) - data_EEF(t)| [cm]")
ax.set_title("Per-step EEF tracking error: mean (line) and min-max range (shaded) across 5 demos")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(0, 100)
ax.set_xlim(0, min_len)
fig.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "err_per_step_overlay.png")
fig.savefig(out2, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")

# === Figure 3: log-scale view of the same (Plan A would be tiny) ===
fig, ax = plt.subplots(1, 1, figsize=(11, 6))
for plan_key in plan_order:
    results = data[plan_key]
    min_len = min(len(r["err_to_orig_per_step_cm"]) for r in results)
    err_matrix = np.array([r["err_to_orig_per_step_cm"][:min_len] for r in results])
    mean_err = np.maximum(err_matrix.mean(axis=0), 0.01)  # avoid log(0)
    steps = np.arange(min_len)
    color = PLAN_COLORS[plan_key]
    ax.semilogy(steps, mean_err, color=color, linewidth=1.5,
                label=PLAN_LABELS[plan_key])
ax.set_xlabel("step t (control steps @ 20 Hz)")
ax.set_ylabel("|replay_EEF(t) - data_EEF(t)| [cm] (log scale)")
ax.set_title("Same data on log scale — Plan A stays in mm regime, others diverge")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(0, min_len)
fig.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "err_per_step_logscale.png")
fig.savefig(out3, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3}")

# === Figure 4: Trajectory visualization in 3D (or 2D x-z) for demo_1 of each plan ===
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(14, 8))
for i, plan_key in enumerate(plan_order, 1):
    ax = fig.add_subplot(2, 2, i, projection='3d')
    r0 = data[plan_key][0]  # first demo
    replay = np.array(r0["replay_traj"])  # (T+1, 3)
    data_traj = np.array(r0["data_traj"])  # (T, 3)
    color = PLAN_COLORS[plan_key]
    ax.plot(data_traj[:, 0], data_traj[:, 1], data_traj[:, 2],
            'k-', linewidth=1.5, label='data', alpha=0.7)
    ax.plot(replay[:, 0], replay[:, 1], replay[:, 2],
            color=color, linewidth=1.5, label='replay', alpha=0.7)
    ax.scatter(data_traj[0, 0], data_traj[0, 1], data_traj[0, 2],
               c='k', s=50, marker='o', label='start')
    ax.scatter(data_traj[-1, 0], data_traj[-1, 1], data_traj[-1, 2],
               c='k', s=80, marker='X', label='data end')
    ax.scatter(replay[-1, 0], replay[-1, 1], replay[-1, 2],
               c=color, s=80, marker='X', label='replay end')
    ax.set_title(PLAN_LABELS[plan_key], fontsize=10, loc="left")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(fontsize=7)
fig.suptitle("3D EEF trajectory: data (black) vs replay (colored) for demo_1 of each plan",
             fontsize=12)
fig.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "trajectory_3d.png")
fig.savefig(out4, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out4}")

print(f"\nAll plots saved to {OUTPUT_DIR}/")
