# Project Scripts

> Local scripts under `scripts/`. All invoked with `uv run python <path>`.
> See root `AGENTS.md` §3 for a high-level index and §10 for git workflow.

## Layout

```
scripts/
├── AGENTS.md                       ← **You are reading this file**. Index of all 7 local scripts
├── benchmark_pointcloud.py         ← Pointcloud compute overhead (recompute vs static cache)
├── diagnose_collisions.py          ← EEF↔obstacle distance tracking per step
├── diagnose_guidance_gradient.py   ← Per-denoising-step cost / grad_norm / oracle cost logging
└── 2026-06-22_600_rollout_eval/    ← 4 scripts reproducing the 2026-06-22 600-rollout experiment
    ├── run_baseline_eval_matrix.py             ← No-guidance eval, masked-image policy
    ├── run_baseline_eval_matrix_no_mask.py     ← No-guidance eval, un-masked policy
    ├── run_pc1_eval_matrix.py                  ← PC-1 obstacle-guided eval, masked-image policy
    └── run_pc1_eval_matrix_no_mask.py          ← PC-1 obstacle-guided eval, un-masked policy
```

## Diagnostic / benchmark scripts

- `benchmark_pointcloud.py` — measures per-call time of `build_pointcloud_context_fields()` and simulates a 400-step rollout to compare original (recompute every step) vs cached (compute once) strategies. Used to justify the static-caching optimization before the 600-rollout experiment.
- `diagnose_collisions.py` — runs a small set of rollouts and logs per-step EEF↔obstacle distance, distinguishing Δgeo (arm-trajectory-blocked) failures from non-target collisions. Usage:
  ```bash
  uv run python scripts/diagnose_collisions.py \
      --agent outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/best.pth \
      --env PickPlaceBreadCerealMilkCan --n_rollouts 12 --horizon 400 --seed 600
  ```
- `diagnose_guidance_gradient.py` — monkey-patches the algorithm to record per-denoising-step cost, grad_norm, and oracle cost (using exact object geometry). Answers: is cost dominated by denoising noise? Is pointcloud cost much smaller than oracle cost (sparse cloud)?

## 2026-06-22 600-rollout eval scripts

These 4 scripts reproduce the 600-rollout experiment logged in `docs/RESEARCH_LOG.md` (2026-06-22). Each launches a 4-env × 3-seed grid of `run_trained_agent.py` / `run_obstacle_guided_agent.py` jobs and aggregates per-environment summaries.

| Script | Policy | Guidance | Checkpoint |
|---|---|---|---|
| `run_baseline_eval_matrix.py` | masked-image | none | `outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/best.pth` |
| `run_baseline_eval_matrix_no_mask.py` | un-masked | none | `outputs/robomimic/checkpoints/diffusion_policy_can_yq_image/best.pth` |
| `run_pc1_eval_matrix.py` | masked-image | PC-1 | (same masked checkpoint) |
| `run_pc1_eval_matrix_no_mask.py` | un-masked | PC-1 | (same un-masked checkpoint) |

Output root for each: `outputs/robomimic/eval/{baseline,obstacle_guided}/<policy>/`.

### Usage

```bash
uv run python scripts/2026-06-22_600_rollout_eval/run_baseline_eval_matrix.py
```

`ROOT = Path(__file__).resolve().parents[2]` resolves to the project root. **This `parents[2]` (not `parents[1]`) is intentional** — the scripts sit one level deeper than a flat `scripts/` layout.

### Convention for future experiments

Per-experiment subdirectories named `<YYYY-MM-DD>_<short-slug>/` (e.g. `2026-06-22_600_rollout_eval/`) signal that the scripts reproduce a specific historical experiment. Use the experiment date as the prefix.

## Removed scripts (Route B, archived)

These 4 scripts were used for the Route B experiment (switching prediction target to EEF trajectory) which was validated and rejected on 2026-06-26. They were removed on 2026-06-27; results are archived in `docs/route_b_validation/report.md`.

- `add_delta_eef_label.py` — annotated HDF5 demos with `delta_eef_action` (achieved EEF delta)
- `validate_delta_eef_dataset.py` — sanity-checked the labels (shape, NaN/Inf, range)
- `replay_delta_eef_to_video.py` — open-loop replay with MP4
- `calibrate_action_scale.py` — measured 3-4 cm RMSE from `cumsum(action * 0.05)` (physical OSC tracking limit; finding preserved in the report)
