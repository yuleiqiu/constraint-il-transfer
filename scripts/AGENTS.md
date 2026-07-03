# Project Scripts

> Local scripts under `scripts/`. All invoked with `uv run python <path>`.
> See root `AGENTS.md` §3 for a high-level index and §10 for git workflow.

## Layout

```
scripts/
├── AGENTS.md
├── benchmark_pointcloud.py
├── check_osc_forward_model_grad.py
├── diagnose_collisions.py
├── diagnose_control_timing.py
├── diagnose_forward_model_random_rollout.py
├── diagnose_guidance_gradient.py
├── diagnose_guidance_update_effect.py
├── osc_forward_model.py
├── train_osc_eef_forward_model.py
└── 2026-06-22_600_rollout_eval/
    ├── run_baseline_eval_matrix.py
    ├── run_baseline_eval_matrix_no_mask.py
    ├── run_pc1_eval_matrix.py
    └── run_pc1_eval_matrix_no_mask.py
```

## Categories

- **Diagnostic / benchmark** (in `scripts/`) — standalone scripts for per-step EEF↔obstacle tracking, denoising-step cost logging, pointcloud overhead measurement, OSC control timing, guidance update counterfactuals, and forward-model random-rollout validation. See docstrings for usage.
- **Forward model** (in `scripts/`) — `train_osc_eef_forward_model.py`, `osc_forward_model.py`, and `check_osc_forward_model_grad.py` train and validate the OSC action-chunk to EEF-trajectory surrogate used by guidance diagnostics.
- **Eval matrices** (in `scripts/2026-06-22_600_rollout_eval/`) — 4 scripts that reproduce the 600-rollout experiment from 2026-06-22. Each launches a 4-env × 3-seed grid of inner robomimic scripts and aggregates per-environment summaries. See `docs/RESEARCH_LOG.md` (2026-06-22 entry) for the experiment background.

## Convention for future experiments

Reproduction scripts go in a per-experiment subdirectory named `<YYYY-MM-DD>_<short-slug>/` (e.g. `2026-06-22_600_rollout_eval/`). Use the experiment date as the prefix.
