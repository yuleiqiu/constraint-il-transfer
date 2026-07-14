# Project Scripts

> Local scripts under `scripts/`. All invoked with `uv run python <path>`.
> See root `AGENTS.md` §3 for a high-level index and §10 for git workflow.

## Layout

```
scripts/
├── AGENTS.md
├── diagnose_collisions.py
├── diagnose_control_timing.py
├── eef_pose_osc_policy/
    ├── create_abs_eef_osc_dataset.py
    ├── create_delta_eef_pose_osc_dataset.py
    ├── analyze_delta_eef_eval.py
    ├── diagnose_delta_eef_policy_traj.py
    ├── eval_delta_eef_multienv.py
    ├── plot_delta_eef_eval_cases.py
    ├── verify_abs_eef_osc_dataset.py
    ├── verify_delta_eef_pose_osc_dataset.py
    ├── smoke_abs_eef_osc_wrapper.py
    └── smoke_delta_eef_pose_osc_wrapper.py
└── 2026-06-22_600_rollout_eval/
    ├── run_baseline_eval_matrix.py
    └── run_baseline_eval_matrix_no_mask.py
```

## Categories

- **Diagnostic / benchmark** (in `scripts/`) — standalone scripts for per-step EEF↔obstacle tracking and OSC control timing. See docstrings for usage.
- **EEF-pose OSC policy** (in `scripts/eef_pose_osc_policy/`) — dataset converters, expert replay validators, robomimic wrapper smoke tests, delta-policy pose-trajectory diagnostics, multi-env eval tooling, and eval analysis for full-pose EEF actions executed by `OSC_POSE`. Training runbook: `docs/eef_pose_osc_policy_training.md`; conclusion: `outputs/eef_pose_osc_policy/README.md`.
- **Eval matrices** (in `scripts/2026-06-22_600_rollout_eval/`) — baseline launchers from the 2026-06-22 evaluation branch. Guided / PC1 launchers were removed with the archived guidance implementation. See `docs/RESEARCH_LOG.md` (2026-06-22 entry) for the experiment background.
- **Archived forward-model / ranking results** — implementation scripts have been removed. Keep result documents only: `docs/forward_model_guidance_next_steps.md`, `docs/action_chunk_ranking_report.md`, and `outputs/forward_model/osc_eef_forward_image_v15/summary.md`.

## Convention for future experiments

Reproduction scripts go in a per-experiment subdirectory named `<YYYY-MM-DD>_<short-slug>/` (e.g. `2026-06-22_600_rollout_eval/`). Use the experiment date as the prefix.
