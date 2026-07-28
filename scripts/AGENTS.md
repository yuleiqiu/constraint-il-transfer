# Project Scripts

> Local scripts under `scripts/`. All invoked with `uv run python <path>`.
> See root `AGENTS.md` §3 for a high-level index and §10 for git workflow.

## Layout

```
scripts/
├── AGENTS.md
├── diagnose_collisions.py
├── diagnose_control_timing.py
├── 2026-07-20_intent_diagnostic/
│   ├── analyze_intent.py
│   └── test_analyze_intent.py
├── 2026-07-22_task_constrained_guidance/
│   ├── capture_preentry_states.py
│   ├── feasibility.py
│   ├── fixed_state_feasibility.py
│   ├── test_feasibility.py
│   └── README.md
├── 2026-07-23_lan_o3dp_reproduction/
│   ├── create_target_pointcloud_dataset.py
│   ├── verify_phase1_dataset.py
│   ├── overfit_one_batch.py
│   └── README.md
├── 2026-07-27_policy_manifold_diagnostic/
│   ├── manifold_feasibility.py
│   ├── run_fixed_state.py
│   ├── test_manifold_feasibility.py
│   └── README.md
├── 2026-07-27_semi_manual_waypoint_teacher/
│   ├── evaluate_waypoint_teacher.py
│   ├── test_waypoint_teacher.py
│   └── README.md
├── eef_pose_osc_policy/
│   ├── create_abs_eef_osc_dataset.py
│   ├── create_delta_eef_pose_osc_dataset.py
│   ├── analyze_delta_eef_eval.py
│   ├── diagnose_delta_eef_policy_traj.py
│   ├── eval_delta_eef_multienv.py
│   ├── plot_delta_eef_eval_cases.py
│   ├── verify_abs_eef_osc_dataset.py
│   ├── verify_delta_eef_pose_osc_dataset.py
│   ├── smoke_abs_eef_osc_wrapper.py
│   └── smoke_delta_eef_pose_osc_wrapper.py
├── guided_denoising/
│   ├── common.py
│   ├── same_state_diagnostic.py
│   ├── paired_rollout_eval.py
│   ├── aggregate_results.py
│   ├── render_paired_rollout.py
│   ├── visualize_guidance_vectors.py
│   └── visualize_final_trajectory_comparison.py
└── 2026-06-22_600_rollout_eval/
    ├── run_baseline_eval_matrix.py
    └── run_baseline_eval_matrix_no_mask.py
```

## Categories

- **Diagnostic / benchmark** (in `scripts/`) — standalone scripts for per-step EEF↔obstacle tracking and OSC control timing. See docstrings for usage.
- **Clean-policy intent diagnostic** (in `scripts/2026-07-20_intent_diagnostic/`) — CPU-only offline analysis of the 600 existing unguided delta-EEF rollouts. It measures closest target progress and segment-to-footprint clearance for every pre-target chunk, selects one fixed critical chunk per episode, applies the three-way intent / obstruction partition and A/B/C Wilson decision rule, and produces deterministic CSV / JSON summaries and representative plots. Run the companion `test_analyze_intent.py` directly for the pure-Python geometry and selection tests. Canonical result: `outputs/guided_denoising/intent_diagnostic/report.md`.
- **Task-constrained guidance** (in `scripts/2026-07-22_task_constrained_guidance/`) — completed Stage 1 diagnostic that preserves the frozen `2 / 16 / 8` horizon contract. It captures pre-entry target-directed-obstructed states and applies a CPU-testable eight-step XY feasibility correction with swept segment clearance, target-progress, trust-region, and action-bound constraints; it preserves Z, orientation, and gripper actions. The 20-layout preregistered gate failed, so posterior-gradient Stage 2 was not implemented. See `outputs/task_constrained_guidance/stage1_report.md`.
- **LAN-O3DP reproduction** (in `scripts/2026-07-23_lan_o3dp_reproduction/`) — completed point-cloud dataset generation, HDF5/runtime parity verification, fixed-batch overfit smoke testing, and 40-demo subset preflight. The resulting point-cloud policies missed the clean-source gate; the epoch-350 guidance closure also failed its paired task gate. See `docs/guided_denoising/lan_o3dp_epoch350_closure.md`.
- **Policy-manifold feasibility diagnostic** (in `scripts/2026-07-27_policy_manifold_diagnostic/`) — completed phase-free fixed-state upper-bound test over the existing two-distractor pre-collision state bank. It samples the frozen delta-EEF policy repeatedly, fits a shrinkage covariance over executable EEF waypoint trajectories, and tests whether a target-progress-preserving path can satisfy expanded obstacle boundaries while remaining inside an empirically calibrated Mahalanobis support ellipsoid. Support-constrained projection is feasible in 1/20 states versus 14/20 when the support bound is removed; projected actions are not executed.
- **Semi-manual waypoint teacher** (in `scripts/2026-07-27_semi_manual_waypoint_teacher/`) — oracle Cartesian grasp-and-lift teacher with discrete yaw search over the 20 held-out pre-collision states. It restores every candidate from the same simulator state, selects the smallest safe yaw, replays only that route, and hands back to the frozen policy. The teacher upper bound reaches 18/20 Task success with 2/20 distractor collisions; it is a correction-data feasibility result, not a deployment method.
- **EEF-pose OSC policy** (in `scripts/eef_pose_osc_policy/`) — dataset converters, expert replay validators, robomimic wrapper smoke tests, delta-policy pose-trajectory diagnostics, multi-env eval tooling, and eval analysis for full-pose EEF actions executed by `OSC_POSE`. Training runbook: `docs/eef_pose_osc_policy_training.md`; conclusion: `outputs/eef_pose_osc_policy/README.md`.
- **Guided denoising** (in `scripts/guided_denoising/`) — captures fixed clean-image delta-EEF states, sweeps the agreed guidance scales under identical observations and diffusion seeds, visualizes either one DDIM step's pushed target or the paired final `scale=0` / guided trajectories after all DDIM steps on full and locally zoomed `agentview` images, runs matched baseline / guided rollout pilots, and reports Task SR together with Safe SR / CR / NCR and the four-way outcome partition. Raw HDF5 / JSONL artifacts under `outputs/guided_denoising/` are ignored; summaries and reports remain trackable.
- `render_paired_rollout.py` replays selected recorded pairs, verifies the exact
  simulator-state hash and task/collision outcome, and writes synchronized
  baseline-versus-guided videos.
- **Eval matrices** (in `scripts/2026-06-22_600_rollout_eval/`) — baseline launchers from the 2026-06-22 evaluation branch. Guided / PC1 launchers were removed with the archived guidance implementation. See `docs/RESEARCH_LOG.md` (2026-06-22 entry) for the experiment background.
- **Archived forward-model / ranking results** — implementation scripts have been removed. Keep result documents only: `docs/forward_model_guidance_next_steps.md`, `docs/action_chunk_ranking_report.md`, and `outputs/forward_model/osc_eef_forward_image_v15/summary.md`.

## Convention for future experiments

Reproduction scripts go in a per-experiment subdirectory named `<YYYY-MM-DD>_<short-slug>/` (e.g. `2026-06-22_600_rollout_eval/`). Use the experiment date as the prefix.
