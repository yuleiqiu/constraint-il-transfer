# Project Scripts

Completed RGB-D perception pilot (2026-09-16):
`2026-09-15_tabletop_rgbd_perception/README.md`. Replaces runtime geom masks
with fixed GroundingDINO + SAM initialization and Cutie tracking while keeping
the frozen epoch-50 absolute-EEF policy. The 10-layout clean and RaisedMilk
pilot fails decisively (oracle/perceived Task 10/10 versus 0/10 in both), so no
full50 expansion, training, or planner integration was launched. Oracle masks
remain evaluation-only. Start with
`outputs/tabletop_rgbd_perception_2026-09-15/conclusion.md`.

Local planner implementation (authorized2026-09-11):
`2026-09-11_tabletop_local_planner/README.md`. Reuses MVEE fitting with
root-local conservative segment checks, visibility-graph bypass and persistent
transport control. Oracle geometry; inclusive task scoring and actual MP4s.
This is separate from the stopped MVEE denoising-guidance probe.

Completed output-preservation ablation (2026-09-11 review):
`2026-09-10_tabletop_xy_preserve/README.md`. Exact non-XY source reference,
unchanged point cost/height gate, layout28 pilot then fixed10-layout development
comparison; released0.5s placement stopping and per-rollout MP4s.

Placement rescore (2026-09-10): `2026-09-10_tabletop_placement_rescore/README.md`.
Offline contact reconstruction and0.5s released-residence scoring; censored
early terminations stay unknown under residence-only scoring.
`inclusive_rescore.py` applies the later user-confirmed OR criterion and resolves
all50 pairs without simulation. No policy rerun is implied.

Completed corrected full-episode evaluation (authorized 2026-09-09, reviewed 2026-09-10):
`2026-09-09_tabletop_guidance_reeval/README.md`. Fixed 50-layout source / original
target-lifted guidance comparison, two GPUs partitioned by layout, in-run videos
for all 100 formal branches, separate repeated preflight and scorer audit data.

Completed bounded diagnostic (user-authorized 2026-09-07, reviewed 2026-09-08):
`2026-09-07_tabletop_handoff/README.md`. Compares source, original point guidance,
short / long oracle-route help and complete teacher without training. Includes
a reset-metadata scoring audit; do not reuse old Tabletop success labels without
reading `outputs/tabletop_handoff_diagnostic/conclusion.md` from project root.

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
├── 2026-07-21_sample_refine/
│   ├── fixed_state_sweep.py
│   ├── paired_rollout_eval.py
│   ├── aggregate_results.py
│   ├── common.py
│   └── README.md
├── 2026-07-22_task_constrained_guidance/
│   ├── capture_preentry_states.py
│   ├── feasibility.py
│   ├── fixed_state_feasibility.py
│   ├── test_feasibility.py
│   └── README.md
├── 2026-07-23_lan_o3dp_reproduction/
│   ├── create_target_pointcloud_dataset.py
│   ├── create_lan40_subset.py
│   ├── verify_phase1_dataset.py
│   ├── overfit_one_batch.py
│   ├── preflight_lan40.py
│   ├── test_create_lan40_subset.py
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
├── 2026-07-28_targeted_correction/
│   ├── paired_correction.py
│   ├── collect_corrections.py
│   ├── extract_correction_pairs.py
│   ├── verify_correction_data.py
│   ├── smoke_mixed_training.py
│   ├── screen_checkpoints.py
│   ├── test_paired_correction.py
│   └── README.md
├── 2026-07-29_gated_support_selector/
│   ├── selector.py
│   ├── runtime.py
│   ├── fixed_state_preflight.py
│   ├── paired_rollout_eval.py
│   ├── test_selector.py
│   └── README.md
├── 2026-07-29_oracle_branch_value/
│   ├── branch_value.py
│   ├── simulator_branch_eval.py
│   ├── analyze_results.py
│   ├── test_branch_value.py
│   └── README.md
├── 2026-07-29_persistent_correction_oracle/
│   ├── persistent_oracle.py
│   ├── merge_partitions.py
│   ├── test_persistent_oracle.py
│   └── README.md
├── 2026-08-07_empty2d_guidance/
│   ├── generate_data.py
│   ├── verify_data.py
│   ├── preflight.py
│   ├── train.py
│   ├── screen_checkpoints.py
│   ├── paired_evaluation.py
│   ├── merge_geometry_evaluations.py
│   ├── render_videos.py
│   ├── render_environment_overview.py
│   ├── aggregate_seeds.py
│   └── README.md
├── 2026-08-12_empty2d_smooth_guidance/
│   ├── paired_ablation.py
│   ├── render_typical.py
│   └── README.md
├── 2026-08-12_lan_o3dp_hand_off_guidance/
│   ├── create_state_bank.py
│   ├── render_2x2.py
│   ├── rho_trajectory_sweep.py
│   ├── summarize_formal.py
│   └── summarize_rho_sweep.py
├── 2026-08-14_tabletop_guidance/
│   ├── waypoint_teacher.py
│   ├── evaluate_teacher.py
│   ├── collect_demos.py
│   ├── verify_demos.py
│   ├── analyze_demos.py
│   ├── evaluate_baseline.py
│   ├── counterfactual_rescue_eval.py
│   ├── aggregate_counterfactual_rescue.py
│   ├── open_loop_gradient_audit.py
│   ├── test_waypoint_teacher.py
│   └── README.md
├── 2026-08-25_tabletop_mvee_guidance/
│   ├── render_dual_view_preflight.py
│   ├── render_ellipsoid_geometry_preflight.py
│   ├── collect_failure_state_bank.py
│   ├── ellipsoid_counterfactual_rollout.py
│   ├── ellipsoid_margin_sweep.py
│   ├── render_failure_video.py
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
│   ├── test_same_state_diagnostic.py
│   ├── visualize_guidance_vectors.py
│   └── visualize_final_trajectory_comparison.py
└── 2026-06-22_600_rollout_eval/
    ├── run_baseline_eval_matrix.py
    └── run_baseline_eval_matrix_no_mask.py
```

## Categories

- **Tabletop RGB-D perception comparison**
  (`2026-09-15_tabletop_rgbd_perception/`) — pinned GroundingDINO Swin-T OGC,
  SAM ViT-B and Cutie provider, deterministic RGB-D unprojection/FPS, isolated
  oracle audit, prompt calibration, offline weight manifest, GPU smoke and
  paired H600 pilot. The frozen source succeeds 10/10 in both environments;
  perceived input succeeds 0/10 in both. RaisedMilk obstacle mean IoU is 0.021.
  Gate failed; do not run expansion automatically. Canonical conclusion:
  `outputs/tabletop_rgbd_perception_2026-09-15/conclusion.md`.

- **Tabletop wider demonstration support** (`2026-08-26_tabletop_wide_support/`)
  — teacher generation, 100-demo collection and verification, checkpoint
  screening, RaisedMilk evaluation, aggregation, and paired video replay.
  This uses original nearest-point guidance, not MVEE. Completed without a
  reliable safe-task gain; start with its README and
  `outputs/tabletop_wide_support/conclusion.md`. No successor is selected.

- **Diagnostic / benchmark** (in `scripts/`) — standalone scripts for per-step EEF↔obstacle tracking and OSC control timing. See docstrings for usage.
- **Clean-policy intent diagnostic** (in `scripts/2026-07-20_intent_diagnostic/`) — CPU-only offline analysis of the 600 existing unguided delta-EEF rollouts. It measures closest target progress and segment-to-footprint clearance for every pre-target chunk, selects one fixed critical chunk per episode, applies the three-way intent / obstruction partition and A/B/C Wilson decision rule, and produces deterministic CSV / JSON summaries and representative plots. Run the companion `test_analyze_intent.py` directly for the pure-Python geometry and selection tests. Canonical result: `outputs/guided_denoising/intent_diagnostic/report.md`.
- **Sample-refine** (in `scripts/2026-07-21_sample_refine/`) — archived four-candidate sampling and masked-refinement experiment. The complete method left all 20 paired task outcomes unchanged and added 76% inference time. Reproduction requires the archive tags documented in `docs/guided_denoising/sample_refine_pilot.md`.
- **Task-constrained guidance** (in `scripts/2026-07-22_task_constrained_guidance/`) — completed Stage 1 diagnostic that preserves the frozen `2 / 16 / 8` horizon contract. It captures pre-entry target-directed-obstructed states and applies a CPU-testable eight-step XY feasibility correction with swept segment clearance, target-progress, trust-region, and action-bound constraints; it preserves Z, orientation, and gripper actions. The 20-layout preregistered gate failed, so posterior-gradient Stage 2 was not implemented. See `outputs/task_constrained_guidance/stage1_report.md`.
- **LAN-O3DP reproduction** (in `scripts/2026-07-23_lan_o3dp_reproduction/`) — completed point-cloud dataset generation, HDF5/runtime parity verification, fixed-batch overfit smoke testing, and 40-demo subset preflight. The resulting point-cloud policies missed the clean-source gate; the epoch-350 guidance closure also failed its paired task gate. See `docs/guided_denoising/lan_o3dp_epoch350_closure.md`.
- **Policy-manifold feasibility diagnostic** (in `scripts/2026-07-27_policy_manifold_diagnostic/`) — completed phase-free fixed-state upper-bound test over the existing two-distractor pre-collision state bank. It samples the frozen delta-EEF policy repeatedly, fits a shrinkage covariance over executable EEF waypoint trajectories, and tests whether a target-progress-preserving path can satisfy expanded obstacle boundaries while remaining inside an empirically calibrated Mahalanobis support ellipsoid. Support-constrained projection is feasible in 1/20 states versus 14/20 when the support bound is removed; projected actions are not executed.
- **Semi-manual waypoint teacher** (in `scripts/2026-07-27_semi_manual_waypoint_teacher/`) — oracle Cartesian grasp-and-lift teacher with discrete yaw search over the 20 held-out pre-collision states. It restores every candidate from the same simulator state, selects the smallest safe yaw, replays only that route, and hands back to the frozen policy. The teacher upper bound reaches 18/20 Task success with 2/20 distractor collisions; it is a correction-data feasibility result, not a deployment method.
- **Targeted paired corrections** (in `scripts/2026-07-28_targeted_correction/`) — fresh-layout D2 collector, append-only raw HDF5 schema, per-teacher-step frozen-policy queries, deterministic pair extraction, mixed-training preflight, and paired checkpoint screening. C1 passes with 25 successful eligible episodes and 2,849 pairs. Positive BC, a 90/10 exposure control, and paper-aligned SDP all inject strict safe-progress support but fail the closed-loop D2 gate. SDP reaches 10/20 direct and 11/20 projection-feasible states while its best D2 Task SR remains 0.25 versus source 0.35. See `outputs/targeted_finetuning/sdp_result_report.md`.
- **Gated corrective-support selector** (in `scripts/2026-07-29_gated_support_selector/`) — source-default oracle gate with 32-sample SDP candidate filtering and least-deviation selection. A no-substantial-retreat preflight finds candidates in 11/20 strict states, but the paired D2 gate lowers Task SR from 0.25 to 0.15 with zero failure-to-success and two success-to-NCR transitions. The method is stopped; see `outputs/gated_support_selector/report.md`.
- **Empty2D smooth-guidance successor** (in `scripts/2026-08-12_empty2d_smooth_guidance/`) — fixed strong-guidance Circle experiment with H400/H500/H600 first-passage outcomes and mirrored obstacle offsets. At H600, offset hinge and Huberized-hinge guidance both reach 20/20 Task/Safe success versus 13/20 for their centered controls. See `outputs/empty2d_guidance/smooth_h600_circle/report.md`.
- **LAN hand-shell mechanism diagnostic** (in `scripts/2026-08-12_lan_o3dp_hand_off_guidance/`) — frozen D0--D3 full-physics / hand-off 2x2 and executed-trajectory rho sweep. Hand-shell removal modestly improves execution but does not make one-point guidance task-effective. See `outputs/lan_o3dp_hand_off_guidance/formal_seed812_n50_summary/report.md`.
- **Tabletop point-cloud guidance environment** (in `scripts/2026-08-14_tabletop_guidance/`) — controlled 3-D mechanism environment with obstacle-free direct PickPlaceCan demonstrations and static Milk/Cereal evaluation variants. Its targeted counterfactual evaluation creates 4/11 safe rescues and removes contact in 10/11 eligible pairs, but causes six collision-free stalls; the open-loop audit isolates the single-selected-point geometry limitation. See `outputs/tabletop_guidance/counterfactual_rescue_5scene_5noise/conclusion.md`.
- **Tabletop MVEE geometry gate** (in `scripts/2026-08-25_tabletop_mvee_guidance/`) — validates a task-aligned Backview, dual-view point-cloud fusion, deterministic obstacle MVEE fitting, conservative gripper / target ellipsoid guidance primitives, and full-rollout counterfactuals from freshly generated collision-failure states. See `scripts/2026-08-25_tabletop_mvee_guidance/README.md`.
- **Oracle branch value** (in `scripts/2026-07-29_oracle_branch_value/`) — cloned-simulator existence test for a ranking-only world/value model. On three strict states, 96 SDP chunks all avoid immediate contact but produce zero Can contacts and zero Task successes after full source continuation; the waypoint control solves all three. The 20-state expansion is stopped because no successful branch exists for an oracle to select. See `outputs/oracle_branch_value/pilot_3state/result_report.md`.
- **Persistent-correction oracle** (in `scripts/2026-07-29_persistent_correction_oracle/`) — temporal source-recoverability existence test with partitioned execution and validated merging. The existing SDP-6k policy is replanned for up to 20 consecutive chunks, with an exact frozen-source continuation from every safe boundary. Full coverage is 7/20 states and 27/80 trajectories; six recoverable states succeed under all four samples, while all 52 trajectories in the thirteen uncovered states fail. Recoverability windows are non-monotonic and correction alone never finishes. See `outputs/persistent_correction_oracle/full_20/conclusion.md`.
- **Empty2D guidance** (in `scripts/2026-08-07_empty2d_guidance/`) — completed point-geometry mechanism test. It generates and gates a fixed conditional-RBF absolute-XY dataset, trains a standard low-dimensional Diffusion Policy with required online W&B, screens five source checkpoints on D0, calibrates paper-aligned predicted-clean guidance independently on Circle/Square, runs 100 held-out paired rollouts per geometry, and renders environment and paired-rollout videos. Seed-42 epoch 100 passes the fixed source gate at 50/50 D0 success with zero clipping; the preregistered paper-scale H320 tests retain `Safe SR=0` and `CR=1`. The strong-guidance H600 successor above establishes the later positive regime. Seeds 43/44 were not run. See `outputs/empty2d_guidance/conclusion.md`.
- **EEF-pose OSC policy** (in `scripts/eef_pose_osc_policy/`) — dataset converters, expert replay validators, robomimic wrapper smoke tests, delta-policy pose-trajectory diagnostics, multi-env eval tooling, and eval analysis for full-pose EEF actions executed by `OSC_POSE`. Training runbook: `docs/eef_pose_osc_policy_training.md`; conclusion: `outputs/eef_pose_osc_policy/README.md`.
- **Guided denoising** (in `scripts/guided_denoising/`) — completed inference-only experiment tooling. It captures fixed clean-image delta-EEF states, sweeps guidance scales under identical observations and diffusion seeds, visualizes DDIM updates and final trajectories, runs matched baseline / guided rollout pilots, and reports Task SR together with Safe SR / CR / NCR. Preserve it for reproduction; its local-clearance result is historical evidence for the recoverability boundary, not an active experiment. Raw HDF5 / JSONL artifacts under `outputs/guided_denoising/` are ignored; summaries and reports remain trackable.
- `render_paired_rollout.py` replays selected recorded pairs, verifies the exact
  simulator-state hash and task/collision outcome, and writes synchronized
  baseline-versus-guided videos.
- **Eval matrices** (in `scripts/2026-06-22_600_rollout_eval/`) — baseline launchers from the 2026-06-22 evaluation branch. Guided / PC1 launchers were removed with the archived guidance implementation. See `docs/RESEARCH_LOG.md` (2026-06-22 entry) for the experiment background.
- **Archived forward-model / ranking results** — implementation scripts have been removed. Keep result documents only: `docs/forward_model_guidance_next_steps.md`, `docs/action_chunk_ranking_report.md`, and `outputs/forward_model/osc_eef_forward_image_v15/summary.md`.

## Convention for future experiments

Reproduction scripts go in a per-experiment subdirectory named `<YYYY-MM-DD>_<short-slug>/` (e.g. `2026-06-22_600_rollout_eval/`). Use the experiment date as the prefix.
