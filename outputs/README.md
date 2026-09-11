# Experiment outputs index

Latest implementation: `tabletop_local_planner_2026-09-11/`, MVEE local planner
using oracle compiled geometry and persistent bypass routes. Start with
`conclusion.md` and `video_index.md`; this is bounded engineering validation,
not a full-bank performance claim.

Current inclusive scoring: `tabletop_guidance_reeval_2026-09-09/inclusive_rescore/report.md`.
User-confirmed old strict OR released0.5s placement: source/guidance Task46/31,
Safe39/30 out of50; task rescues2/regressions17, safe rescues4/regressions13.
Earlier strict and residence-only reports retain their original criteria.

Latest completed ablation: `tabletop_xy_preserve_2026-09-10/`, user-authorized
guidedXY/source-nonXY composition. Source/guided/protected Task10/6/9 and
Safe9/6/9 on10 fixed development layouts; four separate layout28 pilot runs.
Expansion gate failed. Start with `conclusion.md`; videos have Markdown links.

This directory contains one top-level folder per research stage. Large model
files live only under `robomimic/train`; the other folders are compact result
records, fixed state banks, reports, and visualizations.

## Latest corrected evaluation

- `tabletop_guidance_reeval_2026-09-09/`: user-authorized corrected source /
  target-lifted comparison, all 100 branches complete, reviewed 2026-09-10.
  Task SR 56%/56%, Safe SR46%/54%; 11 safe improvements / 7 regressions,
  no statistically established aggregate benefit. Start with `conclusion.md`.
  Every actual formal execution has its own video and stepwise audit artifacts.
  `video_index.md`, `report.md` and `summary.json` provide review navigation.
  No clean-source rerun.

## Latest completed stages

- `tabletop_handoff_diagnostic/`: three-layout oracle-route / handoff diagnostic,
  45 main branches complete. Guidance safely solves 27, short help solves 43,
  long help solves 43/49 but fails the required retreat on 27. Start with
  `conclusion.md`. Also confirms a Tabletop reset-metadata scoring defect; read
  this before interpreting archived wide-support success labels.

- `tabletop_wide_support/`: wider obstacle-free demonstrations with original
  single-point guidance (not MVEE). Transport-only guidance has no significant
  Task/Safe gain and suffers replay drift; always-on guidance harms completion.
  Start with `conclusion.md`.
- `tabletop_mvee_guidance/`: stopped geometry probe with small rollout tests;
  contact removal became stalls. Contract and evidence:
  `scripts/2026-08-25_tabletop_mvee_guidance/README.md` from project root.

The bounded Tabletop diagnostic is complete; no new training is selected.
The following results remain evidence for future decisions, not an automatic
work queue.

## Retained conclusions

- `persistent_correction_oracle/`: canonical evidence for the active
  temporally persistent corrective-option candidate. Start with
  `full_20/conclusion.md`.
- `targeted_finetuning/`: positive D1 result and the closed D2
  correction-training boundary. Start with `sdp_result_report.md`.
- `tabletop_guidance/`: controlled 3-D source baseline plus the targeted
  raised-Milk counterfactual result. Start with
  `counterfactual_rescue_5scene_5noise/conclusion.md`: guidance rescues 4/11
  eligible pairs and removes contact in 10/11, but produces six safe stalls.
- `empty2d_guidance/`: completed idealized point-geometry guidance test. The
  original H320 gate is negative; the H600 offset successor reaches 20/20 Safe
  and Task success versus 13/20 for the centered control. Start with
  `smooth_h600_circle/report.md` and then `conclusion.md` for history.
- `eef_pose_osc_policy/`: reliable manipulation source-policy and executable
  action-interface evidence. Start with `README.md`.

## Supporting diagnostics

- `policy_manifold_diagnostic/`: frozen-policy support / capacity evidence.
- `gated_support_selector/`: closed selector experiment.
- `oracle_branch_value/`: one-chunk candidate-existence pilot.
- `lan_o3dp_paper_aligned/`: archived paper-aligned D2 guidance result.
- `lan_o3dp_hand_off_guidance/`: archived hand-collision-disabled guidance
  diagnostics.
- `guided_denoising/`: historical inference-only experiments.
- `task_constrained_guidance/`: archived feasibility and full-rollout checks.
- `sample_refine/`: archived candidate-sampling experiment.
- `forward_model/`: report-only archive for the removed forward-model path.

## Model storage

- `robomimic/train/`: retained checkpoints and their training logs. Follow
  `docs/checkpoint_retention.md` before deleting anything here.
- `robomimic/eval/`: compact checkpoint-evaluation metadata.

Historical snapshot after the 2026-08-18 cleanup: the full tree was 20.74 GB
(19.3 GiB). Of that,
20.33 GB is retained model/training storage; all other result folders total
about 404 MB. The Tabletop cleanup retained only its selected epoch-100
checkpoint and eleven canonical or mechanism-relevant result directories.
See `CLEANUP_2026-08-18.md` for that historical audit. These are not current
disk totals; subsequent wide-support training added substantial local storage.
See `docs/checkpoint_retention.md` from project root for retention policy.
