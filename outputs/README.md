# Experiment outputs index

This directory contains one top-level folder per research stage. Large model
files live only under `robomimic/train`; the other folders are compact result
records, fixed state banks, reports, and visualizations.

## Latest completed stages

- `tabletop_wide_support/`: wider obstacle-free demonstrations with original
  single-point guidance (not MVEE). Transport-only guidance has no significant
  Task/Safe gain and suffers replay drift; always-on guidance harms completion.
  Start with `conclusion.md`.
- `tabletop_mvee_guidance/`: stopped geometry probe with small rollout tests;
  contact removal became stalls. Contract and evidence:
  `scripts/2026-08-25_tabletop_mvee_guidance/README.md` from project root.

No next experiment is selected. The following results remain evidence for
future decisions, not an automatic work queue.

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
