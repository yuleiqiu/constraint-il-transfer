# Tabletop guidance artifact index

This index separates the evidence needed for the current conclusion from the
many exploratory directories produced while debugging obstacle height,
guidance strength, and tangent direction. Paths are relative to this folder.

## Canonical evidence

- `counterfactual_rescue_5scene_5noise/`: targeted five-layout / five-noise
  result with 4/11 safe rescues, merged metrics, two open-loop audits, and final
  predicted trajectories. Raw partitions were deleted after an exact rebuild
  matched the merged rows, summary, report, and manifest content.
- `baseline_epoch100_eval50_h600_seed26000.json`: formal 50-rollout clean
  source validation, Task SR 35/50.
- `videos/`: representative clean-source baseline videos.
- `milk_baseline_seed817_n10/`: flat-Milk geometry check; Milk is too low to
  obstruct the carried Can.
- `milk_raised_baseline_seed819_n10/`: ten-layout raised-Milk baseline.
- `raised_milk_rho45_multilayout_seed819_ep1_4_5_7_8/`: fixed five-layout
  state bank and radial-guidance comparison.
- `raised_milk_output_tangent_smooth01_multilayout_ep1_4_5_7_8/`: final
  smooth 8 mm direct-tangent paired comparison.
- `raised_milk_output_tangent_smooth01_ep4_visual_retry/`: safe-success replay
  for episode 4; interpret together with its failed first rendered replay.
- `raised_milk_output_tangent_smooth01_typical_ep4_ep7_visual/`: failed first
  episode-4 rendered replay and representative episode-7 safe stall.

## Mechanism diagnostics retained for interpretation

- `milk_direct_visual_seed817_ep0/`: confirms the unraised Milk height
  mismatch.
- `raised_milk_rho45_stall_ep0_visual/`: radial-guidance stall.
- `raised_milk_output_fixed_tangent_ep0_h600_visual/`: abrupt 4 cm tangent
  single-state safe success; not a valid aggregate result.

## Deleted superseded exploratory groups

The following groups were deleted on 2026-08-18 after the canonical artifacts
above were verified. They contained only single-state smoke tests, incorrect
implementations, duplicate videos, or short-horizon debugging rollouts:

- `milk_*smoke*`, `milk_raised_liftgate_*`: early obstacle-height, phase-gate,
  and low-rho smoke tests.
- `raised_milk_rho40_*`, `raised_milk_rho100_*`,
  `raised_milk_rho45_h1200_*`, `raised_milk_rho_mid_*`,
  `raised_milk_rho_existence_*`: single-state rho / horizon searches.
- `raised_milk_rho45_tangent01_*`, `raised_milk_rho45_tangent05_*`: early
  denoiser-cost tangent attempts.
- `raised_milk_rho45_tangent05_forward_*`,
  `raised_milk_tangent_forward_dot_diagnostic_*`: incorrect clean-space
  direction-selection attempts.
- `raised_milk_output_tangent_ep0_h160/`,
  `raised_milk_output_tangent_forced_gate_ep0_h160/`,
  `raised_milk_output_tangent_goal_forward_ep0_h160/`,
  `raised_milk_output_fixed_tangent_ep0_h160/`: short implementation-debug
  rollouts superseded by the H600 results.
- `raised_milk_rho45_success_ep4_visual*`: radial-guidance episode-4 replay
  sensitivity checks superseded by the aggregate summaries.

The tabletop result root retains only canonical and interpretation-relevant
directories. Seven non-selected or duplicate training checkpoints were also
deleted; the exact audit is in
`checkpoint_cleanup_candidates_2026-08-18.json`.
