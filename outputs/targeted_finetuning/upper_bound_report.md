# Targeted Positive-BC Upper-Bound Result

Status: completed development gate, 2026-07-28.

## Question

Can 25 verified two-distractor correction episodes inject the missing
task-preserving avoidance support into the epoch-260 delta-EEF Diffusion
Policy without losing the original PickPlace skill?

The tested treatment uses ordinary diffusion noise-prediction loss on positive
correction chunks. It is an upper bound on full-policy adaptation, not the
proposed final method.

## Fixed comparison

All learned variants start from the same epoch-260 checkpoint, use a fresh
optimizer, learning rate `1e-5`, batch size 64, and 10,000 updates. The
treatment samples 50% original clean sequences and 50% intervention-centered
positive correction pairs. The control continues on clean data only. Source
action normalization is fixed for all conditions.

The checkpoint screen uses seed 600, episode indices 0--19, and the same
initial layouts across policies. Torch is configured for deterministic
execution and each episode receives a fixed action seed independent of
previous rollout length. This is a development bank; seeds 601/602 remain
untouched because no treatment checkpoint passed the development gate.

## Closed-loop screen

| condition | updates | D0 Task SR | D2 Task SR | D2 Safe SR | D2 CR | D2 NCR | D2 safe / succ+coll / coll fail / NCR | fail→success | success→failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 0 | 0.95 | 0.35 | 0.30 | 0.55 | 0.15 | 6 / 1 / 10 / 3 | — | — |
| mixed positive BC | 2,000 | 1.00 | 0.05 | 0.05 | 0.30 | 0.65 | 1 / 0 / 6 / 13 | 1 | 7 |
| mixed positive BC | 4,000 | 1.00 | 0.25 | 0.20 | 0.30 | 0.50 | 4 / 1 / 5 / 10 | 3 | 5 |
| mixed positive BC | 6,000 | 1.00 | 0.25 | 0.25 | 0.35 | 0.40 | 5 / 0 / 7 / 8 | 2 | 4 |
| mixed positive BC | 8,000 | 1.00 | 0.25 | 0.25 | 0.35 | 0.40 | 5 / 0 / 7 / 8 | 2 | 4 |
| mixed positive BC | 10,000 | 1.00 | 0.25 | 0.25 | 0.30 | 0.45 | 5 / 0 / 6 / 9 | 2 | 4 |
| clean-only control | 10,000 | 0.95 | 0.35 | 0.30 | 0.65 | 0.05 | 6 / 1 / 12 / 1 | 0 | 0 |

All conditions have zero runtime action clipping. D0 is
`PickPlaceCan`; D2 is `PickPlaceBreadCerealCan`. Transitions are paired
against source on the same 20 D2 layouts.

No mixed checkpoint reaches source D2 Task SR, let alone the preregistered
requirement of at least four failure-to-success and at most one
success-to-failure transition. The 10,000-update clean-only control preserves
exactly the source success/failure assignment, so the task regression is not
explained by continued optimization alone.

At 10,000 updates, 13/15 mixed-policy failures never contact Can. The source
also usually fails before target contact, but on only 12/13 failures. Positive
BC therefore does not reliably convert the dominant approach obstruction into
a completed pick-and-place; it also removes previously successful behavior.

## Fixed-state support diagnostic

The strict diagnostic uses the existing 20 source pre-collision states, 32
policy samples per state, the fixed target-progress tolerance, and the 1 cm
expanded obstacle boundary.

| condition | states with direct safe+progress sample | support-projection feasible |
|---|---:|---:|
| source | 0/20 | 1/20 |
| clean-only, 10,000 updates | 0/20 | 1/20 |
| mixed positive BC, 10,000 updates | 10/20 | 13/20 |

The mixed policy exceeds the preregistered 8/20 support gate by a large margin.
This increase is absent from the update-matched clean control. The correction
pipeline therefore did inject local safe-progress action support; the negative
closed-loop result is not evidence that the teacher data were ignored.

## Decision

Reject the tested 50/50 paired-positive BC recipe as a closed-loop upper
bound. Do not advance it to D1/D3 or the untouched final seeds, and do not
collect more corrections of the same kind yet.

The combined evidence localizes the next problem:

1. verified corrective actions alter the learned distribution in the intended
   local direction;
2. clean-domain skill remains intact;
3. the treatment nevertheless overuses or misselects the corrective mode and
   disrupts previously successful D2 trajectories.

This is precisely the distinction discarded by positive-only BC: it observes
the desired action but not the paired source action that was rejected at that
state. The next method experiment should reuse the immutable
`(O, A-, A+)` file and compare SDP-style set supervision against the existing
positive-BC result. Keep ordinary BC on clean rehearsal data; apply set-valued
targets only to correction pairs. A single lower-exposure positive-BC control
may be included to separate the SDP objective from simply weakening the 50/50
correction weight, but a broad ratio sweep is not justified.

Successor status (2026-07-29): the preregistered SDP and 90/10 exposure-control
experiments were completed and also failed the D2 gate despite retaining
substantial support gains. See
`outputs/targeted_finetuning/sdp_result_report.md`.

## Canonical artifacts

- checkpoint screen:
  `outputs/targeted_finetuning/checkpoint_screen_seed600_n20/screen_summary.md`
- source fixed-state diagnostic:
  `outputs/policy_manifold_diagnostic/fixed_state_d2/report.md`
- mixed fixed-state diagnostic:
  `outputs/targeted_finetuning/support_diagnostic/mixed_u10000/report.md`
- clean-control fixed-state diagnostic:
  `outputs/targeted_finetuning/support_diagnostic/clean_u10000/report.md`
- data report: `outputs/targeted_finetuning/c1_report.md`
