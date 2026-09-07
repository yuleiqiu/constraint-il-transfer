# Tabletop wider-demonstration support conclusion

Status: bounded support test complete; no reliable task-level improvement,
recorded 2026-09-07 from the 2026-08-26--2026-09-02 artifacts.

## Correct chronology and scope

This experiment is **not an MVEE evaluation**.

The immediately preceding Tabletop branch implemented dual-view obstacle
fusion, an obstacle MVEE, and gripper / carried-target ellipsoids. That line
was stopped after geometry preflights and a few small closed-loop probes:
large margins could remove contact but produced collision-free stalls. It was
not expanded into a formal matrix because the additional geometric fidelity
did not address route selection or task progress.

The successor in this directory returned to the original LAN-O3DP-style
nearest-point predicted-clean guidance. It changed the **source demonstration
support**, not the collision cost. The question was:

> If an obstacle-free source policy is trained on a broad distribution of
> task-completing transport routes, can the original point guidance select a
> safe route at deployment without destroying task completion?

This is a controlled Tabletop mechanism test. Its source is a newly trained
point-cloud absolute-EEF LAN-O3DP policy, not the project's reliable
clean-image delta-EEF epoch-260 policy. Results here must not be transferred
to that policy without a separate experiment.

## Wider-support teacher and dataset

The obstacle-free `PickPlaceCanTabletop` teacher uses a straight transport
mean plus a symmetric endpoint-conditioned RBF-GP lateral residual. The
obstacle is absent from teacher generation and is not explicitly included in
the policy observation. The purpose is to place near-direct, upper-bypass, and
lower-bypass task solutions inside the source data distribution without
teaching an obstacle-conditioned avoidance rule. Obstacle occlusion can still
alter the visible target point cloud at evaluation time.

The fixed 20-layout teacher gate passes:

- 20/20 task successes;
- maximum completion length 329 steps, below H600;
- five near-direct, five wide-upper, and six wide-lower achieved routes;
- achieved signed lateral peaks span `-0.136 m` to `+0.153 m`.

The collected dataset contains 100 successful demonstrations with a fixed
90/10 train/validation split. Across all demonstrations it contains 32
near-direct, 22 wide-upper, 22 wide-lower, and 24 intermediate routes. A
post-hoc RaisedMilk footprint proxy classifies roughly half of the paths as
safe and half as intersecting for either mirrored obstacle offset: 47/53 and
48/52 safe/collision. This is evidence that the requested route diversity was
actually collected; it is not a rollout-level avoidance result.

## Source-policy screen

The architecture-matched seed-42 LAN-O3DP model was trained for 300 epochs.
Two checkpoints were compared on the same fixed 50-state clean Tabletop bank
at H600:

| Checkpoint | Task success | Mean steps among successes |
|---|---:|---:|
| epoch 50 | 39/50 = 0.78 | 392.9 |
| epoch 250 | 23/50 = 0.46 | 422.9 |

The exact paired outcome partition is 19 both-success, 7 both-failure, 20
epoch-50-only successes, and 4 epoch-250-only successes. Epoch 50 is therefore
the source used below. The `0.78` clean success rate is adequate for this
bounded comparison but materially weaker than the reliable single-object
clean-image source used by the main project.

## RaisedMilk protocol

All formal rows use the fixed seed-828 bank of 50
`PickPlaceCanTabletopRaisedMilk` initial states and H600. Baseline and guided
conditions use matching saved initial-state hashes, layout seeds, and policy
seeds.

Guidance is the original paper-aligned point method:

- `paper_guided_policy_from_checkpoint` and
  `PaperGuidedLanO3DPUNet`;
- one closest obstacle point selected for the chunk and held fixed through
  its denoising iterations;
- XY-only Huberized-hinge predicted-clean cost;
- `rho=45`, `Q=0.08 m`, smoothing `0.01 m`;
- no tangent output correction (`tangent_ratio=0`);
- no obstacle MVEE and no gripper or carried-target ellipsoid.

Two activation modes were evaluated. `always` allows guidance during reach,
grasp, lift, and transport. `target_lifted` keeps it disabled until the Can is
at least 8 cm above the table and isolates the carried-object transport phase
that most closely resembles the Empty2D path-selection test.

## Formal observed outcomes

| Condition | Task SR | Safe SR | CR | NCR | Four-way outcome: safe / success+collision / collision failure / NCR |
|---|---:|---:|---:|---:|---:|
| baseline | 29/50 = 0.58 | 26/50 = 0.52 | 10/50 = 0.20 | 14/50 = 0.28 | 26 / 3 / 7 / 14 |
| point guidance, `always` | 9/50 = 0.18 | 7/50 = 0.14 | 9/50 = 0.18 | 34/50 = 0.68 | 7 / 2 / 7 / 34 |
| point guidance, `target_lifted` | 31/50 = 0.62 | 27/50 = 0.54 | 9/50 = 0.18 | 14/50 = 0.28 | 27 / 4 / 5 / 14 |

`always` is a clear task regression. Its paired Task transitions are four
failure-to-success versus 24 success-to-failure (two-sided exact McNemar
`p=0.000180`); Safe transitions are 3 versus 22 (`p=0.000157`). Nearly
unchanged aggregate CR hides the mechanism: guidance mostly converts task
successes into collision-free non-completions rather than providing useful
avoidance.

`target_lifted` prevents that blanket disruption but does not establish a
reliable improvement. Its paired transitions are:

- Task: 9 failure-to-success versus 7 success-to-failure, exact
  `p=0.803619`;
- Safe: 8 failure-to-success versus 7 success-to-failure, exact `p=1.0`.

The nominal net changes, `+2/50` Task and `+1/50` Safe, are small and
statistically indistinguishable from balanced paired variation. CR changes by
only `-1/50`, and NCR is unchanged.

## Narrow-support comparator

For context, the earlier direct-demo Tabletop LAN policy was evaluated on the
same RaisedMilk bank with the same `target_lifted` point-guidance parameters.
Its observed baseline/guided metrics are:

| Condition | Task SR | Safe SR | CR | NCR |
|---|---:|---:|---:|---:|
| direct-demo baseline | 17/50 = 0.34 | 5/50 = 0.10 | 34/50 = 0.68 | 11/50 = 0.22 |
| direct-demo + `target_lifted` | 2/50 = 0.04 | 1/50 = 0.02 | 4/50 = 0.08 | 45/50 = 0.90 |

Task transitions are one failure-to-success versus sixteen
success-to-failure (`p=0.000275`). Guidance removes much of the physical
contact but overwhelmingly replaces it with stalls. This comparator supports
the qualitative claim that wider source support and phase localization reduce
guidance-induced task destruction. It does **not** show that wider support is
sufficient for useful avoidance, and the two policies' absolute success rates
must not be treated as a randomized between-policy comparison.

## Replay and pairing limitation

The formal merger verifies equal episode sets, initial-state hashes, layout
seeds, policy seeds, and restored-state hashes. It does not make the GPU point
cloud policy bitwise deterministic across Python processes. Baseline and
guided conditions were produced in separate processes, and later video
replays exposed outcome drift despite matching saved states and seeds.

Across the sixteen selected paired videos rendered after the formal run, the
full `(task, collision, safe)` outcome reproduced for 9/16 baseline branches
and 13/16 guided branches. In the eight-video baseline-collision audit, exact
outcomes reproduced for only 4/8 baselines and 6/8 guided branches. These are
selected diagnostic episodes rather than a random reproducibility sample, but
they establish that nominal episode-level transitions are not strict
counterfactuals across processes.

The aggregate formal counts remain records of the trajectories that were
executed. However, the small `target_lifted` net difference is well within the
observed replay instability. It must not be reported as a causal rescue effect
or a positive task result. A future strict replication would need baseline and
guided branches evaluated in the same process with explicitly controlled
renderer / point-cloud randomness and a shared baseline. Earlier MVEE probes
used a shared-process comparison, but sharing a process alone does not prove
determinism or resolve the source of drift.

## Conclusion and decision boundary

The demonstration intervention succeeded at its directly measured gates: the
teacher and dataset contain substantially wider task-completing routes, and
the selected trained source retains 39/50 clean task success. The policy's
sampled route-mode coverage was not separately quantified, so dataset
diversity must not be cited as proof that every route mode survived training.
The deployment intervention remains inadequate.

Starting the original nearest-point guidance only after lift avoids the severe
reach/grasp disruption caused by `always`, and the wide-source experiment is
far less destructive than the direct-demo comparator. Nevertheless, the
formal observed gain is only two Task successes and one Safe success out of
fifty, with nearly balanced paired regressions, no significant effect, and a
cross-process replay limitation large enough to invalidate a strong paired
claim.

The defensible conclusion is:

> Wider task-completing demonstration support and transport-phase localization
> are not sufficient to make the original single-point predicted-clean
> guidance a reliable Tabletop avoidance method. They reduce catastrophic
> policy disruption, but do not produce a robust improvement in safe task
> completion.

The tested data intervention did not establish a solution; unmeasured learned
route coverage prevents isolating source support as a cause. The one-point
radial cost does not explicitly encode stable bypass
selection, whole-actor geometry, temporal persistence, and a principled
handoff condition. Phase gating by `target_lifted` is also an oracle task phase,
not a general deployment solution.

For future agents:

- do not describe the wide-support table as an MVEE result;
- do not claim `0.58 -> 0.62` as a positive result;
- do not expand another blind `rho` / `Q` sweep or merely add more wide demos
  to the same recipe;
- controlled replay auditing and replicated paired evaluation could test the
  small nominal effect; this is a possible follow-up, not an approved task;
- keep MVEE as an abandoned bounded geometry probe, not the active successor;
- if manipulation research resumes, the better-supported main hypothesis
  remains a correction-only temporally persistent option followed by a learned
  frozen-source recoverability / handoff estimator.

No next experiment has been selected by the user. The suggestions above are
agent interpretations of evidence, not a new research decision or supervisor
instruction. Do not automatically resume the older corrective-option plan.

## Canonical artifacts

The summary JSON, per-episode metrics, state banks, and video replay manifests
are archived in Git with the reproduction scripts. MP4 videos, training
checkpoints/logs, and the HDF5 dataset remain ignored local artifacts; a Git
checkout alone does not restore them. See docs/checkpoint_retention.md from
project root. The source config is separately committed in robomimic
(`e6408e9`); the geometry implementation predates it.

- teacher gate: `teacher_gate_seed16000_n20.json`;
- achieved dataset support and counterfactual footprint proxy:
  `tcp_transport_coverage.json` and `tcp_transport_coverage.png`;
- clean checkpoint comparison:
  `fixed_clean_checkpoint_screen_seed827_n50_parallel/paired_summary.json`;
- wide-source `always` result:
  `raised_milk_paired_seed828_n50/paired_summary.json`;
- wide-source `target_lifted` result:
  `raised_milk_paired_seed828_n50/paired_summary_target_lifted.json`;
- direct-demo comparator:
  `raised_milk_direct_demo_policy_seed828_n50/paired_summary_target_lifted.json`;
- selected paired replay videos and outcome audits:
  `raised_milk_paired_seed828_n50/videos_target_lifted/`;
- reproduction scripts and run contract:
  `../../scripts/2026-08-26_tabletop_wide_support/README.md`.
