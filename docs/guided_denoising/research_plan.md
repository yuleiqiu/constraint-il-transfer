# Guided Denoising Research and Evaluation Plan

Status: active research plan, 2026-07-20.

This document separates three questions that should not be answered by one
large experiment:

1. what fails when a single-object policy is deployed in multi-object scenes;
2. what the method should change at inference time; and
3. how much environment diversity is needed to support the resulting claim.

The current LAN-O3DP-style implementation is a mechanism baseline. It shows
that a frozen delta-EEF diffusion policy can be changed during denoising and
that the change reaches the executed trajectory. It is not yet the project
method.

## 1. Evidence Available Now

The unguided checkpoint has the following deployment behavior:

| distractors | Task SR | Safe SR | CR | NCR |
|---:|---:|---:|---:|---:|
| 0 | 0.907 | 0.907 | 0.000 | 0.093 |
| 1 | 0.707 | 0.687 | 0.213 | 0.100 |
| 2 | 0.253 | 0.240 | 0.647 | 0.113 |
| 3 | 0.200 | 0.153 | 0.680 | 0.167 |

The sharpest drop occurs from one to two distractors. In the two hardest
environments, most failures collide before target contact and never contact
the Can. However, collision is not equivalent to failure: successful
collisions and collision-free task failures both exist.

The first 20-pair guidance pilot changes outcomes but does not establish a
reliable benefit. Task SR increased by 0.10 in both tested environments, while
the safety effect changed sign between them. This setting must not be scaled to
the full matrix.

## 2. P0 Failure-Mechanism Diagnostic — Complete

The offline diagnostic reuses all 600 unguided rollout episodes. Its main
analysis cohort contains only task failures that never contacted Can;
successful episodes calibrate the intent measure, and failures after Can
contact are retained as a downstream comparison.

For a predicted clean chunk with start $p_0$ and reconstructed EEF waypoints
$P=(p_1,\ldots,p_H)$, target intent is the nearest approach rather than only
the endpoint:

$$
\Delta d_{\mathrm{closest}}
=d(p_0,p_{\mathrm{Can}})-\min_{1\le h\le H}d(p_h,p_{\mathrm{Can}}).
$$

A chunk is target-directed iff this 3D quantity is strictly positive.
Endpoint progress, XY progress, and 0.5 / 1 mm thresholds are sensitivity
checks. Obstruction is the minimum signed XY segment clearance between the EEF
polyline and distractor circles at the chunk start, using each robosuite
object's `horizontal_radius`; clearance at or below zero is obstructed, with
+1 / +2 cm margins as sensitivity checks.

Each episode has one critical chunk selected without reference to the computed
category: the chunk containing first obstacle contact for main episodes with
contact, the last chunk for main episodes without any contact, or the chunk
containing first Can contact for success and downstream-failure cohorts. The
three fixed, mutually exclusive categories are:

```text
target_directed_obstructed
non_target_directed
target_directed_clear
```

### Result and decision gate

| distractors | main n | target-directed obstructed | non-target-directed | target-directed clear |
|---:|---:|---:|---:|---:|
| 1 | 32 | 9 (28.1%) | 17 (53.1%) | 6 (18.8%) |
| 2 | 93 | 32 (34.4%) | 40 (43.0%) | 21 (22.6%) |
| 3 | 96 | 36 (37.5%) | 25 (26.0%) | 35 (36.5%) |

Decision **C: mixed / no single dominant mechanism**. In neither hard
environment does target-directed obstruction or non-target-directed behavior
form a majority with a Wilson 95% lower bound above 0.5. The largest category
also changes from non-target-directed with two distractors to
target-directed-obstructed with three, while target-directed-clear is nearly
co-largest in the latter. The one-distractor environment remains a trend
reference and does not decide the gate.

The primary intent measure is calibrated by successful critical chunks, where
85.8% / 92.1% / 90.0% are target-directed in the one / two / three-distractor
environments. Threshold and footprint-margin sensitivity changes proportions
but not the fixed A/B/C conclusion. P1 must therefore report regime-stratified
effects and solve one stated regime first; a pure feasibility guide must not be
presented as explaining every pre-target failure.

Canonical artifacts:
`outputs/guided_denoising/intent_diagnostic/report.md` and
`outputs/guided_denoising/intent_diagnostic/summary.json`.

## 3. Method-Development Gate in the Current Environment

Develop on `PickPlaceBreadCerealCan`, where the baseline exhibits the regime
change but still has enough successful rollouts for comparison. Treat
`PickPlaceBreadCan` and `PickPlaceBreadCerealMilkCan` as easier and harder
held-out clutter levels during method selection.

A candidate method advances only when one guidance configuration:

- improves paired Task SR on both the two- and three-distractor variants;
- does not obtain the gain by exchanging collision failures for an equal
  number of collision-free non-completions;
- preserves the single-object behavior;
- uses the same cost definition, schedule, and hyperparameters across clutter
  levels; and
- explains its gains using the diagnostic regime identified in Section 2.

The four-way outcome partition remains mandatory:

```text
safe success
success with collision
collision failure
collision-free non-completion
```

Small matched pilots are for eliminating candidates. A positive pilot must be
followed by a statistically sized evaluation and multiple independently
trained policy seeds before making a robustness claim.

## 4. Environment Expansion

Environment breadth is important, but full expansion is not the current
bottleneck. Select and freeze the suite now; run only cheap unguided scouting
until the method passes the gate above.

### Selection criteria

An environment enters the suite only if:

- the single-object/source policy has at least about 80% Task SR;
- adding distractors produces at least about a 20 percentage-point drop;
- the source and deployment tasks keep the same semantic skill and target;
- the existing delta-EEF action interface and evaluation metrics remain valid;
- failure is not primarily caused by controller instability or timeout.

### Planned robosuite suite

| family | role | when to run |
|---|---|---|
| PickPlaceCan with 0--3 distractors | main development task; long-horizon reach, grasp, transport, and placement | now |
| Lift with distractors | minimal pre-grasp test that isolates target acquisition and obstruction | baseline scout now; full comparison after the current method gate |
| NutAssemblySquare with distractors | precision and alignment test; checks whether guidance preserves constrained downstream behavior | baseline scout now; full comparison after the current method gate |

The scout should be minimal: verify source-policy competence, measure the
single-to-multi drop, and inspect a small number of failures. Do not tune the
method separately on the scout environments.

## 5. Evaluation Order and Priority

1. **P0 — failure-mechanism diagnostic (complete, decision C):** the clean
   policy exhibits a mixed partition with no mechanism dominant in both hard
   environments.
2. **P1 — method on the current family (next):** choose one stated diagnostic
   regime, develop and falsify candidates using
   paired rollouts in the two-distractor environment, then test unchanged in
   the one- and three-distractor variants.
3. **P2 — policy-seed robustness:** repeat the accepted comparison with at
   least three independently trained source-policy seeds. The current evidence
   uses one checkpoint and cannot establish training robustness.
4. **P3 — cross-task robosuite evaluation:** run the frozen method on Lift and
   NutAssemblySquare only after their unguided scouts meet the entry criteria.
5. **P4 — broader embodiment or simulator:** add this only if the intended
   claim is simulator- or embodiment-general. It is not required to decide the
   method now.

This ordering prevents environment engineering from running ahead of the
scientific question while still committing to a credible generalization test.

## 6. Claim Discipline

The intended claim should match the result of the mechanism diagnostic and the
eventual evidence:

- one PickPlace family supports a controlled single-to-multi case study;
- several robosuite task families support cross-task transfer within one
  simulator and embodiment;
- simulator- or embodiment-general language requires an additional platform.

Until the cross-task stage succeeds, describe results as a controlled study,
not as general multi-object transfer.

## 7. Terminology Migration

`Route B` was a temporary decision-tree label, not a technical concept. Active
documents should use the actual object being discussed:

- `delta EEF pose policy` for the active action representation;
- `EEF action-interface validation` for the archived controller/replay work;
- `OSC-action forward-model path` for the rejected earlier approach.

Migration should be semantic rather than a global string replacement.
Historical reports may say “historically called Route B” once when needed for
traceability. Existing directory names such as `docs/route_b_validation/` and
`outputs/route_b_validation/` should remain archival locators for now because
renaming them would touch scripts and stored references without changing the
research content. New files and active project summaries must not introduce
the old label.
