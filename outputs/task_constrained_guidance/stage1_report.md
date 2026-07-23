# Task-Constrained Eight-Step Feasibility: Stage 1 Report

Date: 2026-07-22
Decision: **the preregistered Stage 1 gate failed; do not implement the
posterior-gradient sampler for this geometry candidate.**

## Question and frozen contract

This experiment tested whether an oracle deployment-time optimizer could make
the frozen single-object policy useful in the specific
`pre-target + target-directed + predicted-obstruction` regime. The checkpoint
was not retrained. Observation, prediction, and executed-action horizons stayed
at `2 / 16 / 8`; every accepted correction executed all eight actions before
the next replan.

The primary optimizer changed only the XY components of the existing eight
delta-EEF actions. It preserved Z, orientation, and gripper, required continuous
EEF-point segment clearance, preserved closest target progress within 0.5 mm,
and bounded every waypoint displacement by 3.5 cm.

Primary physical parameters were:

- 10 mm EEF clearance margin, calibrated as a conservative proxy for the
  observed gripper contacts;
- recovery to the full margin within two waypoints if a later chunk began
  inside the proxy;
- 10 mm predicted-clearance trigger threshold;
- no policy, scheduler, observation, prediction, or action-horizon change.

## Layout selection

The diagnostic searched 290 baseline episodes and saved 20 strict layouts from
`PickPlaceBreadCerealCan`. Every saved state had no obstacle contact at the
chunk start, at least 10 mm start-point clearance, positive predicted target
progress, predicted clearance at or below 10 mm, and its first real pre-target
collision exactly one eight-action chunk later.

The state pools were collected independently:

| seeds | episodes searched | states |
|---|---:|---:|
| 600 | 50 | 3 |
| 601--602 | 61 | 5 |
| 603--606 | 179 | 12 |
| **total** | **290** | **20** |

## Primary results

### Short causal continuation

The constrained problem was feasible in 9/20 layouts. Among those nine paired
executions, the baseline collided within the supplied plus next chunk in seven
cases and the corrected branch in four: three collision-to-safe conversions
and no short-window regression.

All accepted first corrections met the frozen 0.5 mm progress tolerance and
3.5 cm trust bound. No clipping was added by the method. The maximum reported
action-bound excess was `1.073e-6` in the unchanged gripper coordinate and was
already present in the baseline policy output; corrected position coordinates
were in bounds.

### Full continuation to the original 400-step episode limit

Only the nine feasible layouts require a second simulation: the other eleven
receive no accepted correction and are the baseline by construction. Before
first Can contact, the corrected branch reapplied the same solver at every
eligible eight-step replan. It attempted 54 repeated corrections and accepted
25.

| metric over 9 feasible paired layouts | baseline | corrected | paired gain | paired loss |
|---|---:|---:|---:|---:|
| any Can contact | 3 | 6 | 3 | 0 |
| full task success | 0 | 2 | 2 | 0 |
| any distractor collision | 9 | 9 | 0 | 0 |

The method therefore recovered useful task intent in some layouts, including
two full successes, but it did not become a safety method: all nine full
corrected continuations still had at least one distractor collision.

## Preregistered gate

| condition | required | observed | pass |
|---|---:|---:|:---:|
| additional episodes contacting Can | >= 4 | 3 | no |
| additional full task successes | >= 2 | 2 | yes |
| correction-induced action clipping | none | none | yes |
| equal exchange into collision-free non-completion | none | none | yes |
| accepted progress and trust constraints | all | all | yes |

Because the gate was conjunctive, missing the Can-contact condition means
Stage 1 did not advance. Stage 2 noisy-action posterior gradients were not
implemented.

## Bounded failure attribution

Three observations rule out a simple guidance-scale or solver-strength fix.

1. **The EEF proxy is not a reliable gripper-safety objective.** Contact
   inspection found gripper-only contacts, and first contact could occur while
   the EEF point still had about 5.25 mm nominal clearance. Lowering the proxy
   margin from 10 mm to 6 mm on five independent layouts raised feasibility to
   3/5, but produced zero collision-to-safe conversions and one safe-to-
   collision regression.
2. **Target progress is not the sole feasibility bottleneck.** Removing the
   progress requirement in a safety-only upper-bound check made only 2/5
   layouts feasible; both still collided in the short continuation.
3. **The selected regime is not temporally closed.** Some next-chunk real
   collisions no longer satisfied target-directed predicted-obstruction, so
   the stated regime gate correctly remained inactive. Repeated correction
   cannot address these failures without broadening the method beyond the P0
   diagnosis.

Tracking error contributed but did not explain the whole result. Small
corrections were generally tracked to roughly 2 mm during the first chunk yet
could still collide in the next chunk; the one large early successful reroute
had substantially larger transient tracking error but changed the later state
enough to recover Can contact.

## Decision and implication

Stop this specific inference-only candidate: EEF-circle geometry plus
eight-step target-progress constraints is not strong enough to justify the
substantial Stage 2 implementation. Do not run another guidance-scale sweep.

This is not evidence that single-to-multi transfer is impossible. It narrows
what would have to change: a next candidate needs either a swept whole-gripper
SE(3) collision model and a broader temporally consistent task objective, or a
learned value/world model or training-side adaptation. Those are new method
claims, not parameter changes to the present formulation.

Raw state banks and per-state JSONL records remain local under
`outputs/task_constrained_guidance/`; the tracked aggregate is
`stage1_summary.json`.
