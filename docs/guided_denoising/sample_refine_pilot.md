# Sample-Refine Diffusion Policy Pilot

Status: completed negative method-development result, 2026-07-22.

## 1. Question and decision

This experiment tested whether deployment-time batch sampling, task-constrained
candidate selection, and masked diffusion refinement could improve the frozen
delta-EEF Diffusion Policy without retraining it.

Decision: **do not advance the tested sample-refine method to held-out clutter
levels or the full evaluation matrix**. On 20 matched layouts in
`PickPlaceBreadCerealCan`, the complete sample+refine method had exactly the
same four-way outcome as baseline on every layout, while requiring 1.76 times
the inference time per action chunk.

This falsifies the current candidate, not every possible deployment-time
sampling or refinement method. Continuing this line would require a materially
different candidate-value or task-feasibility model; additional margin or
guidance-scale sweeps of the present criterion are not justified.

## 2. Tested method

The separately registered `sample_refine_diffusion_policy` leaves the original
Diffusion Policy and guided-DP implementations unchanged. It:

1. samples four independent 16-step DDIM trajectories from one observation
   encoding, with candidate 0 exactly matching baseline under the same seed;
2. evaluates the executable `[:, 1:9]` slice using nearest staged task progress
   and swept-segment XY obstacle clearance;
3. selects a positive-progress safe candidate when available, otherwise using
   the fixed progress / clearance fallback order;
4. adds the selected seed to DDIM timestep 40 and runs masked refinement at
   `[40, 30, 20, 10, 0]`, editing only executed XY delta-position dimensions;
5. accepts refinement only when both final and predicted clearance improve by
   at least 0.5 mm, task progress loses at most 0.5 mm, positive progress is
   preserved, and no action clipping is required.

Candidate sampling, refinement noise, selection diagnostics, acceptance, and
fallback are deterministic for a fixed observation and seed.

## 3. Fixed-state parameter screen

Parameter selection used only the five collision-producing fixed states from
the two-distractor development environment. Three-distractor fixed states were
held out. The grid fixed `K=4` and refinement timestep 40 while sweeping:

```text
clearance margin: 0.02 / 0.03 / 0.04 m
guidance scale:   0.001 / 0.003 / 0.01
```

The predeclared elimination gates were final-action clipping, more than 2 cm
total waypoint displacement from candidate 0, or less than 90% preservation of
originally positive task progress. Remaining settings were ranked by actual
collision-to-safe short-prefix conversions, median predicted clearance
improvement, smaller margin, then smaller scale.

The formal selected setting was:

| margin | scale | clips | max displacement | intent preservation | safety conversions / regressions | median clearance improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 m | 0.003 | 0 | 9.38 mm | 100% | 1 / 0 | 1.015 mm |

Scale 0.01 was eliminated because maximum displacement reached 23.75 mm. The
2 / 3 / 4 cm margins produced the same conversion, displacement, and intent
results at scale 0.003. Their median-clearance differences were below 0.004
micrometers, so they were treated as a numerical tie and the smaller margin was
selected.

### Sweep audit correction

The first summary incorrectly found no eligible configuration. Two independent
bookkeeping issues were corrected before parameter selection:

- refinement acceptance used a `1e-6` clipping tolerance while rollout
  evaluation used `1e-5`; the observed excess was only `1.07e-6`;
- the sweep counted clipping in a rejected refinement proposal even when the
  fallback action actually returned by the policy did not clip.

The final screen uses the rollout-consistent `1e-5` tolerance and applies the
parameter clipping gate to the final executable action. Proposal clipping
remains recorded separately. The original intermediate summaries are retained
locally for audit but are not canonical results.

## 4. Paired rollout pilot

The pilot used evaluation seeds 700 and 701, ten layouts per seed, horizon 400,
and identical initial simulator states and candidate-0 diffusion seeds across
all four conditions.

| condition | Task SR | Safe SR | CR | NCR | safe / success+collision / collision failure / NCR |
|---|---:|---:|---:|---:|---:|
| baseline | 0.10 | 0.05 | 0.75 | 0.20 | 1 / 1 / 14 / 4 |
| sample only | 0.15 | 0.10 | 0.75 | 0.15 | 2 / 1 / 14 / 3 |
| refine only (`K=1`) | 0.10 | 0.05 | 0.60 | 0.35 | 1 / 1 / 11 / 7 |
| sample + refine | 0.10 | 0.05 | 0.75 | 0.20 | 1 / 1 / 14 / 4 |

The complete sample+refine method preserved all 20 baseline outcome labels:
14 collision failures, four collision-free non-completions, one safe success,
and one success with collision. It neither improved nor regressed Task SR,
Safe SR, CR, or NCR.

The ablations changed outcomes but did not validate the full method:

- sample only converted one baseline NCR into a safe success, giving a 0.05
  Task-SR increase on one discordant pair;
- refine only converted three collision failures into three NCRs, reducing CR
  by 0.15 without changing Task SR;
- sample+refine did not retain the sample-only gain and produced no discordant
  outcome pairs relative to baseline.

## 5. Mechanism diagnostics

Candidate selection and refinement were active rather than silently disabled:

| condition | nonzero selected-candidate rate | refine accepted / triggered | inference overhead | action clips |
|---|---:|---:|---:|---:|
| sample only | 42.3% | — | 1.00x | 0 |
| refine only | 0% | 160 / 983 (16.3%) | 1.77x | 0 |
| sample + refine | 46.2% | 188 / 981 (19.2%) | 1.76x | 0 |

The baseline critical-chunk partition contained nine non-target-directed, four
target-directed-clear, and seven target-directed-obstructed layouts. The one
sample-only success gain occurred in the target-directed-clear stratum. In all
seven target-directed-obstructed layouts, sample+refine retained zero Task SR
and CR 1.0. Refine only reduced collision in this stratum once, but converted
that collision failure into NCR rather than task success.

These results are consistent with P0 decision C: a geometric feasibility
criterion does not address the full mixed failure distribution. More
importantly, it did not solve even the obstruction-labelled pilot stratum in
task-success terms.

## 6. Scope and next-step constraint

The pilot uses one pretrained policy checkpoint and 20 development layouts, so
the isolated sample-only win is not a robustness result. Because the complete
method failed the development gate, no three-distractor, single-object, policy-
seed, or cross-task evaluation was run.

Do not continue by scaling this setting or sweeping nearby margins and scales.
A future candidate should first demonstrate that its selection or acceptance
score predicts executable task completion rather than clearance alone.

Canonical artifacts:

```text
outputs/sample_refine/fixed_state_sweep_final/summary.json
outputs/sample_refine/paired_pilot_d2_n20/manifest.json
outputs/sample_refine/paired_pilot_d2_n20/summary.json
outputs/sample_refine/paired_pilot_d2_n20/report.md
```

Archive revisions:

```text
root repository tag:      archive/s2m-sample-refine-20260722
robomimic repository tag: archive/s2m-sample-refine-20260722
```

At archive time, the robomimic development baseline was `exp/guided-dp`, which
intentionally did not contain the failed sample-refine algorithm. Check out the
robomimic archive tag temporarily when reproducing this pilot.
