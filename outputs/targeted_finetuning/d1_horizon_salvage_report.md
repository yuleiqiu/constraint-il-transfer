# One-Distractor SDP and Horizon-Sensitivity Result

Status: completed paired development gate and untouched-seed validation,
2026-07-29.

## Question

The fixed two-distractor development gate showed that SDP supervision injects
local safe-progress support but does not improve complete-task success at a
400-step horizon. Does the same fixed checkpoint provide useful transfer in
the easier one-distractor environment, and does a longer execution budget
recover safe but slower trajectories?

This is a post-D2 salvage evaluation of an already trained checkpoint. No new
training, checkpoint sweep, guidance module, or stage controller was added.

## Locked comparison

- environment: `PickPlaceBreadCan` (D1);
- source: clean-image delta-EEF epoch 260;
- treatment: the already selected SDP checkpoint at 6,000 updates;
- paired initial layouts and per-episode action seeds;
- deterministic Torch execution;
- development selection: seed 600, 20 episodes;
- untouched validation: seeds 601 and 602, 50 episodes per seed;
- primary extended horizon: 600 steps, with 800 steps run to test whether the
  curve continues to improve.

The SDP checkpoint was selected before looking at D1: it is the best-Safe-SR
SDP checkpoint used by the fixed-state support diagnostic and subsequent
gated-selector experiments.

## Development gate at 400 steps

| policy | Task SR | Safe SR | CR | NCR | safe / success+collision / collision failure / NCR | failure->success | success->failure |
|---|---:|---:|---:|---:|---|---:|---:|
| source | 0.60 | 0.60 | 0.30 | 0.10 | 12 / 0 / 6 / 2 | -- | -- |
| SDP-6k | 0.75 | 0.75 | 0.10 | 0.15 | 15 / 0 / 2 / 3 | 4 | 1 |

This meets the previously used paired transition gate of at least four
failure-to-success and at most one success-to-failure transition. The
positive-BC 6k comparison did not pass: it changed Task SR from 0.60 to 0.55
and CR from 0.30 to 0.15, with two failure-to-success and three
success-to-failure transitions. It therefore showed a safety signal without
preserving task completion, while SDP passed both requirements.

## Untouched-seed result at 400 steps

| policy | Task SR | Safe SR | CR | NCR | safe / success+collision / collision failure / NCR |
|---|---:|---:|---:|---:|---|
| source | 0.66 | 0.65 | 0.25 | 0.10 | 65 / 1 / 24 / 10 |
| SDP-6k | 0.72 | 0.69 | 0.18 | 0.13 | 69 / 3 / 15 / 13 |

Across the 100 paired layouts, SDP repairs eight source failures and breaks
two source successes. The exact paired test for Task success gives
`p=0.1094`. For collision, ten layouts change from collision to no collision
and three change in the opposite direction (`p=0.0923`). Thus the 400-step
validation has favorable effect sizes but does not by itself cross a 0.05
paired-test threshold.

Per-seed Task SR is source / SDP `0.78 / 0.76` on seed 601 and
`0.54 / 0.68` on seed 602. The aggregate improvement is not a claim that the
treatment wins on every environment seed.

## Horizon sensitivity on untouched seeds

The 800-step runs exactly reproduce every success and completion time from
the corresponding 400-step prefix. Success at 400 and 600 can therefore be
read from the same deterministic extended trajectories.

| horizon | source Task SR | SDP Task SR | difference | source CR | SDP CR |
|---:|---:|---:|---:|---:|---:|
| 400 | 0.66 | 0.72 | +0.06 | 0.25 | 0.18 |
| 600 | 0.67 | **0.78** | **+0.11** | 0.26 | 0.19 |
| 800 | 0.67 | **0.78** | **+0.11** | 0.26 | 0.19 |

At 600 steps, the four-way partition is:

| policy | safe success | success+collision | collision failure | NCR |
|---|---:|---:|---:|---:|
| source | 66 | 1 | 25 | 8 |
| SDP-6k | 74 | 4 | 15 | 7 |

The paired Task transitions are 12 failure-to-success versus one
success-to-failure. The two-sided exact paired test gives `p=0.0034`. All six
SDP successes added after step 400 complete by step 473; the source adds one
success at step 415. No policy gains another success between 600 and 800.

The longer horizon also allows late collisions, so CR is evaluated at each
corresponding horizon rather than frozen at its 400-step value. SDP retains a
seven-point lower CR at 600/800, although the paired collision change
(`10` collision-to-clear versus `3` clear-to-collision) gives `p=0.0923`.

## Interpretation and decision

The targeted SDP result is not globally negative. Its useful regime is the
minimum nontrivial transfer setting:

1. in D1, policy-relative correction supervision improves Task success and
   reduces distractor collision on untouched layouts;
2. the corrective behavior is slower than the source behavior, so a
   400-step budget understates its Task success;
3. 600 steps is sufficient for the observed recovery, while 800 provides no
   further benefit;
4. in D2, the same checkpoint still fails because local corrective support
   does not form a reliable long-horizon avoidance/grasp/recovery behavior.

Use D1 at horizon 600 as the positive transfer result and D2 at horizon 400
as the identified capacity boundary. Do not describe SDP as universally
successful across clutter levels, and do not discard it as useless based only
on the D2 gate.

Inference compute time and environment horizon are distinct: extra denoising
compute does not consume simulator steps. The relevant mechanism here is that
the executed corrective trajectory progresses more slowly or takes a longer
route, which requires a larger control-step budget.

## Canonical artifacts

- untouched 400-step source:
  `outputs/targeted_finetuning/d1_final_source_seeds601_602_n50/`
- untouched 400-step SDP:
  `outputs/targeted_finetuning/d1_final_sdp_u06000_seeds601_602_n50/`
- untouched 800-step source:
  `outputs/targeted_finetuning/d1_horizon800_source_seeds601_602_n50/`
- untouched 800-step SDP:
  `outputs/targeted_finetuning/d1_horizon800_sdp_u06000_seeds601_602_n50/`

The superseded 20-rollout development-screen directory was removed during the
2026-08-18 output cleanup; its selection result remains summarized above.
