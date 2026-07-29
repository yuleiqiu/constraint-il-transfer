# SDP: Project-Specific Analysis

Paper: *Set-Supervised Diffusion Policy: Learning Action-Chunking Diffusion
through Corrections*.

## What the method actually changes

For each paired correction `(O, A-, A+)`, SDP defines a per-timestep ball
centered on the positive action:

$$
\hat{\mathcal A}_i
=
\{a:\|a-a_i^+\|_2
\le r\|a_i^+-a_i^-\|_2\}.
$$

The negative action is not trained with a repulsion or preference-ranking
loss. It determines the radius of the acceptable set. SDP then samples from
the current diffusion policy while constraining each denoising update to the
set; an out-of-set single-step action is reset to the corresponding positive
action. The policy is trained with the ordinary diffusion denoising loss on
these constrained target chunks instead of directly imitating `A+`.

The important preservation mechanism is therefore

$$
\pi_{\rm target}(A\mid O)
\propto
\mathbf 1[A\in\hat{\mathcal A}]\,\pi_\theta(A\mid O),
$$

not “push away `A-`.” In principle this keeps targets close to the current
policy while ruling out action regions inconsistent with the correction.

## Why it is now relevant

The project has already built the required immutable pair dataset with exact
`2 / 16 / 8` alignment. The 50/50 positive-only BC experiment gives a sharper
motivation than paper novelty alone:

- source and clean-only have 0/20 direct safe-progress hits and 1/20 feasible
  support projections at the strict 1 cm margin;
- positive BC changes these to 10/20 and 13/20, so corrections do inject the
  intended local support;
- the same checkpoint lowers paired D2 Task SR from 0.35 to 0.25, with two
  failure-to-success and four success-to-failure transitions;
- D0 stays at 1.00, so the failure is not global clean-skill forgetting.

The remaining problem is consistent with overly pointwise corrective
supervision or unstable mode selection. SDP directly tests whether a
policy-relative set target preserves useful task behavior better than
imitating the teacher point.

## Mapping to this repository

| SDP concept | Project representation |
|---|---|
| observation `O` | two policy observations from the pair HDF5 |
| chunk length `T=16` | existing 16-action Diffusion Policy target |
| execution horizon `T_a=8` | existing executed slice `1:9` |
| positive chunk `A+` | normalized `positive_actions` |
| negative chunk `A-` | normalized `negative_actions` queried along the teacher trajectory |
| shared history | chunk index 0 has `A-=A+`, hence a zero-radius singleton |
| clean demonstrations | retain ordinary BC loss and the original HDF5 |
| correction supervision | apply SDP target sampling only to pair samples |

All set distances must be computed after applying the fixed epoch-260 action
normalization. Raw mixed-unit position, rotation, and gripper distances are
not meaningful.

## Important mismatches and risks

1. The project assembles future single-step robot queries collected along the
   teacher trajectory into one negative chunk conditioned on the starting
   observation. This is the same approximation acknowledged by SDP, but it
   remains a limitation.
2. The verified waypoint teacher is accurate at the episode level, but the
   overlapping windows are highly correlated. SDP does not remove the need
   for episode-level train/validation splits or balanced sampling.
3. A set centered at `A+` can still overrepresent corrections. Set supervision
   may reduce point-target overfitting, but success is not guaranteed.
4. The paper uses normalized action spaces, `r=0.1`, `N=16`, 100 diffusion
   steps, and a truncated target sampler beginning at `K_A=16` for its
   accurate online correction settings. These are sensible first values, not
   evidence that they transfer unchanged.
5. Desired-sample generation depends on the current policy and is expensive.
   It must run without gradients and must not accidentally use stale EMA
   weights or mutate rollout state.

## Minimal fair comparison

Reuse the existing training and development contracts:

1. frozen source;
2. 10,000-update clean-only continuation, already complete;
3. 50/50 paired-positive BC, already complete;
4. 50/50 clean BC plus SDP supervision on correction pairs.

Use `r=0.1`, `N=16`, and `K_A=16` as the paper-aligned starting point. Keep
the source normalization, optimizer, learning rate, batch size, update count,
checkpoint cadence, and EMA evaluation unchanged. Include at most one 90/10
clean / positive-BC control to determine whether any SDP gain is merely the
effect of weakening correction exposure; do not launch a broad ratio sweep.

First require unit and fixed-batch tests for set membership, projection,
normalization, target-sampler determinism, no-gradient behavior, and unchanged
clean BC loss. Then use the existing seed-600 D0/D2 screen and strict 20-state
diagnostic. Only a checkpoint passing that development gate proceeds to the
untouched D1/D3 seeds.

## Interpretation boundary

A positive SDP result would show that paired set supervision uses the same
corrections more effectively than positive-only BC. It would not establish a
world model, remove the oracle nature of correction collection, or prove
general multi-object transfer from 25 episodes.

## Project outcome (2026-07-29)

The paper-aligned first comparison was implemented with `r=0.1`, `N=16`, and
`K_A=16` and trained for 10,000 updates. The target sampler was mechanically
nontrivial: final policy-sample retention rose from about 2% at source
initialization to about 82% on later training batches. Nevertheless, no
screened checkpoint improved D2 Task SR over source. The best SDP checkpoint
obtained 0.25 versus source 0.35, with two failure-to-success and four
success-to-failure transitions.

At the same checkpoint, strict 1 cm support coverage reached 10/20 direct
safe-progress states and 11/20 projection-feasible states. The negative result
therefore does not mean SDP ignored the correction pairs. In this project, set
supervision preserved the local support injection but did not solve when the
full policy should select and leave the corrective behavior. A 90/10
positive-BC control failed as well, so reduced correction exposure is not a
sufficient explanation. See
`outputs/targeted_finetuning/sdp_result_report.md`.
