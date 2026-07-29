# SDP Successor Experiment Plan

Status: completed; both the SDP treatment and 90/10 exposure control failed
the development gate, 2026-07-29. See
`outputs/targeted_finetuning/sdp_result_report.md`.

## Objective

Test whether policy-relative set supervision can retain the safe-progress
support injected by corrections without replacing previously successful D2
task behavior. Use the existing immutable `(O, A-, A+)` dataset; collect no new
episodes.

## Fixed controls

- source epoch-260 checkpoint;
- 10,000-update clean-only continuation;
- 10,000-update 50/50 paired-positive BC;
- new 10,000-update 50/50 clean-BC / correction-SDP treatment.

All four use the same observation and action interfaces, epoch-260
normalization, batch size 64, learning rate `1e-5`, fresh optimizer where
applicable, checkpoint cadence, and EMA evaluation.

One additional 90/10 clean / paired-positive BC run is permitted as an
exposure control. Its purpose is singular: determine whether weakening
positive-correction frequency can recover task preservation. Do not tune this
ratio on rollout results.

## Initial SDP definition

On normalized correction pairs, construct one Euclidean ball per timestep:

$$
\|a_i-a_i^+\|_2 \le 0.1\|a_i^+-a_i^-\|_2.
$$

Generate `N=16` desired chunks from the current policy with a truncated
100-step training scheduler starting at `K_A=16`. After each target-denoising
step, retain in-set actions and replace out-of-set actions with `a_i^+`.
Generate targets under `torch.no_grad()` and detach them before the ordinary
diffusion noise-prediction loss. Clean samples continue to use their original
action targets.

The shared historical action at chunk index 0 has zero radius and must remain
exactly equal to the positive/history action. Set geometry is evaluated only
in the fixed normalized action coordinates.

## Preflight gates

Before training:

1. the pair loader exposes both positive and negative normalized chunks while
   preserving the existing positive-BC interface;
2. pure tensor tests cover radius, membership, projection, zero-radius
   history, and batch/time dimensions;
3. every generated desired action lies in its set within tolerance;
4. target generation is deterministic for a fixed seed, creates no parameter
   gradients, and leaves observations and dataset buffers unchanged;
5. `r -> 0` numerically reduces the correction target to positive BC;
6. a CUDA fixed-batch update is finite and updates the intended online
   parameters and EMA only;
7. a clean-only batch reproduces the existing BC loss path.

Record target-generation time and peak GPU memory. If `N=16` is infeasible,
use the paper's cached-target variant with an explicit refresh contract rather
than silently reducing the method.

## Development evaluation

Inspect checkpoints at the same coarse updates as positive BC on:

- seed-600 D0 and D2, 20 fixed episodes each;
- the strict 20-state support diagnostic.

The SDP treatment must preserve the already demonstrated support gain while
meeting the existing closed-loop gate: D0 Task SR at least 0.85, at least four
D2 failure-to-success and at most one success-to-failure transition, no NCR
substitution, and no runtime clipping. If no checkpoint passes, stop before
the untouched final seeds.

If a checkpoint passes, evaluate the same selected checkpoint and its
update-matched controls on D0--D3 using seeds 601/602. Report Task SR, Safe SR,
CR, NCR, the four-way partition, paired transitions, support coverage, and
wall-clock training cost.

## Result and decision

The full paper-aligned SDP chain passed loader, geometry, normalization,
no-gradient, clean-loss-equivalence, CUDA memory, optimizer, and EMA preflight
checks. The 10,000-update treatment completed without numerical or runtime
failure. A 90/10 positive-BC exposure control also completed.

Neither treatment improved closed-loop D2 Task SR over the source:

- source: 0.35;
- best 90/10 checkpoint: 0.25, with 3 failure-to-success and 5
  success-to-failure transitions;
- best SDP checkpoint: 0.25, with 2 failure-to-success and 4
  success-to-failure transitions.

The SDP 6,000-update checkpoint nevertheless reaches 10/20 direct
safe-progress states and 11/20 support-feasible states at the strict 1 cm
margin. Thus set supervision preserves the injected local support but does not
solve task-preserving mode selection. Stop before D1/D3 and seeds 601/602.
Do not use a broad ratio or radius sweep as the next experiment. Canonical
result: `outputs/targeted_finetuning/sdp_result_report.md`.
