# Geometry-Only Guidance Diagnostic

> Historical pre-fix result: this diagnostic used the superseded
> repeated-terminal source. The absorbing-terminal source requires fresh scale
> selection and screen evaluation.

Date: 2026-07-31

## Scope

This is an explicitly authorized geometry-only diagnostic after the latent
dynamics funnel stopped. It uses the frozen RGB source policy, layout manifest,
policy seed 0, common 80-step horizon, oracle obstruction trigger, and
predicted-clean DDIM guidance implementation. It does not load latent dynamics,
inspect the untouched test split, or reopen the confirmation stage.

## Scale selection

The preregistered `0.02 / 0.05 / 0.1` grid produced exactly the same success
set as source on all 90 scale-tuning episodes. A D2 numerical smoke test showed
that `0.1` changed collision clearance by only about 0.1 mm, so the grid was
dimensionally too small for the implemented meter-valued cost.

The replacement grid `0 / 0.5 / 1 / 2 / 5 / 10 / 20` was evaluated only on
the frozen scale-tuning split (30 layouts per difficulty). Scale 5 and 10 tied
under the fixed score `D1 SR + D2 SR + 0.25 * D0 SR`; the smaller-scale
tie-break selected scale 5.

| Scale | D0 SR | D1 SR | D2 SR |
|---:|---:|---:|---:|
| 0 | 30/30 | 16/30 | 2/30 |
| 0.5 | 30/30 | 16/30 | 2/30 |
| 1 | 30/30 | 17/30 | 3/30 |
| 2 | 30/30 | 18/30 | 4/30 |
| **5** | **30/30** | **20/30** | **3/30** |
| 10 | 30/30 | 20/30 | 3/30 |
| 20 | 30/30 | 19/30 | 3/30 |

See `scale_selection.json` for the machine-readable audit.

## Paired screen

Scale 5 was then frozen and evaluated on the disjoint screen split (100
layouts per difficulty) with identical layout and diffusion seeds.

| Difficulty | Method | Success | Collision | Timeout | Mean activations |
|---|---|---:|---:|---:|---:|
| D0 | Source | 100/100 | 0/100 | 0/100 | 0.00 |
| D0 | Geometry | 100/100 | 0/100 | 0/100 | 0.00 |
| D1 | Source | 53/100 | 5/100 | 42/100 | 0.00 |
| D1 | Geometry | 58/100 | 0/100 | 42/100 | 0.18 |
| D2 | Source | 10/100 | 17/100 | 73/100 | 0.00 |
| D2 | Geometry | 16/100 | 7/100 | 77/100 | 0.32 |

Paired transitions:

- D0: 0 failure-to-success, 0 success-to-failure; exact parity.
- D1: 5 failure-to-success, 0 success-to-failure, net `+5/100`, exact paired
  `p=0.0625`.
- D2: 8 failure-to-success, 2 success-to-failure, net `+6/100`, exact paired
  `p=0.109375`.

The trigger activated on 18/100 D1 and 32/100 D2 episodes. In D1 it converted
all five source collision failures to success and introduced no regression. In
D2 it reduced collisions from 17 to 7, but two source successes became
timeouts and two source collisions also became timeouts. The 73 source timeout
episodes remained timeouts.

## Decision

Geometry guidance clearly improves local safety and produces real closed-loop
failure-to-success transitions, but it misses the preregistered rapid-screen
threshold of `F->S - S->F >= 10/100`; the best result is D2 at `+6/100`.
Therefore this diagnostic does not advance to three-seed confirmation. The
fixed-state recovery-ranking stage is unnecessary because the transition gate
already failed. Per the visualization condition, no success-only guidance GIF
is promoted as a positive result.

The mechanism boundary is informative: when the obstruction gate fires,
geometry can remove collisions, but it does not address the much larger
non-triggered timeout set or guarantee a source-recoverable post-avoidance
state.
