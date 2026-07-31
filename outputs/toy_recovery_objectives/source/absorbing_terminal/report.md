# Absorbing-Terminal Source Retraining

Date: 2026-07-31

## Change

The Toy trajectory window previously repeated the final nonzero delta action
after an episode ended. All 200 D0 demonstrations move in positive X, and
their final action has mean position delta approximately `(0.0473, 0.0002) m`.
Repeating that action taught the policy to continue through the goal.

The corrected window retains repeated leading actions only for observation
alignment. Every action slot after the recorded episode is now the absorbing
zero delta `(0, 0)`. No demonstrations were recollected or modified on disk;
the same D0 HDF5 bytes were used.

## Training

- random initialization, policy seed 0;
- 200 D0 demonstrations only;
- 5,000 updates, batch size 64, EMA weights;
- runtime: 365.902 seconds on an RTX 6000 Ada;
- final minibatch loss: 0.014709;
- five 1,000-update checkpoints retained for possible dynamics recollection.

Final checkpoint:

```text
outputs/toy_recovery_objectives/train/source_seed0_absorbing_v3/source_update_005000.pth
```

SHA-256:

```text
e37a058d03ab7078319e60f4ed77c98b43823eb60267893439ab158ab7ab53a0
```

## Calibration gate

The candidate and previously frozen layout manifests are byte-identical. The
new checkpoint passes the unchanged 50-layout-per-difficulty calibration gate.

| Difficulty | Oracle SR | Legacy source SR | Absorbing source SR |
|---|---:|---:|---:|
| D0 | 50/50 | 50/50 | 50/50 |
| D1 | 50/50 | 27/50 | 28/50 |
| D2 | 50/50 | 3/50 | 7/50 |

Outcome composition:

| Difficulty | Source | Success | Collision | Timeout |
|---|---|---:|---:|---:|
| D0 | Legacy | 50 | 0 | 0 |
| D0 | Absorbing | 50 | 0 | 0 |
| D1 | Legacy | 27 | 7 | 16 |
| D1 | Absorbing | 28 | 2 | 20 |
| D2 | Legacy | 3 | 6 | 41 |
| D2 | Absorbing | 7 | 3 | 40 |

Paired legacy-to-absorbing transitions are net `+1/50` in D1 (9 F->S, 8
S->F) and `+4/50` in D2 (6 F->S, 2 S->F). D0 has exact parity. See
`source_terminal_comparison.json` for the full transition partition.

## Interpretation and boundary

Absorbing padding removes the incorrect training target and yields a clean,
passing source checkpoint. It improves D2 success and reduces collisions, but
it does not eliminate timeout: zero terminal targets teach stopping after the
expected approach, not recovery from an off-demonstration lateral miss.

This checkpoint supersedes the legacy repeated-terminal checkpoint for future
Toy experiments. Existing dynamics and geometry-guidance results were produced
with the legacy encoder and are historical. Latent dynamics must be recollected
from the five new source checkpoints before latent guidance; geometry scale
must be retuned before any new screen. No screen or test layout was inspected
in this retraining stage.

## Verification

- 19 Toy unit tests passed;
- full CUDA train/eval smoke passed before retraining;
- final checkpoint reload reports schema `toy-recovery-source-v2`, update
  5,000, EMA weights, and the expected ResNet18-SpatialSoftmax encoder;
- calibration manifest byte parity passed.
