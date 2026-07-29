# Targeted paired-correction C1 report

Date: 2026-07-28
Status: pass

## Result

The C1 collection gate passes with 25 train-eligible, successful,
collision-free `PickPlaceBreadCerealCan` episodes. They deterministically
produce 2,849 paired correction windows:

| split | episodes | pairs |
|---|---:|---:|
| train | 19 | 2,120 |
| validation | 6 | 729 |
| total | 25 | 2,849 |

Every eligible episode passes open-loop command replay from its exact initial
simulator state. Replay reproduces final task success, target contact, and the
absence of distractor contact. All positive and negative action components
remain inside the epoch-260 source normalization range.

## Collection audit

The C1 file contains 29 raw episodes: 25 eligible selected branches and four
retained but excluded audit episodes. The append-only attempt log contains 62
fresh layouts:

| collection outcome | layouts |
|---|---:|
| accepted at collection time | 29 |
| target contact before trigger | 13 |
| no trigger before horizon | 12 |
| task failure after teacher | 7 |
| no safe teacher candidate | 1 |

Four initially accepted raw episodes are excluded from training:

| raw episode | exclusion |
|---|---|
| `demo_8` | C0 final task success did not reproduce under replay |
| `demo_17` | replay introduced a distractor contact absent from collection |
| `demo_22` | the selected branch had distractor contact before the trigger |
| `demo_24` | final task success did not reproduce under replay |

`demo_22` exposed a collection-contract loophole: the teacher interval was
safe, but the frozen policy had already contacted a distractor. The contract,
collector, and verifier now require the complete selected branch to be free of
distractor contact, including before the trigger and after policy handoff.
Three fresh eligible branches replaced the excluded C1 episodes.

The eligible yaw distribution is 21 at 0 degrees, one at -20 degrees, two at
+20 degrees, and one at +45 degrees. Thus the set contains both direct
Cartesian corrections and nonzero-yaw avoidance routes.

## Verification

The final replay maxima across all 25 eligible episodes are:

| quantity | maximum |
|---|---:|
| EEF position error | 0.6787 cm |
| EEF orientation error | 0.01366 rad |
| gripper-qpos error | 0.00840 |
| recomputed realized-action error | 0.00204 |

Normalized positive and negative actions have zero components outside
`[-1, 1]`. The most extreme positive normalized component is 0.8486; the most
extreme negative normalized component is 1.0 at the gripper command.

Canonical machine-readable verification:
`outputs/targeted_finetuning/c1/c1_verification.json`.

Final artifact fingerprints:

| artifact | SHA-256 |
|---|---|
| raw HDF5 | `6d46153af1c2ce772f8a744b2fafd25e97b912236889172bac2aa2b2365b7975` |
| pair HDF5 | `54ccc5f67b146f183a6f0867e8e05f5b42c9702ef6430f463c5bf5411ecb56aa` |
| attempt JSONL | `49c8092db14b3463384cb7dca176cf2cb0a47521eb80b58e4e7b212c02d97321` |
| verification JSON | `47376587c13f312c49773b98f0a340a0abc9a706a0086a47f8f0fe79f319d27f` |

## Decision

C1 passes. Data collection is sufficient for the preregistered small
upper-bound experiment. The next implementation step is the balanced
clean/correction positive-BC loader, followed by a no-update batch/loss
preflight and then the frozen-source, clean-only continuation, and targeted
mixed-fine-tuning comparison.
