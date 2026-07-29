# Targeted paired-correction C0 report

Status: **pass**, 2026-07-28.

## Contract

- adaptation environment: `PickPlaceBreadCerealCan`;
- source: epoch-260 clean-image delta-EEF Diffusion Policy;
- horizons: observation / prediction / execution = `2 / 16 / 8`;
- collection seeds: seed 700, episodes 0--27;
- held out: seeds 600--607 and the strict 20-state diagnostic bank;
- trigger: pre-target, closest-target progress positive, predicted swept XY
  clearance at most 1 cm, and start clearance at least 1 cm;
- teacher: grasp-and-lift waypoint route with discrete yaw search;
- maximum teacher yaw increment: 0.015 rad;
- acceptance: selected teacher safe pick plus complete task success;
- pair: one shared historical action plus 15 per-step positive / negative
  actions queried along the teacher trajectory.

## Collection result

Twenty-eight fresh layouts were attempted:

| outcome | layouts |
|---|---:|
| complete successful collection branch | 11 |
| no trigger before horizon | 7 |
| target contact before trigger | 8 |
| task failure after a safe teacher pick | 2 |

One of the 11 initially accepted branches did not reproduce final task success
under open-loop command replay. It remains in the raw audit file as
`train_eligible=false`; a new layout was collected as its replacement.

The final C0 dataset contains:

| item | count |
|---|---:|
| train-eligible episodes | 10 |
| retained replay-rejected episodes | 1 |
| train pairs | 1,024 |
| validation pairs | 111 |
| total eligible pairs | 1,135 |
| selected yaw 0 deg | 8 |
| selected yaw +20 deg | 2 |

Split assignment is by episode / layout. Overlapping windows from one episode
never cross the split.

## Normalization gate

All positive teacher and negative frozen-policy actions remain inside the
epoch-260 min-max coordinate system:

| action | outside values | largest normalized magnitude |
|---|---:|---:|
| positive | 0 | 1.000 |
| negative | 0 | 1.000 |

An initial collection configuration using the diagnostic teacher's 0.04-rad
yaw increment failed this gate: two +20-degree episodes produced nine unique
rotation-z transitions reaching normalized 1.98. Those artifacts are retained
under `outputs/targeted_finetuning/rejected_maxrot004/`. Reducing the fixed
increment to 0.015 rad brought the same yaw routes within range; the largest
positive rotation-z value is 0.837 normalized.

## Replay gate

All 10 train-eligible episodes:

- restore the exact initial MuJoCo state;
- reproduce complete task success;
- reproduce target contact;
- remain free of distractor contact.

Maximum open-loop replay differences across eligible episodes are:

| quantity | maximum |
|---|---:|
| EEF position | 0.583 cm |
| EEF orientation | 0.0137 rad |
| gripper qpos | 0.00840 |
| realized action component | 0.00191 |

Intermediate flattened-state and rendered-pixel equality are not required:
the OSC desired-goal state is not contained in MuJoCo's flattened state.
Task/contact outcomes and bounded executable EEF replay are the gate.

## Decision

C0 passes. The paired representation, action alignment, collector, extractor,
normalization gate, and replay verifier are suitable for the remaining
small-dataset collection stage. This is a data-pipeline result, not evidence
that correction fine-tuning improves policy performance.

Canonical machine-readable result:
`outputs/targeted_finetuning/c0_verification.json`.
