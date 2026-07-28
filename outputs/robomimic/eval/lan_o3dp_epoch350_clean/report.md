# LAN-O3DP Epoch-350 Clean Evaluation

Checkpoint:
`model_epoch_350_image_v15_delta_eef_pose_osc_pointcloud_lan40_seed42_success_0.65.pth`

Protocol: frozen checkpoint, `PickPlaceCan`, rollout horizon 400, fixed evaluation
seeds 600 / 601 / 602, 50 independent episodes per seed.

| eval seed | successes | episodes | Task SR |
|---:|---:|---:|---:|
| 600 | 27 | 50 | 0.540 |
| 601 | 27 | 50 | 0.540 |
| 602 | 21 | 50 | 0.420 |
| **pooled** | **75** | **150** | **0.500** |

The pooled Wilson 95% interval is `[0.421, 0.579]`. The mean of the three
seed-level rates is 0.500 and their sample standard deviation is 0.069.

## Decision

`source_policy_not_acceptable_for_method_claim`

The independent clean result is 15 percentage points below the checkpoint
filename's training-time 20-rollout value of 0.65. More importantly, 0.50 is
well below the project's approximately 0.80 source-competence gate. The
checkpoint can still answer the narrow closure question—whether the
LAN-aligned representation and DDPM guidance mechanism help at all—but it
should not be used as the final source policy for a positive transfer claim.

The per-seed raw artifacts are retained under `seed600/`, `seed601/`, and
`seed602/`.
