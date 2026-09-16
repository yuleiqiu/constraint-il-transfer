# RGB-D perception comparison conclusion

Status: **stopped at pilot gate — FAIL**.

## Decision

Do not launch the 40-layout expansion, policy training, or planner integration from this stage. The frozen wide-support epoch-50 policy did not tolerate the measured RGB-D point-cloud shift, and the obstacle reconstruction is not adequate for planner use.

## Controlled policy result

| Environment | Oracle Task / Safe / CR / NCR | Perceived Task / Safe / CR / NCR | Task rescues / regressions |
|---|---:|---:|---:|
| clean | 10/10 / 10/10 / 0/10 / 0/10 | 0/10 / 0/10 / 0/10 / 10/10 | 0 / 10 |
| RaisedMilk | 10/10 / 9/10 / 1/10 / 0/10 | 0/10 / 0/10 / 0/10 / 10/10 | 0 / 10 |

All recorded actions were finite and satisfied the executable absolute-world EEF contract. The perceived zero-collision result is not a safety improvement: every perceived rollout failed the task, with all failures classified NCR.

## Perception evidence

Target/goal mean mask IoU was 0.826/0.984 on clean and 0.849/0.974 on RaisedMilk. Despite these high mean overlaps, perceived Task SR was 0/20. One clean target track became invalid at frame 359; RaisedMilk had no task-mask failure.

RaisedMilk obstacle mean mask IoU was 0.021, with 4768/6010 frames lacking a valid Milk reconstruction and 1602 identity-swap frames. Support detections do not constitute faithful support reconstruction: support IoU was effectively zero.

The result establishes a deployment shift failure for this frozen policy/provider combination. It does not by itself identify whether the dominant cause is point sampling, small temporally structured mask errors, or policy sensitivity, and it is development-bank simulator evidence rather than held-out perception or real-robot validation.

## Reproducibility boundary

GroundingDINO and SAM ran once at episode start; Cutie propagated fixed IDs with no detector retry or mid-episode reinitialization. Oracle segmentation/contact/object state were confined to evaluation. Each pair restored the same saved state and policy seed in one process and matched initial RGB-D, calibration, policy RNG, and all native non-point-cloud observations.

See [report.md](report.md), [pilot_summary.json](pilot_summary.json), [pilot_gate.json](pilot_gate.json), and [video_index.md](video_index.md).
