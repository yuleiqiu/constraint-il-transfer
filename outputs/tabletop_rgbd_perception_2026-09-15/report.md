# RGB-D perception comparison — pilot

Frozen epoch-50 absolute-EEF source policy; GroundingDINO + SAM initialization and Cutie tracking.
Oracle segmentation is evaluation-only and is not supplied to the perceived policy observation.

| Environment | Condition | Task | Safe | CR | NCR | Perception failures |
|---|---|---:|---:|---:|---:|---:|
| clean | oracle_mask_source | 10/10 | 10/10 | 0/10 | 0/10 | 0 |
| clean | perceived_mask_source | 0/10 | 0/10 | 0/10 | 10/10 | 1 |

clean paired Task rescues/regressions: 0/10; Safe: 0/10.

| raised_milk | oracle_mask_source | 10/10 | 9/10 | 1/10 | 0/10 | 0 |
| raised_milk | perceived_mask_source | 0/10 | 0/10 | 0/10 | 10/10 | 0 |

raised_milk paired Task rescues/regressions: 0/10; Safe: 0/9.

## Perception diagnostics

Per-frame means below are audit metrics only; oracle values were never inserted into the perceived policy observation.

| Environment | Role | Mask IoU | 5 mm P/R | Chamfer (mm) | Centroid error (mm) | Pred/oracle points |
|---|---|---:|---:|---:|---:|---:|
| clean | target | 0.826 | 0.884/0.997 | 4.9 | 10.3 | 867.7/727.9 |
| clean | goal | 0.984 | 0.993/0.999 | 0.5 | 0.8 | 3132.6/3101.4 |
| clean | obstacle | 1.000 | 0.000/0.000 | n/a | n/a | 0.0/0.0 |

clean: 5769 perceived frames; task tracking failures 1; obstacle-invalid frames 0; identity-swap frames 0; latency p50/p95 33.3/37.4 ms.

| raised_milk | target | 0.849 | 0.908/0.994 | 6.1 | 11.0 | 800.5/696.7 |
| raised_milk | goal | 0.974 | 0.993/0.999 | 0.5 | 0.9 | 3188.6/3126.1 |
| raised_milk | milk | 0.072 | 0.076/0.100 | 244.8 | 181.8 | 996.4/1030.9 |
| raised_milk | support | 0.000 | 0.000/0.000 | 679.9 | 402.8 | 6500.6/344.5 |
| raised_milk | obstacle | 0.021 | 0.021/0.100 | 521.1 | 373.7 | 7497.0/1375.4 |

raised_milk: 6010 perceived frames; task tracking failures 0; obstacle-invalid frames 4768; identity-swap frames 1602; latency p50/p95 34.1/38.4 ms.

Gate: **FAIL**.

stop this stage; do not launch expansion, training, or planner work.

This is development-bank simulator evidence, not held-out perception or real-robot validation.
