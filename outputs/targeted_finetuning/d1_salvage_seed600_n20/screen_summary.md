# Targeted Fine-Tuning Checkpoint Screen

Development bank only: seed 600, episode indices 0--19, with paired per-episode action seeds and deterministic torch execution. Final claims must use untouched seeds.

| checkpoint | updates | env | Task SR | Safe SR | CR | NCR | safe / succ+coll / coll fail / NCR | fail→success | success→failure | clips |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 0 | PickPlaceBreadCan | 0.600 | 0.600 | 0.300 | 0.100 | 12 / 0 / 6 / 2 | 0 | 0 | 0 |
| sdp_u06000 | 6000 | PickPlaceBreadCan | 0.750 | 0.750 | 0.100 | 0.150 | 15 / 0 / 2 / 3 | 4 | 1 | 0 |

Eligible mixed checkpoints: none.
No checkpoint passes the development gate.

The provisional choice still requires its update-matched clean-only control, the fixed 20-state support diagnostic, and D0--D3 evaluation on seeds 601/602.
