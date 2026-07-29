# Targeted Fine-Tuning Checkpoint Screen

Development bank only: seed 600, episode indices 0--19, with paired per-episode action seeds and deterministic torch execution. Final claims must use untouched seeds.

| checkpoint | updates | env | Task SR | Safe SR | CR | NCR | safe / succ+coll / coll fail / NCR | fail→success | success→failure | clips |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 0 | PickPlaceCan | 0.950 | 0.950 | 0.000 | 0.050 | 19 / 0 / 0 / 1 | 0 | 0 | 0 |
| source | 0 | PickPlaceBreadCerealCan | 0.350 | 0.300 | 0.550 | 0.150 | 6 / 1 / 10 / 3 | 0 | 0 | 0 |
| mixed10pct_u02000 | 2000 | PickPlaceCan | 1.000 | 1.000 | 0.000 | 0.000 | 20 / 0 / 0 / 0 | 1 | 0 | 0 |
| mixed10pct_u02000 | 2000 | PickPlaceBreadCerealCan | 0.200 | 0.100 | 0.300 | 0.600 | 2 / 2 / 4 / 12 | 1 | 4 | 0 |
| mixed10pct_u04000 | 4000 | PickPlaceCan | 1.000 | 1.000 | 0.000 | 0.000 | 20 / 0 / 0 / 0 | 1 | 0 | 0 |
| mixed10pct_u04000 | 4000 | PickPlaceBreadCerealCan | 0.150 | 0.100 | 0.350 | 0.550 | 2 / 1 / 6 / 11 | 0 | 4 | 0 |
| mixed10pct_u06000 | 6000 | PickPlaceCan | 0.950 | 0.950 | 0.000 | 0.050 | 19 / 0 / 0 / 1 | 0 | 0 | 0 |
| mixed10pct_u06000 | 6000 | PickPlaceBreadCerealCan | 0.250 | 0.200 | 0.350 | 0.450 | 4 / 1 / 6 / 9 | 3 | 5 | 0 |
| mixed10pct_u08000 | 8000 | PickPlaceCan | 0.950 | 0.950 | 0.000 | 0.050 | 19 / 0 / 0 / 1 | 0 | 0 | 0 |
| mixed10pct_u08000 | 8000 | PickPlaceBreadCerealCan | 0.200 | 0.150 | 0.300 | 0.550 | 3 / 1 / 5 / 11 | 2 | 5 | 0 |
| mixed10pct_u10000 | 10000 | PickPlaceCan | 0.950 | 0.950 | 0.000 | 0.050 | 19 / 0 / 0 / 1 | 0 | 0 | 0 |
| mixed10pct_u10000 | 10000 | PickPlaceBreadCerealCan | 0.100 | 0.050 | 0.250 | 0.700 | 1 / 1 / 4 / 14 | 0 | 5 | 0 |

Eligible mixed checkpoints: none.
No checkpoint passes the development gate.

The provisional choice still requires its update-matched clean-only control, the fixed 20-state support diagnostic, and D0--D3 evaluation on seeds 601/602.
