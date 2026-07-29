# Targeted Fine-Tuning Checkpoint Screen

Development bank only: seed 600, episode indices 0--19, with paired per-episode action seeds and deterministic torch execution. Final claims must use untouched seeds.

| checkpoint | updates | env | Task SR | Safe SR | CR | NCR | safe / succ+coll / coll fail / NCR | fail→success | success→failure | clips |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 0 | PickPlaceCan | 0.950 | 0.950 | 0.000 | 0.050 | 19 / 0 / 0 / 1 | 0 | 0 | 0 |
| source | 0 | PickPlaceBreadCerealCan | 0.350 | 0.300 | 0.550 | 0.150 | 6 / 1 / 10 / 3 | 0 | 0 | 0 |
| mixed_u02000 | 2000 | PickPlaceCan | 1.000 | 1.000 | 0.000 | 0.000 | 20 / 0 / 0 / 0 | 1 | 0 | 0 |
| mixed_u02000 | 2000 | PickPlaceBreadCerealCan | 0.050 | 0.050 | 0.300 | 0.650 | 1 / 0 / 6 / 13 | 1 | 7 | 0 |
| mixed_u04000 | 4000 | PickPlaceCan | 1.000 | 1.000 | 0.000 | 0.000 | 20 / 0 / 0 / 0 | 1 | 0 | 0 |
| mixed_u04000 | 4000 | PickPlaceBreadCerealCan | 0.250 | 0.200 | 0.300 | 0.500 | 4 / 1 / 5 / 10 | 3 | 5 | 0 |
| mixed_u06000 | 6000 | PickPlaceCan | 1.000 | 1.000 | 0.000 | 0.000 | 20 / 0 / 0 / 0 | 1 | 0 | 0 |
| mixed_u06000 | 6000 | PickPlaceBreadCerealCan | 0.250 | 0.250 | 0.350 | 0.400 | 5 / 0 / 7 / 8 | 2 | 4 | 0 |
| mixed_u08000 | 8000 | PickPlaceCan | 1.000 | 1.000 | 0.000 | 0.000 | 20 / 0 / 0 / 0 | 1 | 0 | 0 |
| mixed_u08000 | 8000 | PickPlaceBreadCerealCan | 0.250 | 0.250 | 0.350 | 0.400 | 5 / 0 / 7 / 8 | 2 | 4 | 0 |
| mixed_u10000 | 10000 | PickPlaceCan | 1.000 | 1.000 | 0.000 | 0.000 | 20 / 0 / 0 / 0 | 1 | 0 | 0 |
| mixed_u10000 | 10000 | PickPlaceBreadCerealCan | 0.250 | 0.250 | 0.300 | 0.450 | 5 / 0 / 6 / 9 | 2 | 4 | 0 |
| clean_u10000 | 10000 | PickPlaceCan | 0.950 | 0.950 | 0.000 | 0.050 | 19 / 0 / 0 / 1 | 0 | 0 | 0 |
| clean_u10000 | 10000 | PickPlaceBreadCerealCan | 0.350 | 0.300 | 0.650 | 0.050 | 6 / 1 / 12 / 1 | 0 | 0 | 0 |

Eligible mixed checkpoints: none.
No checkpoint passes the development gate.

The provisional choice still requires its update-matched clean-only control, the fixed 20-state support diagnostic, and D0--D3 evaluation on seeds 601/602.
