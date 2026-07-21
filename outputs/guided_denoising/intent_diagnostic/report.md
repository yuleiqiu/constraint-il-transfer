# P0 Clean-Policy Intent Diagnostic

Status: complete, 2026-07-20.

## Result

**Decision C: mixed / no single dominant mechanism.**

The diagnostic reuses the 600 existing unguided rollouts. The main analysis is restricted to task failures that never contacted Can; successful episodes calibrate the intent metric and failures after Can contact are a downstream comparison cohort.

| distractors | main n | target-directed obstructed | non-target-directed | target-directed clear |
|---:|---:|---:|---:|---:|
| 1 | 32 | 9 (28.1%; 95% CI 15.6--45.4%) | 17 (53.1%; 95% CI 36.4--69.1%) | 6 (18.8%; 95% CI 8.9--35.3%) |
| 2 | 93 | 32 (34.4%; 95% CI 25.5--44.5%) | 40 (43.0%; 95% CI 33.4--53.2%) | 21 (22.6%; 95% CI 15.3--32.1%) |
| 3 | 96 | 36 (37.5%; 95% CI 28.5--47.5%) | 25 (26.0%; 95% CI 18.3--35.6%) | 35 (36.5%; 95% CI 27.5--46.4%) |

The one-distractor environment is a trend reference only. In each of the two hardest environments, neither candidate dominant class satisfies the fixed requirement that its Wilson 95% lower bound exceed 0.5. The third class is stated explicitly in the table and decision record rather than folded into obstruction or loss of intent.

## Metric sensitivity

| distractors | closest 3D >0 | >0.5 mm | >1 mm | endpoint 3D >0 | closest XY >0 | endpoint XY >0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 46.9% | 21.9% | 18.8% | 28.1% | 40.6% | 25.0% |
| 2 | 57.0% | 53.8% | 50.5% | 53.8% | 43.0% | 38.7% |
| 3 | 74.0% | 70.8% | 65.6% | 69.8% | 60.4% | 55.2% |

| distractors | clearance <=0 | clearance <=+1 cm | clearance <=+2 cm |
|---:|---:|---:|---:|
| 1 | 71.9% | 71.9% | 71.9% |
| 2 | 60.2% | 69.9% | 75.3% |
| 3 | 53.1% | 60.4% | 67.7% |

These are marginal intent and obstruction rates; the full 3 x 3 threshold/margin category partition is in `summary.json`.

For calibration, primary closest-3D target-directed rates in successful episodes are 1 distractor: 85.8%, 2 distractors: 92.1%, 3 distractors: 90.0%.

## Definitions

For every pre-target chunk, the predicted EEF polyline is the chunk start followed by all raw predicted executable delta-EEF waypoints. Target intent is

```text
closest progress = distance(start, Can) - min distance(waypoint, Can)
```

and is positive only when the chunk gets strictly closer in 3D. Endpoint and XY variants, plus 0.5 mm and 1 mm intent thresholds, are retained in the CSV and JSON sensitivity results.

Obstruction is the minimum signed XY segment clearance from that polyline to any distractor circle at the chunk start. Circle radii come from robosuite `horizontal_radius`; zero or negative clearance is obstructed. The JSON also reports +1 cm and +2 cm margin sensitivity.

Critical chunks are fixed before computing categories: the chunk containing first obstacle contact for main episodes with contact, the last chunk for main episodes with no contact, and the chunk containing first Can contact for calibration and downstream cohorts.

## Integrity and scope

- Read 12 HDF5 files and 600 episodes; emitted 600 critical rows and 18012 pre-target chunk rows.
- Main-analysis counts are 32 / 93 / 96 for 1 / 2 / 3 distractors.
- This is an EEF-point / object-footprint diagnostic, not a collision model for the gripper, arm, or whole robot.
- No counterfactual observation, extra rollout, policy training, collision geom ID, or new guidance cost is used.

## Artifacts

- `chunk_metrics.csv`: continuous metrics for every pre-target chunk.
- `critical_episode_metrics.csv`: exactly one selected chunk per episode.
- `summary.json`: environment, seed, cohort, threshold sensitivity, decision, and representative selection records.
- `category_proportions.png` and `progress_clearance_scatter.png`: aggregate views.
- `representative_cases/`: deterministic median-nearest top-view examples in the two hardest environments.
