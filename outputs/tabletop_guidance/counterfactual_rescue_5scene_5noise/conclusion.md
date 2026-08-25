# Tabletop paired counterfactual rescue conclusion

Status: small targeted five-layout / five-noise evaluation complete, 2026-08-24.

## Question and protocol

This evaluation asks whether paper-aligned radial predicted-clean guidance can
rescue the subset of Tabletop rollouts for which the source policy succeeds
without an obstacle but fails through collision after a raised Milk obstacle is
inserted. It does not use source-policy failures or obstacle rollouts that still
complete the task in the rescue denominator.

For each of five previously fixed RaisedMilk layouts (episodes 1, 4, 5, 7, and
8), five fixed diffusion-noise seeds (`910000`--`910004`) run three H600
branches:

1. obstacle-free source environment, no guidance;
2. matched raised-Milk environment, no guidance;
3. matched raised-Milk environment with radial guidance.

The dynamic simulator state, Can pose, GoalTray pose, and EEF pose match
numerically between the clean and obstacle branches. Guidance uses the existing
canonical setting `rho=45`, `Q=0.08 m`, XY-only Huberized hinge with `0.01 m`
smoothing, and no output-tangent correction. The obstacle is not an explicit
policy input, although camera occlusion changes the target-only point cloud;
the initial task-pointcloud maximum absolute difference is 0.495--0.560 across
the five layouts.

An eligible pair is exactly:

`clean.success AND obstacle_baseline.collision_failure`.

A rescue is exactly:

`eligible AND obstacle_guided.safe_success`.

## Paired result

Across all 25 triads, the clean source succeeds in 17. The eligibility filter
retains 11 pairs. Guidance produces four safe task-completing rescues:

- Rescue Rate: **4/11 = 0.364**;
- naive seed-level Wilson 95% interval: **0.152--0.646**;
- guided outcomes on eligible pairs: 4 safe successes, 6 collision-free
  non-completions, and 1 collision failure.

The result is strongly layout-dependent:

| Layout | Clean success | Eligible | Safe rescues | Rescue rate |
|---:|---:|---:|---:|---:|
| 1 | 5/5 | 0 | 0 | n/a |
| 4 | 5/5 | 5 | 4 | 0.80 |
| 5 | 0/5 | 0 | 0 | n/a |
| 7 | 4/5 | 4 | 0 | 0.00 |
| 8 | 3/5 | 2 | 0 | 0.00 |

All four rescues occur in layout 4. Seeds within a layout are clustered rather
than independent layout samples, so 4/11 must not be presented as a general
raised-Milk success rate. At the layout level, only one of the three layouts
that contain eligible pairs has any rescue.

## Collision-pair result

Every eligible obstacle-baseline rollout has carried-Can/Milk contact (11/11).
Five also have gripper/Milk contact, split across four hand and four finger
episodes; some episodes contain both. One has carried-Can/MilkSupport contact.
No eligible rollout has an arm-link collision.

Under guidance, physical obstacle contact falls from 11/11 eligible rollouts to
1/11. The remaining contact contains carried Can, gripper hand, and Milk. The
EEF itself is a MuJoCo site rather than a contact geom; its XY Milk-footprint
overlap proxy falls from 10/11 to 1/11.

This is strong evidence of a real geometric repulsion effect. It also confirms
the cost mismatch: the dominant physical collision is carried Can versus Milk,
while the cost directly constrains only EEF XY position.

## Open-loop gradient audit

The extended audit uses a pre-contact chunk from layout 4 / seed 910000. Its
baseline first contacts Milk seven executed steps after the audited chunk
starts. Baseline and guided reverse processes use the exact same observation
and initial diffusion tensor.

- all 100 reverse timesteps have nonzero `grad(A_k) J`;
- all eight executable waypoints are hinge-active at every reverse timestep;
- full normalized guidance-update norm is 0.087--0.371, with executable-slice
  norm 0.045--0.294;
- the final executable XY targets differ from baseline by 1.31 cm on average
  and 2.21 cm at maximum;
- clearance from the one fixed selected obstacle point rises from 5.77 mm to
  8.30 mm on the closest executable waypoint;
- nevertheless, minimum clearance to the full visible obstacle cloud changes
  from 3.24 mm to 2.59 mm and remains far inside `Q=80 mm`.

Therefore none of the first two failure diagnoses applies: the threshold is
active, the gradient is nonzero, and scheduler/denoiser dynamics do not erase
the final executable change. The audit instead exposes a limitation of the
single fixed nearest-point cost: it moves the trajectory away from the selected
point without making the whole predicted executable chunk geometrically safe.
Closed-loop replanning can still turn this displacement into a rescue in layout
4, but does not do so reliably in layouts 7 and 8.

The same nominal layout/noise audit was replay-sensitive: one replay ended as a
collision failure with first contact at step 182, while the extended audit
replay first contacted at step 151 and later completed with collision. The
within-audit baseline/guided open-loop comparison is exact, but the closed-loop
result is not bitwise replay-stable. This reinforces the small-sample claim
boundary.

## Conclusion

The proposed blanket conclusion that rescue rate remains near zero is **not
supported** by this targeted experiment. The tested guidance can create true
safe task-level rescues in the simplified 3-D environment, reaching 4/5 on one
eligible layout, and it removes physical collision in 10/11 eligible pairs.

It is also **not a robust task-level avoidance solution**. Six of eleven
eligible pairs become collision-free stalls, one still collides, and two
eligible layouts have zero rescues. The most defensible conclusion is:

> LAN-O3DP-style predicted-clean EEF guidance has a mechanically active and
> sometimes task-completing avoidance regime in the controlled 3-D Tabletop
> environment, but its success is sharply layout-dependent. It usually removes
> contact, yet often trades collision for non-completion because a one-point EEF
> cost does not encode carried-object geometry or a persistent task-compatible
> bypass and handoff.

This result motivates route persistence / carried-object-aware geometry rather
than another scalar sweep of `rho` or `Q`.

## Canonical artifacts

- merged metrics and generated report: `merged/`;
- exact per-episode physical contact pairs: `merged/episode_metrics.jsonl`;
- extended timestep audit: `open_loop_scene4_seed910000_v2/audit.json`;
- baseline/guided final clean trajectories and fixed noise:
  `open_loop_scene4_seed910000_v2/final_trajectories.npz`;
- first strict-failure audit replay: `open_loop_scene4_seed910000/`;
- partition provenance: `merged/manifest.json`. The four raw partitions were
  deleted after a `/tmp` rebuild matched the retained merged metrics, summary,
  report, and non-path manifest content.
