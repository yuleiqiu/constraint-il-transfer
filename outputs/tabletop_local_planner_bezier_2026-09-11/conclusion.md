# Compact cubic Bezier bypass

Completed2026-09-11, user-authorized path shortening and smoothing. Optional
`--route-style compact_bezier`; original rectangle mode and recordings retained.

## Result

All three previously examined layouts remain safe successes, with no contact.
Each was executed twice; full repeated actions/states/trace observations match.
The six new MP4s decode and their frame counts match recorded video steps.

| Layout | Planned length rectangle / curve | Actual bypass length rectangle / curve | Actual shortening |
|---|---|---|---|
|0|52.8 /40.2 cm|53.4 /40.6 cm|23.9%|
|25|51.4 /39.3 cm|52.0 /39.3 cm|24.4%|
|28|57.6 /47.6 cm|58.3 /47.3 cm|18.9%|

Both variants: Task3/3, Safe3/3, CR0/3, NCR0/3; four-way partition3/0/0/0.
Task and safe transitions both0 rescues/0 regressions. Repeats are excluded
from these primary counts. The bounded shortening/no-regression gate passes.

Within the executed bypass interval, resample the EEF path every10mm and
measure adjacent displacement-vector angles. Maximum angles decrease from
75.0/74.4/62.1 degrees to18.0/22.4/19.7; p95 decreases from31.7/31.6/25.3 to
15.3/18.5/16.2. This indicates gentler internal turns, not a jerk measurement
or continuous-acceleration guarantee at intervention/handoff boundaries.

## Change and checks

The same MVEE envelopes,2mm fitting padding,10mm safety margin, trigger,
held height/orientation/gripper and handoff condition are preserved. Initial
observations, states and executed actions match the archived rectangle until
the first intervention; the generated starting rectangle is exactly identical.

Shorten interior vertices toward neighboring chord midpoints, retaining5mm
additional certified slack. Round corners with cubic Bezier arcs; shrink arc
extent when unsafe. Recursive subdivision bounds the entire curve's deviation
from its chord, and that bound is subtracted from certified chord clearance.
Validate joins and actual runtime command segments. A failed refinement retains
the safe original route; no safety-margin reduction is used to shorten it.

Tracking changes from stopping at sparse corners to monotone observed
arc-length progress and6mm target spacing across dense curve points. Thus this
is a combined path/tracking refinement, not an isolated spline-only ablation.
Fourteen geometry/controller tests pass, including new curve-bounding,
shortening and arc-length-progress tests. All saved executed actions match
output chunks; repeated full trajectories match on the same GPU/process.

This is still oracle-geometry, task-informed transport on three development
layouts, not full50-layout validation or evidence of universal safety.

## Artifacts

- [Videos](video_index.md): old rectangle, new curve and repeat MP4 links.
- [Actual bypass comparison](bypass_comparison.png).
- `report.md`, `summary.json`: paired path and outcome metrics.
- `artifact_integrity.json`: all six video checks.
- `gpu0/`, `gpu1/`: manifests, reset/result/repeat audits; local detailed traces,
  actions, observations and MP4s remain ignored and preserved.
- Implementation: `scripts/2026-09-11_tabletop_local_planner/smoothing.py`.
