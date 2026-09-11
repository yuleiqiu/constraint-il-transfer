# MVEE local planner: implementation and bounded closed-loop validation

Completed2026-09-11. Frozen wide-support epoch50 plus a root-local persistent
bypass planner, using the fork's MVEE fitting. No model training or fork edits.
The implementation gate passes; no full50-layout validation was launched.

## Primary outcomes

Criterion: old strict success OR released placement held0.5s. H600, identical
initial states and policy seeds. Safety counts robot/target obstacle contact
through the earliest success. All primary runs and repeats have their own MP4.

| Layout | Source | Local planner | Trigger / handoff step |
|---|---|---|---|
|0|safe success,331 steps|safe success,331 steps|168 /280|
|25|success with contact,459 steps|safe success,441 steps|272 /384|
|28|success with contact,470 steps|safe success,347 steps|160 /288|

Primary counts exclude repeats:

| Condition | Task SR | Safe SR | CR | NCR | Safe / success+contact / collision failure / NCR |
|---|---:|---:|---:|---:|---|
|Source|3/3|1/3|2/3|0/3|1 /2 /0 /0|
|Local planner|3/3|3/3|0/3|0/3|3 /0 /0 /0|

Task transitions:0 rescues /0 regressions. Safe transitions:2 rescues (25,28) /
0 regressions. This meets the predeclared engineering gate of at least one
extra safe completion with no lost source task success.

## What was executed

The source runs until a grasped/lifted transport chunk fails the conservative
ellipsoid-clearance check. A visibility graph generates an explicit bypass
around the obstacle's inflated footprint. The route, entry height/orientation
and closing gripper persist across chunks. Observed progress advances route
corners; immediate output motion is rechecked. Handoff requires route completion
and a clear, non-substantially-backward source chunk. All three tested cases
execute bypasses, hand off, and then finish with the source.

Source sampling is unchanged when the planner is inactive. It is still queried
at actual intervention-produced observations while the planner is active.
This does not eliminate intervention-induced state OOD or guarantee handoff
recoverability. No original denoising guidance is used in planner runs.

## Verification and boundaries

- 11 geometry/controller tests passed. Tests include swept interior collision,
  rotating actor offsets, parity with existing MVEE point clearance, workspace
  infeasibility, exact source passthrough, persistent channels and handoff risk.
- 12 actual executions:6 primary plus6 repeat branches. Layout0 repeats exactly
  in one process;25/28 repeats exactly in a separate GPU1 process. Full recorded
  states/actions/trace observations match. Every pair's initial observations
  match. All source runs match the available archived source trajectory prefix.
- 12 MP4s decoded with frame counts verified. Saved executed actions exactly
  match flattened output chunks. See `artifact_integrity.json`.
- Source snapshots are retained in `executed_source/`; final code formatting has
  identical ASTs. Fork revisions and hashes are in `provenance.json`.

These are three deliberately chosen, previously examined development layouts,
not held-out or population results. Repeats improve execution confidence but
are not additional independent layouts. No general SR improvement is established.

This prototype uses oracle compiled collision geometry and known goal position.
It does not yet use the existing dual-view pointcloud provider. Ellipsoid fitting
is reused; continuous interpolated-pose bounds and route control are new. No
full-arm reachability, self-collision or task-fixture constraints are certified.
Controller tracking, articulation and grasp slip are not guaranteed by geometric
checks; actual contacts are assessed from the closed-loop simulator records.

The safe source case0 also triggers14 modified chunks. Thus the conservative
trigger can intervene unnecessarily. Sharp route corners, long detours,
no-route holds, grasp loss and short-horizon handoff remain limitations to test
on wider layouts. The current success is not attributed to geometry alone:
explicit bypass, persistent control and protected transport channels changed
together relative to source.

## Artifacts

- `video_index.md`: all12 MP4s, direct VS Code links, no HTML.
- `executed_paths.png`: actual EEF paths and planned bypasses; not collision proof.
- `results.json`, `summary.json`, `report.md`: scores, partitions and transitions.
- `preflight/`, `development/`, `rescue_replay/`: configurations, per-branch scores,
  reset audits, local detailed traces, actions/states/observations and videos.
- Script contract: `scripts/2026-09-11_tabletop_local_planner/README.md` from root.
