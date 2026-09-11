# Tabletop route / handoff diagnostic

Main matrix complete, reviewed 2026-09-08: three selected layouts, five
conditions, three identical-seed repetitions = 45 executed branches. The
conditional near-goal handoff probe for layout 27 is reported separately.

## Main result

Each cell below has the same outcome in all three repetitions. These repeats
audit reproducibility; the sampling unit is the layout, not the individual run.

| Layout | Source | Original point guidance | 8-action help then source | Longer bypass then source | Full teacher |
|---|---|---|---|---|---|
| 43 | collision failure | collision-free non-completion | safe success | safe success | safe success |
| 49 | collision failure | collision failure | success with collision | safe success | safe success |
| 27 | collision failure | safe success | collision failure | collision-free non-completion | safe success |

| Condition | Task SR | Safe SR | CR | NCR | Four-way state counts: safe / success+contact / collision failure / NCR | Task F→S / S→F | Safe F→S / S→F |
|---|---:|---:|---:|---:|---|---|---|
| source | 0/3 | 0/3 | 3/3 | 0/3 | 0 / 0 / 3 / 0 | — | — |
| point guidance | 1/3 | 1/3 | 1/3 | 1/3 | 1 / 0 / 1 / 1 | 1 / 0 | 1 / 0 |
| short help | 2/3 | 1/3 | 2/3 | 0/3 | 1 / 1 / 1 / 0 | 2 / 0 | 1 / 0 |
| long help | 2/3 | 2/3 | 0/3 | 1/3 | 2 / 0 / 0 / 1 | 2 / 0 | 2 / 0 |
| teacher | 3/3 | 3/3 | 0/3 | 0/3 | 3 / 0 / 0 / 0 | 3 / 0 | 3 / 0 |

Baseline failure is an eligibility condition, so zero paired regressions here
cannot establish task preservation on initially successful states. These tiny,
selected-state rates do not estimate general deployment performance and do not
support significance claims.

## What the cases establish

- **43:** original point guidance prevents completion, while eight teacher
  actions followed by source finish safely. That short prefix moves Can about
  5.25 mm laterally. A longer bypass also works, so persistence is not required
  by this tested case. Completion steps: short 351, long 433, teacher 393.
- **49:** eight teacher actions (about 4.43 mm lateral Can displacement) still
  lead to collision, despite eventual task completion. The longer prefix leads
  to safe completion at step 513. The source after short help moves onto the
  opposite side from the teacher's chosen bypass. This motivates studying
  commitment to a route, but does not establish a universal duration requirement.
- **27:** original guidance itself produces a repeatable safe completion at
  step 455. Longer teacher assistance followed by source avoids collision but
  does not satisfy the complete task. Thus neither "guidance never works" nor
  "longer help is always better" is supported.

Layout 27's long-help failure is specifically a terminal retreat failure: at
H600 the Can is inside the synchronized goal region at z=0.8468 m, grasp is
false, but EEF-to-Can distance is only 0.0194 m. The inherited success criterion
requires retreat beyond approximately 0.0424 m (1-tanh(10*d) < 0.6). Do not
describe this as failure to reach the tray or complete absence of task recovery.
Source-alone and short-help branches also finish with the Can near the tray
and insufficient retreat, having contacted the obstacle earlier.

All complete teachers are safe, establishing executable references. All long
handoffs retain the grasp and place the entire gripper/Can past the obstacle's
projection plane, with measured margins 42.6 / 43.0 / 44.0 mm for 43 / 49 / 27.
This plane check is a handoff-location diagnostic, not a safety guarantee.

## Reproducibility and selection

Every main branch restores the actual executed source prefix with zero maximum
difference in saved simulator state and complete observation history, including
point cloud. Within each layout/condition, all three recorded full trajectories
(EEF, Can, contact labels, grasp and task labels) match exactly. First source
chunks match exactly. The first repetition records video during execution;
later repetitions omit video and reproduce the recorded traces. Source
continuations reproduce the baseline screening outcomes.

The prefix is replayed through the controller; mid-episode positions alone are
not used as a complete controller snapshot. Teacher actions update actual
observation history. Sampling directly through the existing chunk generator
avoids a stale action queue at handoff. The H600 budget includes the original
prefix: intervention starts at steps 112 / 224 / 168 for 43 / 49 / 27.

Screen order was the seven archived baseline collision failures: 8, 17, 25,
37, 43, 48, 49. Fresh synchronized evaluation excludes 8/25/37 (success with
contact), 17 (collision-free non-completion), and 48 (safe success). This yields
43 and 49. Before extension outcomes, the next candidates were declared as the
remaining archived contact cases 18, 27, 28. Case 18 succeeds with contact and
is excluded; 27 is a fresh collision failure and is selected. Screening stops
at three eligible layouts; 28 is not evaluated. All nine screens are retained.

Each layout's conditions/repeats execute in one process. The independent
extension uses GPU 1 while the first two layouts use GPU 0; no cross-GPU
determinism is assumed. Raw metrics and replay audits live in
`replicated/analysis.json` and `extension/analysis.json`.

## Interpretation limits and next decision

### Conditional near-goal probe (separate from the 45-run matrix)

Because long help fails on 27 while its complete teacher succeeds, continue the
same teacher route to above the tray before handing off. All three additional
runs are collision-free non-completions at H600. At termination Can is in the
goal region at z=0.8458 m, grasp is false, and EEF-to-Can distance is only
0.0177 m: retreat is again incomplete. The full recorded trajectories reproduce
exactly across the three runs, with exact boundary restoration. Thus handing
off at this additional later point does not solve this particular finishing
failure. It does not prove that no successful handoff window exists elsewhere.
See `extension/ep27_near_goal.mp4` and the separately labelled rows in
`extension/analysis.json`. Total diagnostic branches: 45 main + 3 conditional;
the initial five-branch pilot is retained separately and not pooled.

This is an oracle-route existence diagnostic, not a learned deployment method.
The teacher uses simulator geometry, fixed route waypoints and a known task
phase. Short versus long assistance changes both duration and displacement.
Diffusion RNG is aligned to source invocations, not absolute time when teacher
actions replace source calls. Hold/replanning or time-aligned-noise controls
are needed before attributing a short-help benefit specifically to its direction.

The observations justify studying how a useful bypass is selected and retained,
and how task completion behaves after handoff. They do not yet identify an
online predictor, prove local-guidance insufficiency over a method class, or
justify training a specific corrective option. No further training is selected.

## Separate confirmed evaluation defect

Saved XML/state restoration can leave Python-side `goal_xy` / `obstacle_xy`
inconsistent with restored static bodies. Six independent reset probes have
identical saved dynamic states but goal-coordinate errors of 19.9--46.2 mm.
Synchronizing goal coordinates changes 16/54 predefined synthetic boundary
membership labels. See `reset_metadata_audit.json` and `reset_goal_regions.png`.

The present diagnostic synchronizes these coordinates. Historical wide-support
Task/Safe/NCR scores and checkpoint-screen scores need a scoring audit; their
corrected rates have not been measured. This defect does not explain every
trajectory/observation drift, invalidate all earlier environments, or prove
that the original guidance method is effective generally.
