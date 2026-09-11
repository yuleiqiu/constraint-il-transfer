# Non-XY output preservation: bounded development result

Complete2026-09-10, reviewed2026-09-11. Thirty development rollouts on the
preselected layouts0,5,10,15,20,25,30,35,40,45; four separate pilot rollouts on28.
Every execution has its own MP4; `video_index.md` links them for VS Code.

## Development result

All conditions use released placement with0.5s residence and H600. Release is
the disclosed conservative absence of contact with any gripper collision geom;
goal membership uses the original physical XY/height bounds. The old retreat
success criterion is logged but never terminates execution.

| Condition | Task SR | Safe SR | CR | NCR | Safe / success+contact / collision failure / NCR |
|---|---:|---:|---:|---:|---|
| Source |10/10|9/10|1/10|0/10|9 / 1 / 0 / 0|
| Original guidance |6/10|6/10|1/10|3/10|6 / 0 / 1 / 3|
| GuidedXY + source non-XY |9/10|9/10|1/10|0/10|9 / 0 / 1 / 0|

Protected versus original guidance: four task/safe rescues (0,10,40,45), one
task/safe regression (25), net+3/10. Versus source: zero rescues and one task
regression (25); safe outcomes are identical in all ten layouts, with zero
safe rescues or regressions. Unchanged CR does not mean identical contact events.

The treatment reduces some damage from the original guidance on this set but
does not produce a safety gain over the source. It fails the predeclared
expansion gate because Task SR is below the source, despite passing the
relative-to-guidance improvement threshold. No expanded validation or training
was launched. These ten previously examined development layouts are not a
held-out generalization test; their source100% Task SR is not a population rate.

## What this ablation establishes

At every active chunk, source and guided candidates use identical observation
and sampling RNG. ExecutedXY exactly equals guidedXY; executedZ, axis-angle and
gripper exactly equal the source reference, with maximum non-XY error0.
The cost, instantaneous height gate, geometry and parameters were held fixed.
Same-input channel preservation is not preservation of the source's entire
closed-loop behavior: the guidedXY changes subsequent observations, so later
source references can change. The ablation groups Z, rotation and gripper
preservation together and cannot isolate Z as the sole cause of any rescue.

## Layout28 pilot

Both original and protected fail with collision in both repeats. Original
first contact is256; protected first contact is180. The protected Can remains
near z0.993m through the early transport interval but later loses grasp after
contact and drops. At200 the grasp predicate is false while the commanded
gripper remains closing; this is not evidence of a premature commanded release.
Protection of reference non-XY outputs did not solve this layout.

Both pilot conditions exactly repeat full recorded states/actions/observation
hashes. The original-guidance pilot also reproduces all600 archived original
guided actions exactly. Initial observations match across all development
condition triples. These checks do not claim repeated outcome coverage across
all ten development layouts or cross-GPU determinism.

## Artifacts

- `summary.json`, `results.jsonl`: development metrics and paired transitions.
- `video_index.md`:30 development and4 separate pilot MP4 links, no HTML.
- `video_integrity.json`: decoding/frame-count audit for all34 videos.
- `dev_gpu*/epXX/CONDITION/`: per-step score/contact trace, physical states and
  actions, chunk observations/clouds, paired candidate/output arrays and reset
  audit. Source arrays for original guidance are NaN where not sampled.
- `pilot/gate_pass.json`: repeated-execution and archived-action checks.
- Script contract: `scripts/2026-09-10_tabletop_xy_preserve/README.md` from root.

The fixed-bank scoring rescore and original50-layout strict results remain
separate. Do not mix their denominators or criteria with this development set.
