# Semi-manual waypoint teacher

This bounded prototype tests whether a simple oracle Cartesian route can leave
the frozen policy's local action support, avoid the distractors, and hand
control back to the reliable delta-EEF Diffusion Policy.

The current route is intentionally simple:

1. lift the EEF vertically above the tallest active object;
2. move horizontally above the Can;
3. descend to a configurable handoff height above the Can;
4. resume the frozen policy with ordinary eight-action replanning.

The route preserves the current EEF orientation, keeps the gripper open, and
uses the existing world-frame delta `OSC_POSE` interface. It is an oracle
teacher feasibility check, not a proposed policy method.

The optional `grasp_handoff` route continues from the overhead waypoint to the
median demonstrated Can grasp offset, closes the gripper, lifts the Can, and
then hands control back. A relative world-Z yaw can be applied above the Can to
test whole-gripper clearance:

Run a small headless pilot from the repository root:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-07-27_semi_manual_waypoint_teacher/evaluate_waypoint_teacher.py \
  --state-limit 8 \
  --route grasp_handoff \
  --search-yaw \
  --out /tmp/semi_manual_waypoint_teacher.json
```

Use `--relative-yaw-deg 45` or `90` to test an alternate vertical grasp
orientation. `--restore-yaw-before-handoff` is experimental; large per-step
rotation commands caused substantial position tracking drift in the first
bounded test.

With `--search-yaw`, each candidate grasp is executed from an independent
reset of the exact saved simulator state. A candidate passes when it contacts
Can, lifts Can by at least 5 cm, and has no distractor contact during the
teacher route. Search stops after testing both signs of the smallest yaw
magnitude with a passing candidate. The selected route is then replayed from
the saved state and handed back to the frozen policy.

The default state inputs are the three banks that make up the 20 strict
one-chunk-lead collision states from the task-constrained guidance diagnostic.

## Exploratory result

On the first eight saved states, the zero-yaw `grasp_handoff` route changed
baseline / teacher-then-policy Task success from `1/8` to `4/8` and obstacle
collision from `8/8` to `5/8`, with three failure-to-success transitions and
no success-to-failure transition in that run. All four states that failed to
contact Can collided with the tall Cereal during the final grasp descent.

On the first three states, a 90-degree yaw made all three teacher grasps
collision-free and all three moved Can, but the frozen policy failed placement
from the unfamiliar rotated grasp. A 45-degree yaw retained two full successes
and made all three grasps collision-free. These are small mechanism results,
not an evaluation claim. They show that semi-manual waypoint collection is
viable, while a reliable teacher needs a whole-gripper grasp-orientation rule
and a task-compatible handoff pose.

The later discrete-yaw search used one unchanged configuration over all 20
strict saved states. It selected a safe lifted grasp in 18/20 states. Baseline
versus teacher-then-policy counts were:

| metric | baseline | yaw-search teacher + policy |
|---|---:|---:|
| Task success | 1/20 | 18/20 |
| Can contact | 4/20 | 19/20 |
| distractor collision | 20/20 | 2/20 |

There were 17 failure-to-success transitions and no success-to-failure
transition in these deterministic-state runs. Fifteen states retained zero
yaw; the other selected yaws were `+20` twice and `-20`, `-30`, and `+45`
once each. One of the two states without a safe candidate still completed
with Cereal contact; the other never contacted Can. This is a teacher upper
bound on a deliberately selected collision-state bank, not a deployment
policy evaluation.
