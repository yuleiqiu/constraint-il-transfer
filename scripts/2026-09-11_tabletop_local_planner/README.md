# MVEE local planner: implementation gate

User authorized implementation2026-09-11. Reuses the existing fork's MVEE
fitting, with a root-local path planner; does not resume denoising margin sweeps.
No source policy training or dependency fork edits.

## Question and gate

Can an explicit persistent bypass create safe completion while preserving the
frozen source's task behavior? First engineering checks: layout0 (safe source),
25 (source success with contact),28 (difficult guidance failure). These are
previously examined development cases, not a performance/generalization test.
Two exact repeats of0 baseline/planner test replay and unintended intervention.
25/28 each baseline/planner test trigger, execution and source handoff.
Every executed branch has its own MP4 and saved actions, observations and traces.
Do not expand to the full bank unless this engineering set gains at least one
safe success over source with no lost task successes, and execution checks pass.

## Method and assumptions

- Frozen wide-support epoch50, original seed828 state bank, H600, absolute world
  EEF XYZ + world axis-angle + gripper, OSC_POSE. No delta accumulation.
- Baseline uses its unchanged sampled actions. Source sampled once per chunk in
  all conditions. Planner uses no random sampling and no additional rendering.
- New inclusive stopping: original strict success OR in physical goal XY/height,
  no gripper collision-geom contact for11 consecutive20Hz samples (0.5s).
  Contact counts only through first success. Historical other scores stay separate.
- Oracle compiled obstacle AND support geometry, gripper collision geoms and Can
  mesh are fitted with existing MVEE routines. Padding2mm; planner margin10mm.
  This is a geometry-assisted prototype, not a demonstrated pointcloud method.
- Fit obstacle once (static scene). Fit target in body frame once, then update its
  EEF-local attachment from observed pose each chunk. Closed-grasp gripper fit
  is cached; jaw articulation and grasp slip are not predicted.
- Start only while grasped and lifted, and when the source chunk cannot be
  certified clear. Negative conservative clearance is not proof of collision.
- Build visibility graph over current pose, an obstacle-exit waypoint along the
  known goal direction, and four corners of an inflated XY rectangle. Find
  shortest certified route. Restrict EEF XY inside table footprint minus25mm;
  this is not full-arm IK, self-collision or task-fixture checking.
- Check every graph edge and output chunk with interpolated 3-D poses and a
  conservative rigid-motion bound between samples. Only modeled ellipsoids are
  certified; controller tracking and real physics still require rollout checks.
- Latch the route, entry height/orientation, and closing gripper. Target increments
  at most6mm/action; advance a route corner after observed EEF gets within8mm.
  Refresh current target attachment and recheck remaining immediate motion.
- Handoff after route completion only if the next source chunk is certified
  clear and its mean goal-direction displacement is at least-5mm. This short
  gate does not prove source recoverability or future success.
- Infeasible route: hold and log explicitly. Invalidated route: hold and log.
  No silent risky fallback, new direction, or margin sweep. Lost grasp returns
  to source and disables further planner interventions for that episode.

## Run

```bash
uv run pytest -q scripts/2026-09-11_tabletop_local_planner/test_geometry.py
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python \
  scripts/2026-09-11_tabletop_local_planner/run.py \
  --out outputs/tabletop_local_planner_2026-09-11/preflight \
  --episodes 0 --repeats 2
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 uv run python \
  scripts/2026-09-11_tabletop_local_planner/run.py \
  --out outputs/tabletop_local_planner_2026-09-11/development \
  --episodes 25 28
```

## Implementation review

Eleven geometry/controller tests pass, including between-waypoint collision,
rotating carried offsets, existing MVEE clearance parity, infeasible routes,
exact source passthrough, latched transport channels and guarded handoff.
The three primary engineering layouts all finish safely with the planner,
versus one safe source completion; this is not a population estimate.
A separate `rescue_replay` run repeats25/28 on GPU1 and is excluded from counts.
`analyze.py --out <output-root>` checks saved/executed action equality, baseline
archived prefixes, rescue replay state/action/trace equality, and all MP4 frames.
Executed source snapshots are archived before formatting; AST equivalence
confirms formatting has not changed the rollout implementation.
Full50-layout evaluation has not been launched in this implementation stage.

The optional `--conditions guided` branch runs the archived original point
guidance setting under inclusive stopping. It was not part of this implementation
gate. `source_chunks` means the sampled proposal before local planning; for the
optional guided branch it is already guided, not a same-noise unguided reference.
