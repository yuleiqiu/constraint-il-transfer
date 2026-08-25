# Tabletop MVEE guidance

This branch-local successor begins with the geometry representation gate. It
now implements deterministic obstacle MVEE fitting, conservative gripper and
target ellipsoids, and an opt-in differentiable predicted-clean separation
cost. Closed-loop evaluation is not yet run.

The Tabletop environment adds a fixed `backview` based on the camera selected
in the clean `third_party/vlsa-aegis` reference repository at commit
`57b1aef`:

- the reference position `[-0.4, 1.07, 1.25]` metres is retained;
- the optical axis is aimed at world point `[0.0, 0.0, 0.86]`, the middle of
  this task's start-obstacle-goal corridor at carried-object height;
- the resulting MuJoCo quaternion is
  `[0.1447130, 0.1035081, -0.5724828, -0.8003794]` in `wxyz` order.

The reference SafeLIBERO tabletop scene and this project use robosuite's same
default `TableArena` coordinate frame, so the Backview position transfers
directly. Its reference quaternion is not used: that optical axis is almost
horizontal and gives the GoalTray zero visible pixels in this task. The
retargeted view covers the Can, obstacle, GoalTray, and sampled corridor from
table height through 1.10 m. It is added only by `PickPlaceCanTabletop` and its
evaluation variants, not to the global arena.

Render the fixed RaisedMilk dual-view preflight:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-25_tabletop_mvee_guidance/render_dual_view_preflight.py
```

The figure overlays oracle Can, goal, and obstacle segmentation plus a
projected transport centerline on Agentview and Backview. It then shows both
unprojected obstacle clouds and the single deterministic FPS sample drawn after
fusion. The source policy's existing `task_pointcloud` input remains
single-view and unchanged; the new multi-view provider is opt-in for
deployment-only geometry.

Fit and render the full geometry representation:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-25_tabletop_mvee_guidance/render_ellipsoid_geometry_preflight.py
```

The obstacle MVEE is fitted to every valid point from both views, not only the
512-point policy sample. The final axes are inflated by the measured numerical
constraint violation so all supplied points remain enclosed. Gripper collision
geoms are fitted in the EEF-site frame and therefore transform rigidly with a
predicted delta pose. The Can collision mesh is fitted separately and its
EEF-local MVEE is recorded, so a grasped target can be carried as a second
actor ellipsoid.

The guidance cost uses the center-line support radii of each oriented actor
ellipsoid and the obstacle MVEE. It is a conservative differentiable
support-plane clearance, not an exact Euclidean ellipsoid distance. Gradients
are evaluated only on the executable predicted-clean action slice and may
affect delta position and rotation; the gripper command remains unchanged.

## Fresh collision-failure state bank

The archived 2026-08-14 state bank is intentionally not used as a paired input
here. Its saved model XML predates the task-aligned Backview, and patching that
XML would make static obstacle placement and the current camera model less
trustworthy. Instead, generate fresh states from the current environment and
keep only full baseline collision failures:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-25_tabletop_mvee_guidance/collect_failure_state_bank.py \
  --checkpoint <tabletop-lan-checkpoint> \
  --out-dir outputs/tabletop_mvee_guidance/fresh_failure_bank_seed991 \
  --layouts 12 --target-failures 3 --seed 991 --horizon 600
```

Canonical fresh bank:
`outputs/tabletop_mvee_guidance/fresh_failure_bank_seed991/fresh_initial_state_bank.pkl`.

## Full-rollout ellipsoid counterfactual

Run one fresh failure state from the beginning of the episode with baseline and
ellipsoid guidance sharing the same model XML, dynamic state, policy seed, and
CUDA RNG:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-25_tabletop_mvee_guidance/ellipsoid_counterfactual_rollout.py \
  --checkpoint <tabletop-lan-checkpoint> \
  --state-bank outputs/tabletop_mvee_guidance/fresh_failure_bank_seed991/fresh_initial_state_bank.pkl \
  --scene 0 --bank-seed 991 --policy-seed 991000000 \
  --guidance-scale 45.0 --ellipsoid-margin 0.05 \
  --out-dir <result-dir>
```

The rollout is closed-loop from the initial layout; it is not a frozen
pre-collision chunk audit. Guidance context is rebuilt at every policy chunk
from the current dual-view obstacle cloud and current gripper / grasped-can
geometry. Baseline and guided runs use the same full initial state, so the
controller state is reset between conditions.

## Preliminary closed-loop result

On the three fresh collision failures (seed 991, episodes 0/3/4), `scale=45`,
`margin=1 cm` activated the ellipsoid cost near the first collision and delayed
but did not remove Can-Milk contact. `margin=5 cm` removed physical contact in
all three episodes and kept minimum clearance between 5.97 cm and 7.92 cm, but
all three became collision-free non-completions. The geometry can reason about
the obstacle volume and avoid contact; it still does not preserve task progress
well enough to finish within H600. This is mechanism evidence on the
Tabletop LAN-O3DP checkpoint, not a claim about the clean-image source policy.

## Same-process margin comparison

Across separate Python processes the unguided baseline can drift because the
Tabletop LAN checkpoint consumes a GPU-rendered point cloud and the environment
RNG is not fully captured by the state bank. Margin comparisons therefore run
in one process with one shared baseline:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-25_tabletop_mvee_guidance/ellipsoid_margin_sweep.py \
  --checkpoint <tabletop-lan-checkpoint> \
  --state-bank outputs/tabletop_mvee_guidance/fresh_failure_bank_seed991/fresh_initial_state_bank.pkl \
  --scene 3 --bank-seed 991 --policy-seed 991000003 \
  --guidance-scale 45.0 --margins 0.01 0.03 0.05 \
  --out-dir outputs/tabletop_mvee_guidance/fresh_failure_bank_seed991/inprocess_margin_sweep_ep3
```

On episode 3 the shared baseline collides at step 179. `margin=1 cm` and
`margin=3 cm` both remain collision failures; 3 cm activates slightly earlier
(step 160) and shifts the first contact from 179 to 185 but does not create a
task rescue. `margin=5 cm` activates at step 24, removes contact, and becomes
NCR. The useful transition is therefore between 3 cm and 5 cm, not a smooth
1/3/5 progression.

The representative collision-free stall video (episode 3, `margin=5 cm`) is
`outputs/tabletop_mvee_guidance/fresh_failure_bank_seed991/videos/
guided_margin5_ep3.mp4`. At step 590 the gripper still holds the Can,
`can-to-goal` is 0.47 m, and the robot remains beside the Can start; no
transport segment is ever formed.
