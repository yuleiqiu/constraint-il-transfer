# Guided Denoising Implementation Plan

Status: discussion draft. This document defines the implementation contract for
the first delta-EEF guided-denoising experiment. It does not claim that the
method works until end-to-end rollout results show an improvement.

The general mathematical framework is in `formulation.md`. This document is the
implementation contract for the LAN-O3DP-inspired first experiment.

## 1. Objective

Test whether deployment-time cost guidance can improve multi-object task
success for the existing `delta_eef_pose_action` diffusion policy without
retraining it.

The first experiment isolates inference-side guidance. It keeps the existing
RGB-conditioned policy, observation encoder, training data, and checkpoint
frozen. Point-cloud policy training is a later, independent experiment.

The experiment is successful only if robot-level Task SR improves. Lower
predicted cost or fewer collisions alone is not sufficient.

## 2. Scope

The first implementation should:

- reuse the existing trained delta-EEF checkpoint;
- retain the existing two RGB observations and proprioceptive observations;
- guide the reverse sample using a cost evaluated on the predicted clean
  action;
- reconstruct EEF position trajectories directly from delta-EEF actions;
- use simulator oracle obstacle geometry so perception error is not part of the
  first test;
- support an opt-in deployment cost and a disabled baseline path;
- preserve the existing rollout behavior when guidance is disabled;
- expose enough diagnostics to explain whether guidance changes predicted and
  executed trajectories in the intended direction.

The first implementation should not restore the archived OSC forward model,
action-ranking stack, point-cloud observation pipeline, policy retraining, or
post-hoc action refinement.

## 3. Formulation-to-Code Contract

| Formulation object | Implementation meaning |
|---|---|
| $o$ | Existing RGB and proprioceptive policy observation |
| $e^\star$ | Oracle deployment context: current EEF pose and obstacle geometry |
| $A_t$ | Normalized noisy action sequence inside reverse diffusion |
| $\hat A_0^{(t)}$ | Predicted clean normalized action sequence returned by the scheduler / denoiser |
| $F(o,\hat A_0^{(t)})$ | Unnormalize the executed delta-EEF slice and cumulatively reconstruct its EEF trajectory |
| $C_{\mathrm{LAN}}(F(o,\hat A_0^{(t)}),e^\star)$ | XY unsquared penetration cost on the reconstructed trajectory |
| $g^P_t$ | Per-waypoint physical cost gradient with respect to the reconstructed absolute EEF positions |
| $u^P_t$ | Per-waypoint outward push $-g^P_t$ |
| $u^{\Delta}_t$ | Delta-action update obtained by differencing the pushed absolute waypoints |
| $s_t$ | LAN-O3DP-style scheduler factor $\lambda/\sqrt{\bar\alpha_{t-1}}$ |

The first version does not backpropagate through the RGB encoder or denoising
network. Within each reverse step, it only needs autograd through:

```text
predicted clean normalized action (treated as the current guidance point)
-> action unnormalization
-> executed action slice
-> delta-EEF trajectory reconstruction
-> deployment cost
```

The cost gradient stops at the reconstructed absolute EEF waypoints. The
waypoints are pushed outward and then differenced back into delta actions; the
implementation does not backpropagate the cost through `cumsum` to obtain a
delta-action gradient. The resulting delta update is applied directly to the
scheduler's reverse sample.

This is the operational meaning of "LAN-O3DP-style" in this plan; it is not
classifier guidance through the conditional diffusion network.

## 4. Action and Trajectory Semantics

The policy predicts a normalized sequence with prediction horizon `Tp`. The
environment executes only:

```text
start = observation_horizon - 1
end = start + action_horizon
executed_action = predicted_action[:, start:end]
```

Guidance must evaluate this same slice. With the current defaults, this is
`[:, 1:9]`, not `[:, 0:8]`.

Before trajectory reconstruction, the slice must be converted from normalized
policy coordinates to the raw `delta_eef_pose_action` coordinates using the
checkpoint's action-normalization statistics.

Those statistics currently belong to `RolloutPolicy`, not the inner diffusion
policy. The rollout setup must pass the existing checkpoint statistics into the
guidance context; the guidance implementation must not recompute or hard-code
its own scale and offset.

For the position dimensions:

$$
p_{h+1}=p_h+\Delta p_h,
\qquad p_0=p_{\mathrm{eef,current}}.
$$

Because LAN-O3DP uses absolute Cartesian action positions while this project
uses delta positions, its distance calculation cannot be copied directly onto
the first two action dimensions. The cumulative reconstruction above is
mandatory.

The initial cost is position-only. Guidance updates only the XY delta-position
dimensions in the executed slice. It must not directly update Z, orientation,
gripper, the past action at index 0, or the unexecuted suffix.

## 5. Proposed Code Boundaries

### robomimic fork

Keep checkpoint compatibility and make guidance disabled by default.

- `robomimic/algo/diffusion_policy.py`
  - add the smallest possible hook around each reverse-diffusion step;
  - obtain the current step's predicted clean action and reverse sample;
  - invoke guidance only when a context is present and enabled;
  - leave the existing unguided path unchanged.
- `robomimic/utils/guided_denoising_utils.py`
  - clean-action extraction from scheduler output;
  - differentiable action unnormalization;
  - executed-slice selection;
  - delta-EEF trajectory reconstruction;
  - LAN penetration cost and per-waypoint outward push;
  - pushed-waypoint to delta-action conversion;
  - normalized displacement-vector conversion;
  - timestep-scaled reverse-sample update.
- `tests/test_guided_denoising_utils.py`
  - unit and gradient-flow tests independent of robosuite rollout.

The implementation should not use runtime monkey-patching of policy classes.

### Root project

- `scripts/guided_denoising/`
  - same-state diagnostic;
  - paired rollout evaluation;
  - result aggregation.
- `outputs/guided_denoising/`
  - diagnostics and completed experiment reports.

Project-specific obstacle extraction and evaluation metrics should remain in the
root project rather than becoming generic robomimic library behavior.

## 6. First Deployment Cost

For reconstructed EEF waypoints $P=(p_1,\ldots,p_H)$ and oracle obstacle
footprints $(c_j,r_j)$, use the LAN-O3DP-equivalent XY unsquared penetration
cost:

$$
C_{\mathrm{LAN}}(P)
=
\sum_{h,j}
\max\left(
0,
r_j-\left\|p_h^{xy}-c_j^{xy}\right\|_2
\right).
$$

This produces a near-unit outward update for each active waypoint-obstacle
pair, rather than an update whose magnitude shrinks near the safety boundary as
with a squared hinge. The implementation must use an epsilon-safe norm or an
explicit zero-distance fallback so the gradient remains finite at $p_h=c_j$.

For the first experiment:

- $c_j$ comes from simulator oracle object geometry in world coordinates;
- $r_j$ is the object's oracle XY collision footprint plus a named,
  configurable clearance margin;
- only obstacles not belonging to the task target are included;
- only the executed trajectory prefix is costed;
- perception-derived centers, point clouds, robot-link geometry, swept-segment
  cost, and carried-object geometry are deferred.

This deliberately isolates the denoising mechanism. It does not claim that an
EEF-waypoint cost is a complete collision model.

## 7. Guidance Update

Let $P=(p_1,\ldots,p_H)$ be the absolute EEF trajectory reconstructed from the
unnormalized executed delta-position slice. First compute a physical outward
push independently for each waypoint:

$$
g^P_t
=
\nabla_P
C_{\mathrm{LAN}}
\left(P,e^\star\right),
\qquad
u^P_t=-g^P_t.
$$

Turn this direction into a physical waypoint displacement using the LAN
scheduler factor, then construct pushed waypoints:

$$
d^P_t
=
\frac{\lambda}{\sqrt{\bar\alpha_{t-1}}}u^P_t,
\qquad
P'=P+d^P_t.
$$

Then convert the pushed trajectory back to delta positions:

$$
\Delta p'_1=p'_1-p_0,
\qquad
\Delta p'_h=p'_h-p'_{h-1}\quad(h>1),
$$

where $p_0$ is the current measured EEF position. The physical delta-action
update is:

$$
u^\Delta_t=\Delta P'-\Delta P.
$$

This choice reproduces LAN-O3DP's independent absolute-waypoint push in a
delta-action representation. It deliberately does not use
$\nabla_{\Delta P}C=L^T\nabla_PC$, which would accumulate all future waypoint
gradients onto early deltas and produce a different update geometry.

Convert the physical delta-action displacement to normalized action
coordinates using only the checkpoint normalizer's linear scale:

$$
u^\Delta_{t,\mathrm{norm}}=S_{\mathrm{action}}u^\Delta_t.
$$

Do not call the normalizer's ordinary affine `normalize()` on a vector: its
offset applies to points, not displacement vectors.

After the standard DDIM step produces $\widetilde A_{t-1}$, apply:

$$
A_{t-1}^{xy}
=
\widetilde A_{t-1}^{xy}
+
u^\Delta_{t,\mathrm{norm}}{}^{xy}.
$$

Use the current step's $\hat A_0^{(t)}$ to update the current
$\widetilde A_{t-1}$; do not reproduce the official code's accidental one-step
gradient delay. Here $u^P_t$ is a unit-scale direction and $\lambda$ supplies
its physical displacement magnitude. LAN-O3DP's $\lambda=0.03$ therefore
corresponds roughly to a 3 cm waypoint push before division by
$\sqrt{\bar\alpha_{t-1}}$. It is only a reference value: because action
semantics and normalization differ, it must be exposed as a configuration
value and checked in same-state diagnostics before rollouts.

Keep the scheduler factor explicit rather than hiding it inside the cost. A
predicted-noise / score update is a later comparison, not part of the first
implementation.

Record timestep, active penetration count, physical cost, minimum clearance,
raw physical gradient norm, normalized applied-update norm, and resulting
clean-action cost. Additional gradient normalization or clipping must be a
named option and disabled by default.

## 8. Verification Gates

### Gate A: unit correctness

- delta-EEF reconstruction matches a hand-computed trajectory;
- normalized actions are converted to physical units correctly;
- a physical displacement vector is converted with scale but no offset;
- the cost uses exactly the executed slice;
- waypoint cost gradients remain finite;
- a penetrating waypoint receives an outward update and a safe waypoint gets
  zero update;
- differencing pushed waypoints back to delta actions reconstructs exactly the
  intended pushed trajectory;
- a push at one waypoint does not unintentionally translate every later
  waypoint;
- position-only guidance masks orientation and gripper updates;
- past and unexecuted action indices remain unchanged;
- empty / zero cost produces a zero update.

### Gate B: original-behavior check (unguided parity)

Purpose: prove that adding the optional guidance code does not change the
existing policy when guidance is disabled. This catches accidental changes to
the original inference path before evaluating the new method.

With the same observation and random seed:

- guidance disabled reproduces the existing policy output;
- an enabled zero-cost function reproduces the existing output;
- checkpoint loading and standard rollout remain unchanged.

### Gate C: fixed-state guidance check (same-state diagnostic)

Purpose: hold the robot state, object layout, RGB observation, and initial
diffusion noise fixed, then vary only the guidance scale. This separates a real
guidance effect from ordinary diffusion sampling randomness and reveals whether
the update direction and magnitude are sensible before running full episodes.

For saved obstructed states, compare unguided and guided samples generated from
the same initial diffusion noise:

- predicted deployment cost;
- predicted minimum clearance;
- normalized and physical action displacement;
- number of active waypoint-obstacle penetrations;
- executed EEF trajectory and actual clearance after a short prefix;
- task-progress indicators.

Predicted improvement without consistent executed improvement blocks rollout
scaling.

### Gate D: small matched rollout comparison (paired rollout pilot)

Purpose: test whether a guidance change that looks correct in a fixed state
actually helps complete the robot task. Run a small number of full baseline and
guided episodes from matching initial environment seeds before committing to a
large evaluation.

Use identical environment seeds for unguided and guided conditions. Report:

- Task SR;
- Safe SR, CR, and NCR;
- safe success, success with collision, collision failure, and NCR;
- guidance trigger rate and action-update magnitude;
- trajectory reconstruction error and action clipping.

Only scale to the full evaluation matrix if guidance does not simply trade CR
for NCR or materially reduce Task SR.

## 9. Experiment Sequence

The representation experiments remain separated:

1. **RGB policy + oracle geometry guidance**: current implementation target;
2. **RGB policy + perceived geometry guidance**: only after inference-side
   guidance shows useful signal;
3. **point-cloud-conditioned policy + 3D guidance**: later retraining and
   representation comparison.

This ordering prevents a policy-observation change from confounding the first
test of guided denoising.

## 10. Fixed Defaults and Pre-Rollout Selection

Decisions already made:

- freeze the existing RGB-conditioned delta-EEF policy;
- use oracle simulator obstacle geometry;
- use the LAN-equivalent XY unsquared penetration cost;
- push absolute EEF waypoints and difference them back into delta actions;
- use the LAN-O3DP-style direct DDIM reverse-sample update;
- use the current predicted clean action without the official one-step delay;
- transform physical update vectors with normalizer scale and no offset;
- use each distractor's oracle `horizontal_radius` plus a configurable 2 cm
  clearance for the initial circular XY footprint;
- guide all 10 DDIM inference steps in the first diagnostic;
- retain `action_horizon=8` for the first baseline comparison;
- run the same-state scale sweep at `0`, `0.001`, `0.003`, `0.01`, and the LAN
  reference `0.03`.

No conceptual decision blocks implementation. Before paired rollouts, the
diagnostic must still select:

1. the largest scale that improves predicted clearance without implausible
   waypoint displacement or action clipping;
2. a fixed paired set of collision-producing states and rollout seeds from the
   two- and three-distractor environments. Existing outputs contain aggregate
   results but not reusable simulator-state snapshots, so the diagnostic script
   must capture these states before evaluating the sweep.

The selected scale, states, and seeds must be recorded before the paired pilot.
After code lands, `third_party/robomimic/AGENTS.md` must map this contract to the
concrete implementation files and verification commands.
