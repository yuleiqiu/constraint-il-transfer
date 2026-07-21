# LAN-O3DP-Style Guided-Denoising Mechanism Baseline

Status: completed mechanism-baseline record, 2026-07-15. The implementation
passed unit, unguided-parity, and fixed-state checks. Its 20-pair rollout pilot
changed task outcomes but produced inconsistent safety effects, so the tested
setting was not scaled to the full matrix. See `research_plan.md` for the next
research and evaluation gates.

The general mathematical framework is in `formulation.md`. This document is the
technical record for the LAN-O3DP-inspired first experiment, not the
project's final method definition.

## 1. Question Tested

The experiment tested whether deployment-time cost guidance could improve
multi-object task success for the existing `delta_eef_pose_action` diffusion
policy without retraining it.

The experiment isolated inference-side guidance. It kept the existing
RGB-conditioned policy, observation encoder, training data, and checkpoint
frozen.

The acceptance criterion was robot-level Task SR improvement without merely
moving collision failures into collision-free non-completion. Lower predicted
cost or fewer collisions alone was not sufficient.

## 2. Scope

The implemented baseline:

- reused the existing trained delta-EEF checkpoint;
- retained the existing two RGB observations and proprioceptive observations;
- guided the reverse sample using a cost evaluated on the predicted clean
  action;
- reconstructed EEF position trajectories directly from delta-EEF actions;
- used simulator oracle obstacle geometry so perception error was not part of the
  first test;
- supported an opt-in deployment cost and a disabled baseline path;
- preserved the existing rollout behavior when guidance was disabled;
- exposed enough diagnostics to determine whether guidance changed predicted and
  executed trajectories in the intended direction.

The baseline did not restore the archived OSC forward model,
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

The baseline does not backpropagate through the RGB encoder or denoising
network. Within each reverse step, autograd covers only:

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

This is the operational meaning of "LAN-O3DP-style" in this baseline; it is not
classifier guidance through the conditional diffusion network.

## 4. Action and Trajectory Semantics

The policy predicts a normalized sequence with prediction horizon `Tp`. The
environment executes only:

```text
start = observation_horizon - 1
end = start + action_horizon
executed_action = predicted_action[:, start:end]
```

Guidance evaluates this same slice. With the implemented defaults, this is
`[:, 1:9]`, not `[:, 0:8]`.

Before trajectory reconstruction, the slice is converted from normalized
policy coordinates to the raw `delta_eef_pose_action` coordinates using the
checkpoint's action-normalization statistics.

Those statistics belong to `RolloutPolicy`, not the inner diffusion policy.
The rollout setup passes the checkpoint statistics into the guidance context
rather than recomputing or hard-coding scale and offset.

For the position dimensions:

$$
p_{h+1}=p_h+\Delta p_h,
\qquad p_0=p_{\mathrm{eef,current}}.
$$

Because LAN-O3DP uses absolute Cartesian action positions while this project
uses delta positions, its distance calculation cannot be copied directly onto
the first two action dimensions. The cumulative reconstruction above is
mandatory.

The cost is position-only. Guidance updates only the XY delta-position
dimensions in the executed slice. It does not directly update Z, orientation,
gripper, the past action at index 0, or the unexecuted suffix.

## 5. Implementation Boundaries

### robomimic fork

Checkpoint compatibility is preserved and guidance is disabled by default.

- `robomimic/algo/diffusion_policy.py`
  - exposes an identity hook around each reverse-diffusion step;
  - leaves the existing unguided path unchanged.
- `robomimic/algo/guided_diffusion_policy.py`
  - registers the opt-in guided policy variant;
  - obtains the current step's predicted clean action and reverse sample;
  - invokes guidance only when a context is present and enabled.
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

The implementation does not use runtime monkey-patching of policy classes.

### Root project

- `scripts/guided_denoising/`
  - same-state diagnostic;
  - paired rollout evaluation;
  - result aggregation.
- `outputs/guided_denoising/`
  - diagnostics and completed experiment reports.

Project-specific obstacle extraction and evaluation metrics remain in the
root project rather than becoming generic robomimic library behavior.

## 6. Implemented Deployment Cost

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
with a squared hinge. The implementation uses an epsilon-safe norm so the
gradient remains finite at $p_h=c_j$.

For this experiment:

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
$\sqrt{\bar\alpha_{t-1}}$. Because action semantics and normalization differ,
the scale is configurable and was selected through same-state diagnostics
rather than copied directly.

The scheduler factor remains explicit rather than being hidden inside the
cost. A predicted-noise or score update was not tested.

The diagnostics record timestep, active penetration count, physical cost,
minimum clearance, raw physical gradient norm, normalized applied-update norm,
and resulting clean-action cost. Additional gradient normalization and
clipping are disabled by default.

## 8. Verification Results

### Gate A: unit correctness

Result: passed. The independent unit checks cover:

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

Result: passed. With the same observation and random seed:

- guidance disabled reproduces the existing policy output;
- an enabled zero-cost function reproduces the existing output;
- checkpoint loading and standard rollout remain unchanged.

### Gate C: fixed-state guidance check (same-state diagnostic)

Purpose: hold the robot state, object layout, RGB observation, and initial
diffusion noise fixed, then vary only the guidance scale. This separates a real
guidance effect from ordinary diffusion sampling randomness and reveals whether
the update direction and magnitude are sensible before running full episodes.

Ten saved collision-producing states were evaluated with common diffusion
noise. The diagnostic compared:

- predicted deployment cost;
- predicted minimum clearance;
- normalized and physical action displacement;
- number of active waypoint-obstacle penetrations;
- executed EEF trajectory and actual clearance after a short prefix;
- task-progress indicators.

Every nonzero scale improved predicted and short-prefix executed clearance in
all ten states without action clipping. Scale `0.01` was selected as the
largest plausible setting: its maximum waypoint push was 3.43 cm. Scale `0.03`
was rejected because its maximum push reached 10.28 cm and reconstruction error
increased.

### Gate D: small matched rollout comparison (paired rollout pilot)

Purpose: test whether a guidance change that looks correct in a fixed state
actually helps complete the robot task.

The pilot used identical environment and policy seeds for unguided and guided
conditions and reported:

- Task SR;
- Safe SR, CR, and NCR;
- safe success, success with collision, collision failure, and NCR;
- guidance trigger rate and action-update magnitude;
- trajectory reconstruction error and action clipping.

Result: did not pass the scaling gate. Across 20 matched pairs, Task SR rose by
0.10 in each hard environment, but CR fell by 0.20 in the two-distractor case
and rose by 0.20 in the three-distractor case. Safe SR also fell in the latter,
and two guided successes still contained distractor contact. The result shows
that the mechanism changes closed-loop outcomes, but it does not establish a
consistent method benefit. See
`../../outputs/guided_denoising/paired_pilot/report.md`.

## 9. Scope Boundaries

The experiment deliberately kept representation changes separate:

1. **RGB policy + oracle geometry guidance**: implemented here;
2. **RGB policy + perceived geometry guidance**: not tested;
3. **point-cloud-conditioned policy + 3D guidance**: not tested.

This separation prevents a policy-observation change from confounding the
mechanism test. It is not a commitment to pursue the two untested variants.

## 10. Fixed Settings and Selection Outcome

The experiment used:

- a frozen RGB-conditioned delta-EEF policy;
- oracle simulator obstacle geometry;
- the LAN-equivalent XY unsquared penetration cost;
- absolute EEF waypoint pushes differenced back into delta actions;
- the LAN-O3DP-style direct DDIM reverse-sample update;
- the current predicted clean action without the official one-step delay;
- physical update vectors transformed with normalizer scale and no offset;
- each distractor's oracle `horizontal_radius` plus a configurable 2 cm
  clearance for the initial circular XY footprint;
- guidance on all 10 DDIM inference steps in the first diagnostic;
- `action_horizon=8` for the first baseline comparison;
- a same-state scale sweep at `0`, `0.001`, `0.003`, `0.01`, and the LAN
  reference `0.03`.

The fixed-state sweep selected scale `0.01`; `0.03` was rejected as
implausibly large. The paired pilot used five captured collision-producing
states from each of the two hardest environments and matching rollout seeds.
Concrete implementation files and verification commands are mapped in
`third_party/robomimic/AGENTS.md`.
