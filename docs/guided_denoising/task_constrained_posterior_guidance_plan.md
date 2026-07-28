# Task-Constrained Posterior Guidance Execution Plan

Status: Stage 1 stopped at the preregistered gate, 2026-07-22. The posterior-
gradient Stage 2 was not started. See
`outputs/task_constrained_guidance/stage1_report.md`.

This document preserves the preregistered execution plan for the stopped
candidate. It is not a current handoff: Stage 1 failed its gate, Stage 2 was
never implemented, and the project subsequently moved to targeted
training-side adaptation. The LAN-style mechanism baseline and sample-refine
pilot remain separate negative baselines.

## 1. Objective and Scope

Test whether a frozen single-object delta-EEF Diffusion Policy can recover a
specific multi-object deployment regime through inference-time guidance:

```text
pre-target + target-directed + predicted obstruction
```

The method is not intended to explain non-target-directed or
target-directed-clear failures. Results must be reported by the P0 diagnostic
regime as well as by the four-way episode outcome partition.

No policy retraining is permitted. Simulator-oracle target and obstacle
geometry are permitted during method discovery so perception is not a
confounder.

## 2. Frozen Policy Contract

The checkpoint and its temporal contract remain unchanged during the primary
candidate:

| quantity | value | treatment |
|---|---:|---|
| observation horizon | 2 | frozen; determines observation conditioning |
| prediction horizon | 16 | frozen; trained action-sequence length |
| action horizon | 8 | frozen for the primary candidate |
| executed action slice | `[:, 1:9]` | costed and executed unchanged |
| action representation | 7-D delta EEF full pose | frozen |
| denoising scheduler | 10-step DDIM, epsilon prediction | frozen |

The action horizon is used by the current implementation for inference-time
slicing and queuing, not for the diffusion training loss. Nevertheless,
short-prefix replanning changes the closed-loop state distribution and is not
part of the primary candidate. It may be tested later as a separately named
ablation only if the unchanged eight-step execution is diagnosed as the
bottleneck.

## 3. Stage 1: Eight-Step Feasibility Upper Bound

### Question

Before changing the denoising sampler, determine whether the existing
eight-step executable delta-EEF position slice admits a task-preserving
collision correction at all.

For an unguided action chunk, reconstruct its baseline EEF positions
`P_bar = (p_bar_1, ..., p_bar_8)`. The first upper-bound implementation
optimizes only an XY position offset `D = (d_1, ..., d_8)` and defines
`P_xy = P_bar_xy + D`. Keep Z, orientation, and gripper actions exactly equal
to the unguided chunk. Convert the corrected positions back to executable
delta-position actions by differencing from the measured chunk-start EEF
position.

### Optimization contract

The feasibility solver minimizes deviation from the policy trajectory while
enforcing task and execution constraints:

$$
\begin{aligned}
\min_D\quad
& \sum_h w_h\lVert d_h\rVert_2^2
  + \rho_{\mathrm{smooth}}\sum_{h=2}^{7}
    \lVert d_{h+1}-2d_h+d_{h-1}\rVert_2^2 \\
\text{s.t.}\quad
& \operatorname{clearance}_{\mathrm{swept}}(P) \ge m,\\
& \Delta d_{\mathrm{target}}(P)
  \ge \Delta d_{\mathrm{target}}(P_{\mathrm{bar}})
  - \epsilon_{\mathrm{progress}},\\
& \max_h\lVert d_h\rVert_2 \le \delta_{\mathrm{trust}},\\
& A(P) \in \mathcal A_{\mathrm{env}}.
\end{aligned}
$$

The first implementation uses the same oracle object centers and horizontal
radii as P0, but evaluates continuous XY segment clearance rather than only
waypoint penetration. This is an EEF-path feasibility upper bound, not yet a
claim of whole-robot collision modeling. Parameters are expressed in physical
units and recorded in every artifact.

Contact-geom calibration on the first pre-entry bank found gripper-only
contacts and showed that the EEF point can retain about 5.25 mm clearance at
first contact. Stage 1 therefore uses a 1 cm EEF-path margin as a conservative
gripper proxy. When a chunk begins inside this proxy but before physical
contact, the constraints require non-decreasing clearance followed by recovery
to the full margin within two predicted waypoints; requiring the fixed start
point to satisfy the full margin would be mathematically impossible.

### Offline and rollout checks

1. Unit-test identity correction, waypoint-to-delta round trips, continuous
   segment clearance, target progress, action bounds, and preservation of the
   four non-position action dimensions.
2. Run the solver on saved collision-producing fixed states without changing
   the simulator state. Verify feasibility and predicted cost reduction.
3. Run closed-loop matched rollouts on at least 20 layouts whose unguided
   critical chunk is `target_directed_obstructed`. Execute all eight corrected
   actions before replanning, then return control to the frozen policy after
   first Can contact.

### Advance gate

Stage 1 advances only if the 20-layout mechanism pilot produces all of:

- at least four additional episodes that contact Can;
- at least two additional full task successes;
- no action clipping;
- no equal-size exchange of collision failures for collision-free
  non-completions;
- recorded evidence that the accepted corrections satisfy progress and trust
  constraints.

If the upper bound fails, do not implement another diffusion guidance scale
sweep. Diagnose whether failure is caused by the EEF collision proxy, the
eight-step feasible set, or the inability to preserve task progress. Stop the
current inference-only geometry candidate unless one of those assumptions can
be falsified with a bounded diagnostic.

## 4. Stage 2: Posterior Gradient Consistent with the Formulation

Stage 2 begins only after Stage 1 passes. Implement a separately registered
policy variant; do not overwrite the completed LAN baseline.

For noisy action `A_t`, compute the epsilon prediction and an unclipped clean
estimate:

$$
\hat A_0^{(t)}
=
\frac{A_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(A_t,o,t)}
     {\sqrt{\bar\alpha_t}}.
$$

Evaluate an augmented-Lagrangian form of the Stage 1 constraints on the
reconstructed executable slice, and compute the actual noisy-action gradient:

$$
g_t
=
\nabla_{A_t}
\mathcal L_{\mathrm{constrained}}
\left(F(o,\hat A_0^{(t)}),e^\star,P_{\mathrm{bar}}\right).
$$

For epsilon prediction, inject the cost score through a scheduler-consistent
epsilon update before the DDIM reverse step. The observation encoder and policy
weights remain frozen; autograd is required only for the Jacobian-vector path
from the cost to `A_t`.

Use a same-noise unguided branch as the behavioral anchor. At every reverse
step, restore orientation, gripper, past-action, and any non-editable action
coordinates from the unguided latent. An inactive gate and a zero-cost function
must reproduce the original policy exactly.

## 5. Stage 3: Verification and Rollout Gates

### Fixed-state gate

- finite-difference agreement for the gradient with respect to `A_t`;
- lower swept-clearance violation in at least 9 of 10 obstruction states;
- progress and trust constraints satisfied in all accepted states;
- exact preservation of orientation, gripper, and masked indices;
- no NaN, action clipping, or silent zero-gradient behavior;
- short-prefix execution changes in the predicted direction.

### Method-development rollout gate

1. A selected `target_directed_obstructed` mechanism pilot.
2. An unselected matched pilot on `PickPlaceBreadCerealCan`.
3. The same frozen method on `PickPlaceBreadCerealMilkCan`.
4. Single-object preservation on `PickPlaceCan`.

Report Can-contact rate, Task SR, Safe SR, CR, NCR, the four-way outcome
partition, paired outcome transitions, trigger rate, constraint acceptance,
and inference overhead. The method advances only if paired Task SR improves in
both hard clutter levels without an equal exchange into NCR and single-object
behavior is preserved.

## 6. Optional Horizon Ablation

Short-prefix replanning is considered only if Stage 1 or Stage 3 shows that a
valid early correction is undone during the later portion of the unchanged
eight-action queue.

The ablation must not alter the checkpoint's prediction or guidance horizons:
the policy still predicts 16 actions and returns/costs the original eight-step
slice, but an inference wrapper executes only the first two actions, discards
the remaining six, and replans. Before inclusion, it must pass an unguided
single-object preservation pilot. A benefit from this ablation must be reported
separately from the benefit of task-constrained posterior guidance.

## 7. Decision Tree

```text
Stage 1 fails
  -> current cost / geometry / eight-step feasible set cannot supply a useful
     correction; stop posterior-guidance implementation.

Stage 1 passes, Stage 2 fixed-state gate fails
  -> sampler or gradient injection is the bottleneck; permit one bounded
     implementation correction, not a rollout-scale hyperparameter sweep.

Fixed-state gate passes, closed-loop gate fails
  -> the short-horizon geometric objective is not predictive of task
     completion; continuing requires a learned value / world model or a
     training-side adaptation and therefore a revised project claim.

Closed-loop gate passes
  -> freeze the method and proceed to statistically sized evaluation and
     independent source-policy seeds under `research_plan.md`.
```

## 8. Initial Implementation Locations

Root-project experiment tooling belongs under:

```text
scripts/2026-07-22_task_constrained_guidance/
outputs/task_constrained_guidance/
```

The Stage 1 optimizer and tests remain in the root project because they are
experiment-specific. Only the Stage 2 generic sampler and tensor utilities
belong in the robomimic fork. Every change in the fork must preserve checkpoint
loading and unguided parity.
