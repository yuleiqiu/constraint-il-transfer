# Research Log

## Maintenance Rules

- **Format**: reverse chronological, newest on top. Each day is `## YYYY-MM-DD`. Each topic within a day uses `###`.
- **Content**: discussions, findings, decisions with rationale, comparisons with competing works, dead ends.
- **Do NOT write**: raw experiment commands, data directly readable from `outputs/`.
- **When to update**: immediately after each discussion reaches a conclusion. Do not backfill.
- **Marking**: use `[UNVERIFIED]` for hypotheses not yet confirmed.

## Current Unverified Hypotheses

- [UNVERIFIED] Can point-cloud obstacle geometry replace oracle obstacle geometry once action-chunk evaluation is reliable?

---

## 2026-07-14

### Delta EEF multi-environment evaluation separates obstruction from non-completion

Evaluated the best clean-image delta EEF checkpoint over all four PickPlace
environments using three evaluation seeds and 50 episodes per environment and
seed. Standard task success decreases sharply as distractors are added, while
non-target collision rate rises to roughly two thirds on the two hardest
environments.

Trajectory and contact analysis confirms that pre-target physical obstruction
is the dominant hard-environment failure mode: most failures collide before
the first target contact, and most never contact Can. This does not imply that
collision-any is equivalent to failure. The data also contains successful
episodes with incidental distractor contact and collision-free failures where
Can is lifted or transported but not successfully placed.

Decision: future obstacle-aware comparisons should report standard Task SR and
the mutually exclusive Safe SR / CR / NCR metrics. For diagnosis, preserve the
more informative four-way partition:

```text
safe success
success with collision
collision failure
collision-free non-completion (NCR)
```

CR reduction without Task SR improvement can merely transfer probability mass
from collision failure to NCR. Geometry-aware ranking should target pre-grasp
obstruction, while grasp / placement failures remain a separate task-phase
problem.

The delta EEF action interface remains healthy across this evaluation: no
actions were clipped, and reconstructed policy chunks track executed EEF pose
within sub-centimeter mean position error. The old OSC action-to-trajectory
mapping problem has not returned.

This result narrows the earlier historical claim that all failures are
collision-type: that observation was specific to the older policy and
diagnostic setup and does not generalize to the current clean-image delta EEF
policy.

Relevant files:

```text
outputs/eef_pose_osc_policy/multienv_eval_report.md
scripts/eef_pose_osc_policy/eval_delta_eef_multienv.py
scripts/eef_pose_osc_policy/analyze_delta_eef_eval.py
scripts/eef_pose_osc_policy/plot_delta_eef_eval_cases.py
```

## 2026-07-08

### Delta EEF pose replaces the old OSC-action forward-model route

Trained clean-image diffusion policies on executable EEF-pose OSC actions.
The delta EEF policy is the preferred interface:

```text
delta_eef_pose_action = [
  next_obs/robot0_eef_pos - obs/robot0_eef_pos,
  axis_angle(R_next_site @ R_obs_site.T),
  actions[:, 6],
]
```

Controller:

```text
OSC_POSE
input_type = delta
input_ref_frame = world
kp = 500
controller_goal_update_mode = desired
```

The best delta policy reached 0.98 PickPlaceCan rollout success during
training eval. A dedicated diagnostic confirmed that policy action chunks can
be reconstructed into the executed EEF pose trajectory directly:

| rollout | horizon | chunks | mean pos err | max pos err | mean ori err | max ori err |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 334 | 42 | 0.131 cm | 0.541 cm | 0.253 deg | 1.019 deg |

This resolves the action-to-trajectory problem for the active code path:
guidance or ranking can operate directly on the policy-predicted EEF pose
trajectory without a learned OSC forward model.

The absolute EEF policy remains a useful comparison baseline, but was weaker
in training eval: best success 0.82, final success 0.72.

Decision: archive the old OSC-action forward-model / obstacle-guidance /
geometry-only ranking implementation code. Keep result documents for audit,
but do not keep the old implementation in the active repository. Future
guidance, if needed, should be implemented against `delta_eef_pose_action`
trajectory reconstruction rather than resurrecting the original OSC-action
guidance stack.

Relevant files:

```text
outputs/eef_pose_osc_policy/README.md
scripts/eef_pose_osc_policy/diagnose_delta_eef_policy_traj.py
docs/eef_pose_osc_policy_training.md
docs/forward_model_guidance_next_steps.md
docs/action_chunk_ranking_report.md
```

## 2026-07-06

### Route B correction: full-pose EEF can control built-in OSC absolute

Re-ran EEF-pose replay after fixing the action interface. The executable action
is:

```text
[
  next_obs/robot0_eef_pos,
  quat2axisangle(next_obs/robot0_eef_quat_site),  # robosuite xyzw order
  actions[:, 6],
]
```

Environment/controller:

```text
OSC_POSE
input_type = absolute
input_ref_frame = world
kp = 500
```

Critical implementation details:

- use `robot0_eef_quat_site`, not `robot0_eef_quat`, because EEF position is a
  site position and the site quaternion is in robosuite's `[x, y, z, w]` order;
- do not reorder that quaternion before `quat2axisangle`;
- do not clip absolute axis-angle action components to `[-1, 1]`;
- after `env.reset_to(initial_state)`, force controller state/goal refresh with
  `ctrl.update(force=True)` and `ctrl.reset_goal()`.

Result on both `image_v15.hdf5` and the earlier Route B
`image_v15_delta_eef.hdf5` dataset:

| demos | final success | mean pos err | max pos err | mean ori err | max ori err |
|---:|---:|---:|---:|---:|---:|
| 200 | 200/200 = 1.0 | 0.51 cm | 2.90 cm | 0.39 deg | 5.81 deg |

This overturns the previous broad conclusion that built-in robosuite absolute
OSC cannot execute EEF pose labels. The previous Plan B-2 failure tested a
different and incomplete interface: position-only absolute targets, lower
`kp`, altered uncoupling, and, in standalone diagnostics, the wrong quaternion
source / action clipping. It remains true that `delta_eef_action -> OSC delta`
and `real delta_EEF -> OSC command` adapters fail; those are different action
interfaces.

Decision: reopen Route B with a simpler next experiment: train a diffusion
policy to predict the executable full-pose EEF action above and roll it out
through OSC absolute/world mode. The Panda Mink results are now historical
controller exploration, not the required controller path.

Relevant files:

```text
docs/route_b_validation/playback_eef_pose.py
outputs/route_b_validation/playback_eef_pose_all_200.json
```

### Decision: next Route B training should use clean-image DDIM full-pose OSC

Agreed next experiment, not yet executed: train a clean-image diffusion policy
to predict the corrected executable full-pose OSC action. Do not start from the
masked-image policy for this first Route B test.

Planned policy / scheduler setup:

```text
observation setup: clean image policy, no oracle target mask
action target: [next_eef_pos, quat2axisangle(next_eef_quat_site), gripper]
action normalization: min_max
sampler: DDIM
num_train_timesteps: 100
num_inference_timesteps: 10
```

Rationale:

- absolute axis-angle components can exceed `[-1, 1]`, so normalization is
  required for stable DP training;
- inference should unnormalize before `env.step`;
- corrected OSC execution depends on controller metadata and reset handling,
  not just action labels.

Implementation cautions:

- dataset generation must use `robot0_eef_quat_site` in robosuite `[x, y, z, w]`
  order, with no quaternion reordering before `quat2axisangle`;
- the dataset / checkpoint env metadata must use `OSC_POSE`, absolute input,
  world reference frame, and `kp=500`;
- if the rollout path calls `env.reset_to(...)`, controller references / goals
  must be refreshed afterward, matching the successful replay script;
- after action unnormalization, verify that robomimic wrappers do not clip the
  absolute pose action to `[-1, 1]`;
- inference should not need per-step quaternion conversion if the policy output
  is already axis-angle.

Validation order:

1. Generate the absolute EEF OSC dataset.
2. Replay expert actions through the corrected controller.
3. Replay through the robomimic env wrapper to check clipping / reset behavior.
4. Run debug or short training and inspect action dimension, normalization
   stats, checkpoint env metadata, and rollout action scale.
5. Run full PickPlaceCan training only after the above pass; multi-object
   rollout should wait until the single-object clean policy works.

---

### Action-chunk ranking tested: safe chunks exist, geometry-only scoring is not enough

Tested action-chunk ranking as the next inference-time `Delta geo` strategy:

```text
sample K diffusion action chunks
-> score predicted EEF trajectories with obstacle geometry
-> execute selected chunk without gradient-updating actions
```

Same-state diagnostic on `PickPlaceBreadCerealCan` answered the immediate open
question: the base single-object policy does sample geometry-safer chunks. At
`K=16`, forward-model ranking improved actual clearance in 8/10 states with
mean `+1.54 mm`, while cumsum ranking improved 4/10 states with mean
`-0.74 mm`. This supports the forward model as a better action-chunk evaluator
than `cumsum(action * 0.05)`.

Rollout pilots did not show success improvement. On `PickPlaceBreadCerealCan`
with 10 rollouts, pure forward-model ranking reduced collision-any but hurt
success; a gated variant restored success in that small setting but increased
collision duration. On the hardest environment `PickPlaceBreadCerealMilkCan`
with 3 seeds x 20 rollouts:

| condition | success | collision-any | collision steps |
|---|---:|---:|---:|
| no guidance | 53/60 = 0.883 | 0.417 | 23.63 |
| forward-model ranking K=4 | 35/60 = 0.583 | 0.383 | 47.27 |
| forward-model ranking K=16 gated | 50/60 = 0.833 | 0.367 | 36.90 |

Current interpretation:

- The failure mode is not lack of safe samples.
- Forward-model ranking is more faithful than cumsum ranking in same-state
  diagnostics.
- Geometry-only selection is too myopic: it can choose safer but less
  task-preserving chunks, hurting grasp / placement timing and sometimes
  increasing collision duration.
- The oracle-mask baseline is already high even in the hardest environment
  (`127/150 = 0.847` in the prior 600-rollout baseline; `53/60 = 0.883` in
  this pilot), so `Delta geo` is a smaller remaining success bottleneck than
  assumed.

Decision: do not continue tuning geometry-only ranking as a final method while
the corrected full-pose EEF Route B path is open. If ranking is revisited, add
a task-preserving term such as distance to the first sampled chunk, policy
likelihood, or task-progress heuristics. Also re-check whether failure episodes
are truly caused by non-target collisions, because collision-any and success
are not tightly coupled in these pilots.

Relevant doc: `docs/action_chunk_ranking_report.md`.

---

## 2026-07-03

### Route B status narrowed: built-in delta routes fail, Mink expert replay passes, learned Mink EEF policy fails

The 2026-06-26 Route B conclusion was too broad. The corrected conclusion is:

- Built-in robosuite controller routes tested here fail for EEF supervision:
  `delta_eef_action -> OSC`, position-only absolute OSC, and built-in IK all
  diverge in open-loop replay. This was later corrected again on 2026-07-06:
  full-pose absolute OSC replay is feasible when using `robot0_eef_quat_site`
  and the correct absolute action interface.
- `real delta_EEF -> OSC command` adapters improve offline regression but remain unreliable in open-loop replay, so adapter-based execution is rejected.
- Panda + WholeBodyMinkIK can replay expert full-pose absolute EEF targets with high task success.
- A learned full-pose EEF diffusion policy using the Mink interface still obtains 0.0 rollout success.

Current interpretation after the 2026-07-06 correction: the controller-interface
part is feasible with both corrected built-in OSC absolute mode and Mink IK.
The remaining open question is learned closed-loop full-pose EEF target
generation.

Relevant docs: `docs/route_b_validation/report.md`, `docs/route_b_validation/adapter_report.md`, `outputs/route_b_validation/panda_mink_controller/STAGE_CONCLUSION.md`.

### OSC forward model validated as trajectory predictor, not as proof of guidance usefulness

Trained a learned forward model:

```text
f_hat(state_t, OSC action chunk) -> actual future EEF xyz trajectory
```

The model substantially outperforms `cumsum(action * 0.05)` on held-out demo windows and random-action env rollout validation. Random full-range actions increase absolute error, as expected for out-of-distribution inputs, but the learned model still beats cumsum.

Current interpretation: the forward model is useful as an action-chunk evaluator / trajectory predictor. This does not mean the current gradient-based obstacle guidance works.

Relevant docs: `outputs/forward_model/osc_eef_forward_image_v15/summary.md`, `outputs/forward_model/random_rollout_validation/`.

### Gradient guidance remains unreliable

Controlled rollout comparison found that replacing cumsum with the learned forward model inside the current gradient guidance did not improve success rate. A counterfactual same-state diagnostic showed that guidance can improve predicted clearance while actual executed clearance improves inconsistently.

Current interpretation: the bottleneck is likely the guidance update / cost / action-distribution interaction, not autograd or forward-model differentiability.

### Decision: next test action-chunk ranking

Before further tuning gradient guidance, answer whether the base single-object policy samples any geometry-safe action chunks in obstructed scenes.

Next method:

1. Sample multiple candidate action chunks from the diffusion policy.
2. Predict EEF trajectories with the learned forward model.
3. Score/rank candidates using obstacle geometry.
4. Execute the safest candidate without gradient-updating the action tensor.

If ranking cannot find safe chunks, inference-time geometry correction is probably insufficient and the project must introduce a new training distribution or training-side adaptation.

Handoff doc: `docs/forward_model_guidance_next_steps.md`.

---

## 2026-06-26

### Route B validation: initial built-in EEF-controller routes fail open-loop replay

Historical note: this conclusion was narrowed on 2026-07-03 and corrected again
on 2026-07-06. It applies to the specific routes tested here, especially
`delta_eef_action -> OSC delta`, position-only absolute OSC, and built-in IK.
It does **not** apply to corrected full-pose absolute OSC replay.

Tested 4 supervision-signal/controller combinations on 5 demos (full report in `docs/route_b_validation/report.md`):

| Plan | Signal | Controller | End error | Pass? |
|---|---|---|---|---|
| A | OSC delta action | OSC delta | 0.4 cm | ✓ |
| B-1 | `delta_eef_action` (real EEF delta) | OSC delta | 39.3 cm | ✗ |
| B-2 | `next_eef_pos` (absolute target) | OSC absolute | 37.3 cm | ✗ |
| C | `next_eef_pos` cumulative delta | IK delta | 74.1 cm | ✗ |

### Why these EEF-based signals failed

- **OSC is force-control, not position-control.** With `kp=150, damping_ratio=1` the natural frequency is `sqrt(150) ≈ 12 rad/s`, settling time ~0.33s. At 20 Hz (50ms/step), the EEF moves only ~15% of the way to a small per-step target. Compounding over 300 steps gives 28-39 cm lag.
- **`delta_eef_action` is the *achieved* delta in the data, smaller than the *commanded* delta that produced it.** When fed back to OSC as a new command, OSC under-achieves again — feedback loop with gain 0.28 shrinks the target by 28% per step.
- **The tested OSC absolute path was incomplete.** It sent only
  `next_eef_pos`, used zero orientation targets, lower `kp`, and did not use
  the corrected `next_eef_quat_site -> axis-angle` action interface. This
  explains why the historical B-2 result diverged while the corrected
  full-pose OSC absolute replay succeeds.
- **IK controller only supports delta mode for single-arm robots** (`ik.py:265` asserts `use_delta=True` for `num_ref_sites == 1`). The `IKSolver` in `ik_utils.py` supports absolute mode but is not wired into the standard `arm_controller_factory`. The IK step overshoots by 5-10x with default `Kpos=0.95, integration_dt=0.1`. `Kpos` and `Kn` (nullspace gain) are hard-coded into the static-method `compute_joint_positions` and don't honor attribute overrides.

### Re-evaluation: the 3-4cm RMSE from `cumsum(action*0.05)` is not approximation error — it's OSC tracking error

Plan B-1 demonstrates that the *achieved* EEF delta is systematically smaller than the *commanded* delta. The mapping `action → EEF` is not a noise problem; it's the structural tracking lag of a force-control PD law. This means:

- `cumsum(action * 0.05) ≠ actual EEF trajectory` because OSC's force controller is underdamped for small per-step commands.
- The 3-4 cm RMSE we measured earlier is a *physical limit*, not a calibration error.
- Any Route B approach that re-derives EEF from OSC delta action will hit the
  same wall. This does not apply to directly predicting the corrected
  executable full-pose EEF action.

### Historical decision: pause Route B; re-evaluate strategy

This decision is superseded by the 2026-07-06 corrected full-pose OSC absolute
replay result. At the time, three concrete next steps were discussed (see
report section "Re-evaluation of remaining options"):

1. **Multi-task learning** (Plan A + EEF prediction as separate heads). Smallest delta, sidesteps the controller-bypass problem. Cost guidance operates on the EEF prediction; execution uses the action prediction via OSC.
2. **Custom absolute-position controller** (write ~50 lines of code: IKSolver + JointPositionController in absolute mode). Requires fixing the IK nullspace/step-size issues.
3. **Accept the 3-4 cm RMSE and improve guidance differently** (e.g., longer-horizon cost, residual model).

Open question: with 600 rollouts showing guidance *hurts* rather than helps, is the cost gradient even pointing in the right direction at all? This is independent of the mapping error.

---

## 2026-06-22

### Core finding: action→trajectory mapping error is the root cause

The cost function uses `cumsum(action * 0.05)` to convert predicted actions to EEF waypoints. Calibration revealed:
- Actual mapping: 3-4cm RMSE vs OSC PD-controller dynamics
- This error is comparable to obstacle radii (3-5cm), making collision cost unreliable
- `delta_pos_scale` = 0.05 (from controller config) is correct for normalized `[-1,1]` actions, but our actions were not normalized (`normalization = null`). Scale=1.0 was 6-300x worse.
- Even the best approximation (0.05) has 3-4cm error — too large for reliable cost computation

### Cost guidance experiment: no improvement across 600 rollouts

- Baseline vs Guided across 4 environments × 3 seeds × 50 rollouts each
- Guidance slightly **reduces** success rates (-3% to -5%) as distractor count increases
- Guidance trigger rate constant at ~12.5% regardless of environment difficulty → not collision-responsive
- PC cost max = 1.13e-04, oracle cost max = 6.66e-04 → both near zero
- Policy's x0_hat does not deeply penetrate obstacles

### Z-axis check in cost function may be a secondary issue

`obstacle_xyz_cylinder_cost` requires both XY penetration AND EEF below obstacle top Z. In pick-and-place tasks, EEF is always above obstacles → `pen_z` always 0 → cost always 0. `obstacle_xy_cost` (XY only) should be tested.

### Decision: Route B — switch prediction target to EEF trajectory

Historical note: this decision was tested and superseded by the 2026-06-26,
2026-07-03, and 2026-07-06 Route B findings. The viable formulation is not a
position-only EEF trajectory; it is the executable full-pose EEF action
`[next_eef_pos, quat2axisangle(next_eef_quat_site), gripper]`.

Following from the root cause analysis:
1. Rebuild hdf5 dataset with EEF trajectory labels (not just actions)
2. Modify diffusion policy output from `[7, 16]` to `[3, 16]`
3. Eliminate `action_chunk_to_eef_xyz_traj` conversion entirely
4. Re-train and test whether guidance becomes effective

### Discussion: relation to Lan-o3dp

Lan-o3dp predicts EEF trajectory directly — they never face this mapping error. Our work differs in two ways:
1. We discovered and documented this limitation (they didn't)
2. We provide the Δvis + Δgeo decomposition framework (they don't)

### Agent infrastructure planning

Set up project workspace for multi-agent collaboration:
- `AGENTS.md` as entry point for all agents
- `docs/RESEARCH_LOG.md` for discussion history
- `.opencode/agents/paper-reader.md` (Kimi K2.6) for paper analysis
- `.opencode/agents/code-explorer.md` (DeepSeek V4 Pro) for code exploration

## 2026-06-21 (approximate)

### Earlier discussion: diagnostic tools and failure analysis

Created three diagnostic tools:
- `diagnose_collisions.py`: passive EEF-obstacle distance logging during baseline rollouts
- `diagnose_guidance_gradient.py`: per-denoising-step cost/grad_norm logging
- `calibrate_action_scale.py`: action↔EEF delta mapping verification

### Collision diagnostics: all failures are collision-type

On PickPlaceBreadCerealMilkCan (hardest environment), baseline diagnostics show ALL failures involve collisions (hit_threshold < 3cm). Zero non-collision failures. This confirms the Δvis+Δgeo decomposition: oracle mask solves Part A (visual ambiguity), leaving only Part B (physical obstruction).

### Pointcloud caching optimization

Added static obstacle caching in `run_obstacle_guided_agent.py`. Obstacles are static (placed once at reset), so pointcloud is computed once per episode instead of every step. Saves ~674ms per episode (99.8%). Combined with `guidance_start_step_pct=0.7` (skip early denoising steps) for ~20% end-to-end speedup.

### Training-side approaches discussed

Several directions considered for injecting multi-object information during training:
- Object position vector conditioning (oracle in sim, detector+depth in real)
- Structured depth (target-highlighting in depth channel)
- Training-time geometric augmentation (synthetic obstacles in depth)
- Curriculum learning with increasing distractor density

None chosen yet — blocked on the action→trajectory representation issue.
