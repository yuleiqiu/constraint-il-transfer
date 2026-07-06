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

## 2026-07-06

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

Decision: do not continue tuning geometry-only ranking as a final method. If
ranking continues, add a task-preserving term such as distance to the first
sampled chunk, policy likelihood, or task-progress heuristics. Also re-check
whether failure episodes are truly caused by non-target collisions, because
collision-any and success are not tightly coupled in these pilots.

Relevant doc: `docs/action_chunk_ranking_report.md`.

---

## 2026-07-03

### Route B status narrowed: built-in controllers fail, Mink expert replay passes, learned EEF policy fails

The 2026-06-26 Route B conclusion was too broad. The corrected conclusion is:

- Built-in robosuite controller routes fail for EEF supervision: `delta_eef_action -> OSC`, absolute OSC, and built-in IK all diverge in open-loop replay.
- `real delta_EEF -> OSC command` adapters improve offline regression but remain unreliable in open-loop replay, so adapter-based execution is rejected.
- Panda + WholeBodyMinkIK can replay expert full-pose absolute EEF targets with high task success.
- A learned full-pose EEF diffusion policy using the Mink interface still obtains 0.0 rollout success.

Current interpretation: the controller-interface part is feasible with Mink IK, but Route B as learned EEF target generation is not solved. Open-loop expert replay does not imply closed-loop learned policy rollout.

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

### Route B validation: built-in EEF-controller routes fail open-loop replay

Historical note: this conclusion was narrowed on 2026-07-03. It applies to the
built-in robosuite controller routes tested here. The later Panda
WholeBodyMinkIK follow-up can replay expert EEF targets, but learned EEF policy
rollout still fails.

Tested 4 supervision-signal/controller combinations on 5 demos (full report in `docs/route_b_validation/report.md`):

| Plan | Signal | Controller | End error | Pass? |
|---|---|---|---|---|
| A | OSC delta action | OSC delta | 0.4 cm | ✓ |
| B-1 | `delta_eef_action` (real EEF delta) | OSC delta | 39.3 cm | ✗ |
| B-2 | `next_eef_pos` (absolute target) | OSC absolute | 37.3 cm | ✗ |
| C | `next_eef_pos` cumulative delta | IK delta | 74.1 cm | ✗ |

### Why every EEF-based signal fails

- **OSC is force-control, not position-control.** With `kp=150, damping_ratio=1` the natural frequency is `sqrt(150) ≈ 12 rad/s`, settling time ~0.33s. At 20 Hz (50ms/step), the EEF moves only ~15% of the way to a small per-step target. Compounding over 300 steps gives 28-39 cm lag.
- **`delta_eef_action` is the *achieved* delta in the data, smaller than the *commanded* delta that produced it.** When fed back to OSC as a new command, OSC under-achieves again — feedback loop with gain 0.28 shrinks the target by 28% per step.
- **OSC absolute mode has a directional bug when `uncouple_pos_ori=True`.** The force comes out reversed. Setting `uncouple_pos_ori=False` fixes direction but the absolute-mode PD law is still under-tuned for 20 Hz position targets.
- **IK controller only supports delta mode for single-arm robots** (`ik.py:265` asserts `use_delta=True` for `num_ref_sites == 1`). The `IKSolver` in `ik_utils.py` supports absolute mode but is not wired into the standard `arm_controller_factory`. The IK step overshoots by 5-10x with default `Kpos=0.95, integration_dt=0.1`. `Kpos` and `Kn` (nullspace gain) are hard-coded into the static-method `compute_joint_positions` and don't honor attribute overrides.

### Re-evaluation: the 3-4cm RMSE from `cumsum(action*0.05)` is not approximation error — it's OSC tracking error

Plan B-1 demonstrates that the *achieved* EEF delta is systematically smaller than the *commanded* delta. The mapping `action → EEF` is not a noise problem; it's the structural tracking lag of a force-control PD law. This means:

- `cumsum(action * 0.05) ≠ actual EEF trajectory` because OSC's force controller is underdamped for small per-step commands.
- The 3-4 cm RMSE we measured earlier is a *physical limit*, not a calibration error.
- Any Route B approach that re-derives EEF from action will hit the same wall.

### Decision: pause Route B; re-evaluate strategy

Three concrete next steps discussed (see report section "Re-evaluation of remaining options"):

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

Historical note: this decision was tested and superseded by the 2026-06-26 and
2026-07-03 Route B findings. Direct EEF-action policy training is not currently
a working route.

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
