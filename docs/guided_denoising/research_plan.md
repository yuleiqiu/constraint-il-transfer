# Guided-Denoising Stage Record

Status: completed inference-only research stage, superseded 2026-07-28.

The successor targeted fine-tuning stage is also complete. Its final D1/D2
boundary and the persistent-correction result are summarized in
`outputs/targeted_finetuning/sdp_result_report.md` and
`outputs/persistent_correction_oracle/full_20/conclusion.md`. This file is a
compact historical index of the frozen-policy stage. Git history retains the
former detailed working plan.

## 1. Starting Baseline

The reliable clean-image epoch-260 delta-EEF policy produced:

| distractors | Task SR | Safe SR | CR | NCR |
|---:|---:|---:|---:|---:|
| 0 | 0.907 | 0.907 | 0.000 | 0.093 |
| 1 | 0.707 | 0.687 | 0.213 | 0.100 |
| 2 | 0.253 | 0.240 | 0.647 | 0.113 |
| 3 | 0.200 | 0.153 | 0.680 | 0.167 |

Task-level evaluation always required Task SR, Safe SR, CR, NCR, and the
four-way partition:

```text
safe success
success with collision
collision failure
collision-free non-completion
```

Local trajectory clearance was mechanism evidence, not a success criterion.

## 2. P0 Failure Partition

The 600-episode intent diagnostic divided critical failure chunks into:

```text
target_directed_obstructed
non_target_directed
target_directed_clear
```

Two- and three-distractor failures had no common majority category. The fixed
decision was C: mixed / no single dominant mechanism. Consequently, every
candidate had to name its targeted regime and report regime-stratified effects;
obstacle feasibility could not be presented as a universal explanation.

Canonical result:
`outputs/guided_denoising/intent_diagnostic/report.md`.

## 3. Tested Inference-Only Branches

- **LAN-style guided denoising:** changed predicted and executed trajectories,
  but its paired safety effects were inconsistent.
- **Sample-refine:** four-candidate sampling plus masked refinement left all 20
  complete paired task outcomes unchanged while adding 76% inference time.
- **Task-constrained eight-step correction:** recovered two task successes
  among nine feasible layouts, but all nine retained distractor collision and
  the preregistered Can-contact gate failed.
- **LAN-aligned point-cloud policy and DDPM guide:** the source checkpoint
  obtained only 0.50 clean Task SR over 150 episodes; calibrated guidance
  produced no failure-to-success transition and caused paired regressions.
- **One-distractor 5 cm XY expansion:** removed the only baseline collision,
  but Task SR fell 0.70 to 0.65 and NCR rose 0.25 to 0.35.
- **Policy-support diagnostic:** none of 640 samples was safe and
  progress-preserving at the 1 cm boundary; support-constrained projection was
  feasible in 1/20 states versus 14/20 without the support bound.

## 4. Stage Conclusion

For this checkpoint and tested local guidance family:

```text
valid source policy and action interface
    -> large degradation under distractors
    -> oracle inference information changes local motion
    -> local clearance improves without reliable Task SR gains
    -> safe paths often lie outside effective policy support
```

Do not continue with another nearby scale, margin, or denoising-schedule sweep
of the same point-clearance guide. This is not a universal impossibility claim
for all frozen-policy planners or world models.

The next experiment must first create or supply task-preserving avoidance
support. The project selected targeted training-side correction as the
learnability upper bound.

## 5. Canonical Artifacts

- synthesis:
  `docs/guided_denoising/frozen_policy_inference_limitations.md`
- source policy:
  `outputs/eef_pose_osc_policy/multienv_eval_report.md`
- first mechanism baseline:
  `docs/guided_denoising/lan_o3dp_mechanism_baseline.md`
- sample-refine:
  `docs/guided_denoising/sample_refine_pilot.md`
- task-constrained correction:
  `outputs/task_constrained_guidance/stage1_report.md`
- LAN-aligned closure:
  `docs/guided_denoising/lan_o3dp_epoch350_closure.md`
- support diagnostic:
  `outputs/policy_manifold_diagnostic/report.md`
- chronological decisions:
  `docs/RESEARCH_LOG.md`
