# Research Log

## Maintenance Rules

- Reverse chronological; one `## YYYY-MM-DD` section per decision date.
- Keep findings, decisions, rationale, and canonical artifacts.
- Exclude commands, implementation walkthroughs, and data already available in
  `outputs/` or a dedicated report.
- Mark unresolved claims `[UNVERIFIED]`.
- When a result is superseded, retain only the correction and historical
  significance; Git history preserves the full original narrative.

## Current Unverified Hypotheses

- [UNVERIFIED] A materially different, volume-aware geometry model may improve
  one-distractor Task SR with the reliable clean-image delta-EEF policy; the
  tested uniform 5 cm XY expansion does not.
- [UNVERIFIED] A small balanced set of successful obstacle-avoidance
  corrections can inject the missing action support into the epoch-260 policy
  without materially degrading its clean behavior.

## 2026-07-28

### Targeted fine-tuning selected as the next upper-bound experiment

An oracle Cartesian waypoint teacher was tested on the 20 held-out strict
pre-collision states. Discrete yaw search found a safe lifted grasp in 18/20
states. Teacher-then-policy Task success changed from 1/20 to 18/20 and
distractor collision from 20/20 to 2/20, with 17 failure-to-success and no
success-to-failure transitions. This establishes that task-compatible safe
continuations exist outside the frozen policy's effective local support; it is
a selected-state teacher upper bound, not a rollout policy result.

Decision: collect a small set of complete successful hybrid correction
episodes on fresh training layouts, keeping the existing 20-state diagnostic
bank and evaluation seeds held out. Fine-tune the epoch-260 policy with
balanced replay from the original clean demonstrations and oversampled
correction windows. The first experiment is a full-policy adaptation upper
bound, not the final proposed method. Compare it with the frozen checkpoint
and a clean-only continued-training control, and select checkpoints by held-out
Task SR / Safe SR / CR / NCR plus the fixed support diagnostic rather than
training loss.

World-model approaches remain possible successors, but should not be adopted
for narrative novelty before this upper bound establishes whether the task is
learnable from targeted corrective support. Canonical teacher artifact:
`scripts/2026-07-27_semi_manual_waypoint_teacher/README.md`. Active experiment
contract: `docs/targeted_finetuning/plan.md`.

## 2026-07-27

### Frozen-policy inference-only evidence reaches a negative stage conclusion

Across oracle guided denoising, multi-sample refinement, task-constrained
trajectory correction, expanded-boundary guidance, and a policy-support
diagnostic, inference-time changes improved local clearance but did not provide
reliable single-to-multi Task SR gains. At a 1 cm expanded boundary, none of
640 policy samples was safe and progress-preserving; support-constrained
projection was feasible in 1/20 pre-collision states, versus 14/20 when support
was effectively removed.

Decision: for the current epoch-260 delta-EEF checkpoint, stop treating
additional obstacle information or tuning of the same frozen-policy local
guidance family as sufficient. The evidence points to missing
task-preserving avoidance action support; training-side adaptation or an
external controller allowed to leave the policy manifold is required before
gating can help. This is not a universal impossibility claim for all planners.
Canonical synthesis:
`docs/guided_denoising/frozen_policy_inference_limitations.md`.

### One-distractor 5 cm XY expansion trades collision for non-completion

Tested the existing RGB delta-EEF guidance in `PickPlaceBreadCan`, keeping the
source policy and DDIM mechanism frozen while expanding Bread's XY radius by
5 cm. Ten fixed states selected the historically comparable scale 0.01: mean
predicted / executed safety-margin clearance changed from -0.94 / -0.59 cm to
+0.21 / +0.28 cm without action clipping.

The 20-pair pilot did not improve closed-loop performance. Task SR and Safe SR
fell 0.70→0.65. Guidance removed the only baseline collision, but NCR rose
0.25→0.35: one failure became success, two successes became failures, and one
collision failure became NCR.

Decision: reject and do not scale or retune this uniform 5 cm XY point guide.
The one-distractor environment is not itself disproven; a successor must
materially change the geometry, for example by accounting for obstacle height
or task phase, rather than another margin or scale sweep. Canonical report:
`outputs/guided_denoising/breadcan_expanded_5cm_paired_pilot/report.md`.

### LAN-aligned epoch-350 branch fails its closure gates

The seed-42, 40-demo point-cloud DDPM checkpoint obtained clean
`PickPlaceCan` Task SR 0.54 / 0.54 / 0.42 over three fixed 50-episode seeds:
75 / 150 = 0.50 pooled, Wilson 95% interval `[0.421, 0.579]`. It fails the
project's approximately 0.80 source-policy competence gate.

DDPM-100 guidance adaptation and fixed-state calibration were mechanically
successful. At scale 0.003, mean predicted / executed raw clearance increased
from 1.45 / 1.72 cm to 2.01 / 2.15 cm, with 3.15 mm maximum one-step waypoint
displacement. The 20-pair closed-loop pilot nevertheless produced no
failure-to-success transition. Two-distractor Task SR stayed 0.40 while CR
rose 0.40→0.50; three-distractor Task SR fell 0.80→0.60 as two successes
became collision-free failures.

The one-distractor `PickPlaceBreadCan` follow-up also failed: over 20 matched
pairs, Task SR stayed 0.45, Safe SR fell 0.40→0.35, and CR rose 0.05→0.15.
One failure became a success with collision, while one safe success became a
collision failure.

Decision: reject this checkpoint as the final source policy and stop tuning or
scaling the same point-level EEF guidance. This closes the tested branch, not
all possible LAN components. Canonical report:
`docs/guided_denoising/lan_o3dp_epoch350_closure.md`.

### Reliable RGB delta-EEF source retained; LPB deferred

Use the clean-image delta-EEF Diffusion Policy epoch-260 checkpoint as the
frozen source prior. Its independent clean Task SR is 0.907, its action chunks
accurately reconstruct executed EEF trajectories, and Task SR falls to 0.253 /
0.200 with two / three distractors, providing both source competence and a
meaningful transfer gap.

Decision: retain this checkpoint for subsequent method studies, but defer the
LPB direction per the current research decision. Any successor must use the
existing executable delta-EEF action and Task SR / Safe SR / CR / NCR
contract. Do not present this RGB + DDIM policy as LAN-O3DP. Canonical plan:
`docs/guided_denoising/research_plan.md`.

## 2026-07-23

### First point-cloud DDPM source misses the clean gate

The seed-500 point-cloud DDPM-100 training chain passed dataset, geometry,
encoder, checkpoint, and overfit preflight checks. A 600-epoch run peaked at
41 / 50 = 0.82 clean Task SR at epoch 580, below its fixed 0.90 gate.

Decision: do not advance this checkpoint to guidance or paired multi-object
evaluation. Canonical artifacts:
`docs/guided_denoising/lan_o3dp_reproduction_phase1.md` and
`outputs/guided_denoising/lan_o3dp_phase1_training/report.md`.

## 2026-07-22

### Task-constrained eight-step feasibility misses Stage 1

An oracle pre-target, target-directed, predicted-obstruction upper bound
changed only the existing eight-step XY path while preserving task progress,
other action dimensions, bounds, and a trust region. Nine of twenty strict
layouts admitted a correction; Can contact increased 3→6 and Task success
0→2 without paired losses, but the preregistered gate required four additional
Can contacts and observed three. All nine retained distractor collision.

Decision: stop this EEF-circle, eight-step candidate and do not implement its
posterior-gradient Stage 2. A successor needs whole-gripper or temporally
consistent task modeling, not another parameter sweep. Canonical result:
`outputs/task_constrained_guidance/stage1_report.md`.

### Sample-refine fails the two-distractor development gate

The method sampled four DDIM candidates, ranked them by task progress and
swept-segment clearance, and optionally refined XY under intent and trust
constraints. In twenty matched layouts, baseline and complete sample+refine
both obtained Task SR 0.10, Safe SR 0.05, CR 0.75, and NCR 0.20, with identical
four-way outcomes. Sample-only created one safe success; refine-only exchanged
three collision failures for three NCRs. The complete method added 76%
inference time without task benefit.

Decision: falsify this candidate. Further sampling work requires a materially
different task-value or feasibility criterion. Canonical interpretation:
`docs/guided_denoising/sample_refine_pilot.md`.

## 2026-07-21

### Superseded controller-route artifacts removed

Executable delta EEF pose resolved the action-interface question, so obsolete
controller comparisons, adapter experiments, and Mink follow-up code were
removed from the active tree. Historical evidence remains in Git.

Decision: keep the maintained OSC reset/controller support and current
delta/absolute EEF configs; use `outputs/eef_pose_osc_policy/README.md` and
`docs/eef_pose_osc_policy_training.md` as canonical references.

## 2026-07-20

### Intent diagnostic finds mixed hard-environment failure mechanisms

The P0 analysis covered all 600 unguided episodes. Among failures that never
contacted Can, the two-distractor target-directed-obstructed /
non-target-directed / target-directed-clear split was 34.4% / 43.0% / 22.6%;
the three-distractor split was 37.5% / 26.0% / 36.5%. No category passed the
fixed Wilson-majority rule in both environments. Successful critical chunks
were target-directed 92.1% / 90.0% in the two hard environments, validating
the intent measure.

Decision C: no single dominant mechanism. Every method must state its targeted
regime and cannot treat obstacle feasibility as a universal explanation.
Canonical result: `outputs/guided_denoising/intent_diagnostic/report.md`.

### Method discovery precedes environment expansion

Keep the PickPlace family as the development task. Require one unchanged
configuration to improve paired Task SR in the two- and three-distractor
variants without a collision-failure→NCR exchange before full Lift,
NutAssembly, policy-seed, or cross-task evaluation.

Decision: environment breadth is a later evidence question; the immediate
bottleneck is a task-effective method. Canonical plan:
`docs/guided_denoising/research_plan.md`.

## 2026-07-15

### First delta-EEF guided-denoising baseline changes behavior but fails scaling

Implemented opt-in predicted-clean-action guidance for the reliable RGB
delta-EEF policy. Disabled and zero-scale paths reproduce the original policy;
fixed-state guidance improves predicted and short-prefix executed clearance.
Scale 0.01 was selected over 0.03 because the latter produced implausibly
large waypoint pushes.

In twenty matched pairs, Task SR rose 0.10 in both hard environments, but
safety effects disagreed: two-distractor CR fell while NCR rose; three-
distractor CR rose and Safe SR fell. The task gain therefore did not meet the
fixed safety gate.

Decision: retain this as a mechanism baseline, not the project method, and do
not scale it. Canonical record:
`docs/guided_denoising/lan_o3dp_mechanism_baseline.md`.

## 2026-07-14

### Multi-environment evaluation defines the active baseline and metrics

The reliable RGB delta-EEF checkpoint was evaluated over four environments,
three seeds, and fifty episodes per seed. Task SR for zero through three
distractors was 0.907 / 0.707 / 0.253 / 0.200; CR was 0.000 / 0.213 / 0.647 /
0.680. Hard failures often collide before target contact, but successful
collisions and collision-free placement failures both occur.

Decision: report Task SR, Safe SR, CR, NCR, and the four-way partition. CR
reduction without Task SR improvement can be only failure redistribution.
The action interface remains healthy, with no clipping and sub-centimeter mean
trajectory reconstruction error. Canonical report:
`outputs/eef_pose_osc_policy/multienv_eval_report.md`.

## 2026-07-08

### Delta EEF pose replaces the OSC-action forward-model route

The clean-image delta-EEF policy reached 0.98 training-time rollout success.
Its predicted action chunks reconstruct the executed EEF trajectory with
0.131 cm mean position and 0.253° mean orientation error in a successful
rollout. The absolute EEF comparison peaked at 0.82.

Decision: use executable 7-D `delta_eef_pose_action` through `OSC_POSE` as the
active interface. Guidance and ranking can score its reconstructed trajectory
directly; the old OSC-action forward model is archived. Canonical result:
`outputs/eef_pose_osc_policy/README.md`.

## 2026-07-06

### Corrected full-pose EEF replay overturns the earlier controller conclusion

Using `robot0_eef_quat_site`, full-pose targets, absolute/world `OSC_POSE`,
`kp=500`, and controller refresh after reset achieved 200 / 200 expert replay
success with 0.51 cm mean position error. Earlier position-only and delta
routes had tested incomplete or non-equivalent interfaces.

Decision: built-in OSC can execute full-pose EEF labels. This historical
branch was later superseded by the simpler validated delta-EEF policy.

### Geometry-only action ranking is task-myopic

The base policy sampled geometrically safer chunks, and a learned forward
model ranked them more faithfully than `cumsum(action * 0.05)`. Rollout
success nevertheless fell: on the hardest pilot, baseline / K=4 / gated K=16
success was 0.883 / 0.583 / 0.833. Lower collision-any did not compensate for
lost task behavior.

Decision: do not use geometry-only ranking as the final method. A future
ranking method needs an explicit task-preservation or value term. Canonical
report: `docs/action_chunk_ranking_report.md`.

## 2026-07-03

### Learned OSC forward model improves prediction, not guidance

The learned `state + OSC action chunk -> future EEF trajectory` model beat the
cumulative-action proxy on held-out and random rollouts. Replacing the proxy
inside gradient guidance still did not improve Task SR, and same-state
clearance improvements transferred inconsistently to execution.

Decision: the bottleneck was the guidance update, cost, and action-distribution
interaction rather than forward-model accuracy or autograd. The implementation
is archived; canonical handoff:
`docs/forward_model_guidance_next_steps.md`.

## 2026-06-26

### Initial EEF-controller failure was route-specific

Early open-loop tests found that achieved EEF deltas cannot be replayed as new
OSC delta commands: controller tracking lag compounds into large error.
Position-only absolute OSC and the tested IK route also failed. These results
did not test the corrected full-pose absolute interface discovered on
2026-07-06.

Decision: reject adapters that infer executable EEF motion from the old OSC
delta action. Preserve this only as historical motivation for direct
executable EEF supervision.

## 2026-06-22

### Old OSC-action obstacle guidance fails end to end

The `cumsum(action * 0.05)` trajectory proxy had 3–4 cm error, comparable to
obstacle radii. Across 600 rollouts, guidance reduced success by roughly
3–5%, triggered almost independently of clutter, and produced near-zero costs.
A learned forward model later improved trajectory prediction but not guidance.

Decision: stop treating OSC action chunks as executable EEF trajectories and
move to direct EEF supervision. Later delta-EEF results supersede the original
position-only plan.

## 2026-06-21 (approximate)

### Initial diagnostics and project infrastructure

Collision, guidance-gradient, and action-scale diagnostics established the
first evidence base. Early oracle-mask experiments suggested collision-
dominated failures, but the later 600-episode delta-EEF evaluation narrowed
that claim by identifying mixed failure modes.

Established `AGENTS.md`, this research log, and specialist agent definitions.
Training-side multi-object conditioning ideas were deferred until the action
representation problem was resolved.
