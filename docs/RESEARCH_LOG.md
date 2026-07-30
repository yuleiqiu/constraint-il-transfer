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

## 2026-07-29

### Persistent correction creates recoverability windows but covers 7/20 states

Tested the existing SDP-6k checkpoint as a temporally persistent corrective
option on all 20 strict D2 states. Four correction trajectories per state were
replanned every eight actions for at most 20 chunks; every collision-free
boundary was cloned and handed to the frozen source through absolute horizon
600. This expands the three-state pilot without changing its protocol.

The fixed existence gate passes, but state coverage is 7/20. Within the seven
recoverable states, 27/28 trajectories contain a safe handoff; the other
thirteen states have 0/52. Five of seven recoverable states require more than
one correction chunk, and their median earliest handoff is chunk five. No
correction trajectory directly completes the task. Recoverability is
non-monotonic, and 20 boundaries complete the task with collision rather than
safely.

Decision: retain the recoverability-gated corrective option as the active
method hypothesis. Temporal persistence generates source-recoverable states
that the rejected one-chunk candidate set does not contain, and the
non-monotonic windows establish that a safe handoff model is necessary.
However, the current SDP generator's 7/20 oracle coverage is well below the
18/20 waypoint-teacher upper bound. Do not train a handoff estimator against
this generator yet; first train a correction-only temporally extended option
from the existing verified intervention segments, collecting more only if the
stored sequences are insufficient, then repeat the same oracle gate. Method
note: `docs/recoverability_corrective_option/README.md`. Canonical result:
`outputs/persistent_correction_oracle/full_20/conclusion.md`.

### SDP succeeds in D1 when the execution budget matches slower correction

Evaluated the already selected SDP-6k checkpoint in the previously untouched
one-distractor transfer domain rather than treating its D2 failure as a
universal result. On the paired seed-600 development bank at horizon 400, SDP
changed Task SR / CR from 0.60 / 0.30 to 0.75 / 0.10, with four
failure-to-success and one success-to-failure transition, passing the fixed
transition gate.

On 100 untouched paired layouts from seeds 601/602, horizon-400 Task SR / CR
changed from 0.66 / 0.25 to 0.72 / 0.18. Extending both policies equally
showed that the corrective behavior needs more execution steps: at horizon
600, source / SDP Task SR is 0.67 / 0.78, with 12 failure-to-success versus one
success-to-failure transition (two-sided exact paired `p=0.0034`). CR is
0.26 / 0.19. All additional SDP successes finish by step 473, and neither
policy gains another success between horizons 600 and 800.

Decision: retain D1 at horizon 600 as the positive targeted-transfer result
and D2 as the demonstrated capacity boundary. The result does not justify new
modules or a checkpoint sweep. A 400-step budget understates safe but slower
correction; inference compute time itself does not consume environment steps.
Canonical report:
`outputs/targeted_finetuning/d1_horizon_salvage_report.md`.

### Oracle future cannot rescue the current one-chunk SDP candidate set

Tested the proposed simulator-branch value upper bound before training a
world model. On the first three held-out one-chunk-lead collision states, each
source and 32 SDP candidate chunks was executed from the exact saved simulator
state and policy observation history, then handed to the same frozen source
policy for the remaining 400-step episode horizon. The pilot gate required at
least one corrective Task success before expanding to all 20 states.

All 96 corrective chunks avoided physical contact during their first eight
actions, and 30 passed the 1 cm / non-retreat geometry filter. Their complete
continuations nevertheless produced 32 NCRs and 64 collision failures, with
zero Can contacts and zero Task successes. A hindsight oracle therefore had
no successful branch to select. On the same three states, the discrete-yaw
waypoint teacher obtained 3/3 collision-free Task success using 121--144
controlled steps before handoff.

Decision: stop at the pilot gate; do not spend the remaining 561 full
continuations or train a world/value model that only ranks the current
one-chunk SDP samples. This does not reject long-horizon model-based planning
that generates new behavior. It shows that selection alone is not the
remaining problem: the corrective generator lacks a temporally extended
avoidance/grasp/recovery behavior. The next controlled model should be an
explicit multi-step waypoint or residual expert with the source frozen and a
defined handoff. Canonical report:
`outputs/oracle_branch_value/pilot_3state/result_report.md`.

### Oracle-gated corrective sampling still loses task completion

Combined the two preceding partial results by freezing the epoch-260 source
and invoking the SDP-6k policy only under an oracle pre-contact obstruction
gate. On the 20 strict states, requiring SDP candidates to preserve the source
chunk's progress gave 0/20 coverage; the preceding 10/20 support result was
relative to SDP's own slower nominal behavior. Under an explicit
no-substantial-retreat rule, 11/20 states had at least one 1 cm-clearance
candidate among 32 samples.

The paired D2 seed-600, 20-layout gate nevertheless changed Task SR / Safe SR /
CR / NCR from 0.25 / 0.15 / 0.80 / 0.05 to
0.15 / 0.10 / 0.75 / 0.15. Thirteen chunks triggered, five executed a
corrective sample, and eight fell back exactly to source. There were zero
failure-to-success and two success-to-NCR transitions, with no clipping or
observation errors. Four of five selected chunks avoided immediate contact
but did not reliably return to a task-completing trajectory.

Decision: reject and stop this local oracle-gated SDP selector; do not run
D1/D3, extra seeds, or a threshold/candidate-count sweep. The next bounded
question is whether cloned-simulator branch continuation can supply a
task-value upper bound for candidate selection. If even oracle branch scoring
cannot select useful continuations, replace the corrective diffusion policy
with an explicit waypoint/residual expert rather than training a learned
world model. Canonical report:
`outputs/gated_support_selector/report.md`.

### SDP preserves corrective support but fails task-preserving selection

Implemented and trained the paper-aligned SDP comparison on the fixed 2,849
paired correction windows using normalized per-timestep desired-action balls,
`r=0.1`, `N=16`, and truncated target denoising from `K_A=16`. A separate
90/10 clean / paired-positive BC run tested whether the earlier 50/50 result
was only excessive correction exposure. Both treatments completed 10,000
updates with D0 Task SR between 0.95 and 1.00 at every screened checkpoint.

Neither treatment passed the paired D2 gate. The best 90/10 and SDP Task SR
was 0.25 versus source 0.35. The best-Safe-SR SDP checkpoint had two
failure-to-success and four success-to-failure transitions. At the strict
1 cm fixed-state diagnostic, however, SDP reached 10/20 direct safe-progress
states and 11/20 projection-feasible states; the 90/10 endpoint reached 6/20
and 8/20. These sit well above source 0/20 and 1/20.

Decision: reject the tested SDP configuration and lower-exposure positive BC;
do not advance to D1/D3 or seeds 601/602, and do not launch a broad ratio or
radius sweep. The repeated pattern localizes the remaining problem more
strongly: full-policy correction training can create avoidance support, but
does not reliably decide when to enter, sustain, and exit that mode while
preserving grasp and placement. The next controlled upper bound should freeze
the source and test an explicitly gated correction residual / avoidance expert
with a defined handoff. Canonical report:
`outputs/targeted_finetuning/sdp_result_report.md`.

## 2026-07-28

### Positive correction BC injects support but fails the closed-loop gate

Trained full-policy targeted and clean-only continuations from the same
epoch-260 checkpoint for 10,000 updates. The treatment used verified
intervention-centered positive chunks with 50/50 clean rehearsal. On the fixed
20-layout development bank, source and clean-only both obtained D2 Task SR
0.35 with identical success/failure assignments. Every mixed checkpoint from
2,000 through 10,000 updates was worse; the 10,000-update model obtained Task
SR / Safe SR / CR / NCR of 0.25 / 0.25 / 0.30 / 0.45, with two
failure-to-success and four success-to-failure transitions. D0 remained
1.00 and no action clipping occurred.

The same mixed checkpoint nevertheless changes the strict 1 cm fixed-state
support result from source 0/20 direct-hit and 1/20 projection-feasible states
to 10/20 and 13/20. The update-matched clean control remains at 0/20 and 1/20.
Thus corrections successfully inject local safe-progress action support, but
positive-only BC does not preserve or select that mode in a way that improves
the complete task.

Decision: reject this 50/50 ordinary positive-BC recipe; do not run the
untouched final D1/D3 matrix or collect more identical corrections. Reuse the
fixed $(\mathcal O,A^-,A^+)$ dataset for an SDP-style set-supervision
comparison, with at most one lower-exposure positive-BC control to separate
the objective from correction weighting. Canonical report:
`outputs/targeted_finetuning/upper_bound_report.md`. Successor contract:
`docs/targeted_finetuning/sdp_successor_plan.md`.

### Paired-correction C1 dataset passes

Expanded the verified C0 set to 25 train-eligible, successful,
collision-free two-distractor episodes. They yield 2,849 deterministic
correction pairs (2,120 train / 729 validation); every positive and negative
action remains inside the epoch-260 normalization range, and all 25 episodes
reproduce success and contact outcomes under open-loop replay.

One accepted branch revealed that the previous safe-teacher-interval rule
could admit a trajectory that had collided before the trigger. The active
contract now requires the complete selected branch to be distractor-contact
free. That episode and two replay-sensitive C1 episodes remain in the raw
audit file but are excluded, and three fresh verified branches replace them.

Decision: C1 passes; stop correction collection and implement the fixed
50/50 clean-rehearsal / paired-positive BC training interface. No policy
adaptation result has yet been established. Canonical report:
`outputs/targeted_finetuning/c1_report.md`.

### Paired-correction C0 data gate passes

Implemented fresh two-distractor correction collection with exact
`2 / 16 / 8` sample alignment, a per-teacher-step frozen-policy query, an
append-only raw episode layer, deterministic $(\mathcal O,A^-,A^+)$ pair
extraction, source-normalization checks, and open-loop replay verification.

The final C0 set contains 10 train-eligible episodes and 1,135 pairs
(1,024 train / 111 validation). All positive and negative actions remain
inside the epoch-260 normalization range. All eligible episodes reproduce
task success and target contact without distractor contact; maximum replay EEF
position / orientation error is 0.583 cm / 0.0137 rad. One borderline
task-success replay remains in the raw audit file but is excluded.

The first 0.04-rad yaw configuration failed the action-range gate in two
+20-degree episodes, reaching normalized rotation-z 1.98. The fixed collection
configuration uses 0.015 rad and reaches at most 0.837 on that dimension.
Decision: C0 passes and collection may proceed to the approximately 25-episode
stage; no training conclusion has been established. Canonical report:
`outputs/targeted_finetuning/c0_report.md`.

### Targeted fine-tuning selected as the next upper-bound experiment

An oracle Cartesian waypoint teacher was tested on the 20 held-out strict
pre-collision states. Discrete yaw search found a safe lifted grasp in 18/20
states. Teacher-then-policy Task success changed from 1/20 to 18/20 and
distractor collision from 20/20 to 2/20, with 17 failure-to-success and no
success-to-failure transitions. This establishes that task-compatible safe
continuations exist outside the frozen policy's effective local support; it is
a selected-state teacher upper bound, not a rollout policy result.

Decision: use the two-distractor environment as the minimum-hard adaptation
domain, with the original, one-distractor, and three-distractor environments
serving as retention, lower-clutter interpolation, and higher-clutter
extrapolation tests. Collect complete successful hybrid episodes on fresh
layouts as an audit/replay layer, keeping the existing 20-state diagnostic
bank and evaluation seeds held out. During each teacher intervention, query
the frozen policy at every visited teacher state and deterministically derive
paired $(\mathcal O,A^-,A^+)$ chunks. This replaces intervention masks,
oversampled full-episode windows, and manual HDF5 merging.

The first experiment remains a full-policy adaptation upper bound: ordinary
positive-correction BC with balanced original-clean replay. Compare it with
the frozen checkpoint and a clean-only continued-training control, and select
checkpoints by held-out Task SR / Safe SR / CR / NCR plus the fixed support
diagnostic rather than training loss. If it passes, reuse the identical pair
dataset for an SDP set-supervision comparison so the effect of correction data
is separated from the effect of the SDP objective.

World-model approaches remain possible successors, but should not be adopted
for narrative novelty before this upper bound establishes whether the task is
learnable from targeted corrective support. Canonical teacher artifact:
`scripts/2026-07-27_semi_manual_waypoint_teacher/README.md`. Active experiment
contract: `docs/targeted_finetuning/plan.md`.
Paired-data contract:
`docs/targeted_finetuning/paired_correction_data.md`.

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
