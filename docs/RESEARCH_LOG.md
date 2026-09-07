# Research Log

## Maintenance Rules

- Reverse chronological; one `## YYYY-MM-DD` section per decision date.
- Keep findings, decisions, rationale, and canonical artifacts.
- Exclude commands, implementation walkthroughs, and data already available in
  `outputs/` or a dedicated report.
- Mark unresolved claims `[UNVERIFIED]`.
- When a result is superseded, retain only the correction and historical
  significance; Git history preserves the full original narrative.

## Current decision frontier

No next experiment has been selected. Persistent corrective options remain a
candidate supported by earlier evidence, not an instruction to resume training.
See root AGENTS.md for current status; older decisions below are historical.

## 2026-09-07

### Close the wider-support record and rebuild project navigation

The user confirmed the chronology: MVEE was put aside after small geometry
and rollout probes, then the original nearest-point guidance was reused for a
wider-demonstration-support experiment. The branch name does not describe the
method used by the later test.

The August 26--September 2 artifacts establish a successful 20/20 teacher
gate and a 100-demo route-diverse dataset. The selected epoch-50 point-cloud
absolute-EEF source reaches 39/50 clean successes; epoch 250 reaches 23/50.
At H600 on fifty RaisedMilk states, target-lifted point guidance changes Task
SR 0.58 to 0.62 and Safe SR 0.52 to 0.54, with Task transitions 9/7 and Safe
transitions 8/7. Neither is significant. Always-on guidance drops Task SR to
0.18. The direct-demo comparator drops from 0.34 to 0.04 under target-lifted
guidance, trading contact for non-completion.

Selected video replays reproduce full outcome labels in only 9/16 baseline
and 13/16 guided branches. Equal state hashes and seeds therefore do not
establish exact counterfactuals across processes. Learned route coverage was
not directly measured: diverse demonstrations alone do not prove that the
trained policy retained all modes. These findings do not establish useful
guidance from wider demonstrations, nor disprove every support-based approach.
Canonical evidence: outputs/tabletop_wide_support/conclusion.md.

The user authorized documentation restructuring. Root instructions now expose
the latest evidence and explicitly leave the next research decision open.
Historical plans and agent recommendations must not become standing orders.
No supervisor view has been recovered from the available records.

### Retrospective MVEE closure (experiments on August 25)

Dual-view obstacle fusion, MVEE fitting, and actor ellipsoid costs were
implemented. Three fresh collision failures became collision-free stalls at
5 cm margin; smaller margins retained contact. A same-process phase probe
restored early grasp but did not produce safe completion. This was a bounded
probe, subsequently put aside by the user, not a formal positive geometry
result or the active successor. See
scripts/2026-08-25_tabletop_mvee_guidance/README.md.

## 2026-08-24

### Tabletop guidance rescues 4/11 collision failures but usually stalls

Completed a targeted five-layout / five-noise counterfactual evaluation in the
controlled raised-Milk Tabletop environment. Eligibility requires the clean
source branch to succeed and the matched obstacle baseline to fail through
collision. Eleven of 25 triads qualify. Radial predicted-clean guidance at the
locked `rho=45`, `Q=0.08 m` setting produces four safe task-completing rescues,
all in layout 4. On the eligible set it reduces physical Milk contact from
11/11 to 1/11, but the remaining outcomes are six collision-free
non-completions and one collision failure.

The open-loop pre-contact audit confirms that the mechanism is active: all 100
reverse steps have nonzero cost gradients, final executable XY targets move by
1.31 cm on average and 2.21 cm at maximum, and clearance from the selected
point increases. Clearance to the full visible obstacle cloud nevertheless
does not improve. The cost repels the EEF trajectory from one selected point
without representing the carried Can or the whole obstacle volume.

Decision: retain this as a real but sharply layout-dependent task-level
positive, not a robust avoidance solution. Do not continue scalar sweeps of
the same one-point cost. The next bounded representation hypothesis is
dual-view point-cloud fusion with an obstacle MVEE and gripper / carried-target
ellipsoids; route persistence and handoff remain separate temporal needs.
Canonical result:
`outputs/tabletop_guidance/counterfactual_rescue_5scene_5noise/conclusion.md`.

## 2026-08-13

### Increasing rho creates delayed divergence, not useful 3-D avoidance

Measured executed EEF trajectory separation rather than relying on contact
labels for the D2 hand-off XY point-guidance setting. Four fixed seed-812
states spanning persistent collision failure, residual finger contact, a
guidance-induced contact, and a formal collision-clear positive were swept at
`rho = 0.01 / 0.03 / 0.1 / 0.3 / 1.0`; episodes 0 and 4 additionally probed
`rho = 3 / 10`. H600, `Q=0.04 m`, the 1 cm Huber transition, and all finger
physics remained fixed.

Centimeter-scale separation occurs at ordinary and high rho, but mostly
hundreds of steps after guidance activates and after the first collision. Over
the main 20 episode/rho combinations, maximum separation in the first 64
executed steps after activation is only 2.39 mm. In episode 0, guidance starts
at step 48 and collision occurs at step 53; even `rho=10` changes the executed
trajectory by only 0.01 mm through collision. In episode 4, `rho=10` finally
creates 1.19 cm separation before collision, but the finger collision and task
outcome remain unchanged. The formal collision-clear episode 35 never reaches
1 cm separation at any `rho<=1`, confirming a contact-boundary change rather
than a visible bypass.

Task count remains 2/4 for baseline and every main-sweep rho, while collision
labels vary non-monotonically. Position clipping does not explain all large
late deviations: episode 0 at `rho=0.03` reaches 8.76 cm maximum separation
with zero position clips. The evidence instead supports small interventions
being amplified by the frozen closed loop after contact or task failure.

Decision: do not select a larger rho or run a formal matrix for the same XY
point cost. Rho alone can force late or, at extreme scale, locally visible
motion, but it does not generate an early, monotonic, task-preserving avoidance
route. Canonical report:
`outputs/lan_o3dp_hand_off_guidance/rho_trajectory_sweep_summary/report.md`.

## 2026-08-12

### Disabling the Panda hand shell helps execution but does not rescue 3-D point guidance

Completed a held-out four-environment 2x2 using the paper-aligned point-cloud
LAN-O3DP checkpoint: full physics versus only
`gripper0_right_hand_collision` disabled, crossed with baseline versus XY
Huberized-hinge guidance. The frozen setting is H600, `Q=0.04 m`,
`epsilon=0.01 m`, and `rho=0.01`; each cell has 50 seed-812 paired states and
the seed-705 calibration states are excluded. Finger collision meshes and pads
remain physically active.

Hand-off changes baseline Task SR from 0.50 to 0.62 in D2 and from 0.72 to
0.78 in D3. In D2, however, Safe SR remains 0.34: disabling the hand converts
collision failures primarily into successes that still contact through the
fingers, arm, or carried Can. D3 baseline Safe SR improves from 0.34 to 0.48
and CR falls from 0.58 to 0.44 under the modified physics.

The 2x2 interaction does not support the guidance hypothesis. With hand
collision disabled, guidance changes D2 Task/Safe SR only from 0.62/0.34 to
0.62/0.36 and D3 from 0.78/0.48 to 0.80/0.50. Every paired guidance
comparison has exact McNemar `p >= 0.5`. In full physics, hand CR is unchanged
by guidance in both D2 and D3; after hand-off, D2 finger CR rises from 0.32 to
0.36 and D3 finger CR stays 0.34. Guidance reduces some carried-target
contacts, but it does not supply swept-finger geometry or a persistent bypass
route.

Decision: retain hand-off as a mechanism diagnostic and modified-physics
upper bound, not as evidence that current 3-D XY point guidance succeeds. Do
not expand another blind `rho` / `Q` sweep of the same cost. Canonical report:
`outputs/lan_o3dp_hand_off_guidance/formal_seed812_n50_summary/report.md`.

### Strong long-horizon guidance succeeds in offset Empty2D

Promoted the post-hoc `rho=1, Q=0.12` Circle observation into a fixed paired
20-layout experiment with one continuous rollout and first-passage outcomes at
H400, H500, and H600. Deterministically mirrored obstacle offsets break the
head-on symmetry while keeping every direct start-to-goal segment obstructed.
At H600, both hinge and Huberized-hinge guidance reach Task/Safe SR 20/20 with
zero collision. The matched centered conditions reach only 13/20, with the
remaining seven trajectories ending as collision-free non-completions. The
offset guided conditions already reach 19/20 at H400; Huberization mainly
reduces reversal and backward-motion diagnostics rather than changing the H600
outcome.

Decision: retain the preregistered H320 paper-scale tables as their historical
negative result, while treating the H600 offset result as a verified mechanism
positive. Predicted-clean guidance can solve matched point geometry once scale,
horizon, and route symmetry permit a bypass. Its centered failures show that a
radial point cost still lacks stable route selection. Canonical report:
`outputs/empty2d_guidance/smooth_h600_circle/report.md`.

## 2026-08-10

### Empty2D guidance fails both idealized point-geometry gates

Completed all 20 preregistered calibration settings and 100 held-out paired
rollouts for each of Circle and Square using the seed-42 epoch-100 source. Every
calibration candidate and both selected held-out tests retain Task SR `1.00`,
Safe SR `0.00`, CR `1.00`, and NCR `0.00`. Both geometries have zero net
rescues, zero CR reduction, and exact McNemar `p=1.0`. Guidance triggers in the
formal sweep but does not remove any collision.

Post-hoc single-seed Circle diagnostics establish a scale transition outside
the formal grid: `rho=0.1, Q=0.12` visibly deflects the trajectory but still
collides; `rho=1, Q=0.12` remains collision-free but does not reach the goal
within H320. Its later extended-horizon interpretation is recorded in the
2026-08-12 entry above.

Decision at the frozen gate: close the formal paper-scale H320 sweep and do not
train seeds 43 or 44. This remains the historical preregistered result; the
post-hoc longer-horizon mechanism conclusion is recorded on 2026-08-12.
Canonical result: `outputs/empty2d_guidance/conclusion.md`.

## 2026-08-09

### Empty2D source gate passes and freezes epoch 100

The seed-42 sample-prediction source completed 500 epochs with all five formal
checkpoints uploaded to the required online W&B run. On fifty fixed D0
diffusion seeds, epoch 100 reaches 50/50 Task and Safe success with zero NCR,
collision, or action clipping. Later checkpoints reach 0.96 / 0.88 / 0.98 /
0.96 Task SR at epochs 200 / 300 / 400 / 500, so the preregistered Task SR,
NCR, validation-loss, and earlier-epoch ordering selects epoch 100.

Decision: the source gate passes, so proceed to independently calibrated
Circle and Square paired evaluations with epoch 100. This establishes only
source-domain policy health; it is not evidence that guidance works. Canonical
screen: `outputs/empty2d_guidance/source_screen.json`.

## 2026-08-07

### Empty2D replaces the old Toy prototype as the guidance mechanism test

Archived the paper-aligned LAN D2 evaluator and renderer after the fixed
ten-state study found zero collision-to-clear transitions at `Q=0.04` and at
expanded `Q` through 0.12. All four diagnosed contacts were between the Panda
gripper and Cereal, while the carried Can and arm had zero contacts. The
point-EEF cost was active but did not represent swept hand geometry.

Replaced the superseded `toy_recovery` package and artifacts with a minimal
`empty2d_guidance` experiment derived only from MPD's MIT-licensed Empty2D
geometry definitions. The policy sees current XY only and outputs absolute XY;
obstacles are available only to guidance as 256 true boundary points. The
fixed conditional-RBF dataset uses length scale 1.1 and the explicitly biased
amplitude 8. Its 90 training paths contain 33 Circle-safe and 20 Square-safe
counterfactuals, with both upper/lower routes and at least 57 collision paths
for either geometry. Every data gate passes. A real batch-256 CUDA update with
the formal 65,267,202-parameter sample-prediction model has finite loss,
reloads, and produces a finite online action.

Decision: proceed only after the 500-epoch seed-42 source passes D0. The
calibration/test protocol and seeds are frozen in code. A two-geometry pass
would validate the implementation in idealized point geometry and support,
but not prove, a swept-geometry explanation for D2. It cannot establish that
geometry is the unique D2 failure cause. Canonical index:
`outputs/empty2d_guidance/README.md`.

## 2026-07-31

### Absorbing-terminal source supersedes the first Toy source chain

The first corrected PointNav image policy used ResNet18 + SpatialSoftmax and
passed its calibration gate at D0/D1/D2 success `1.00 / 0.54 / 0.06`, but its
training windows repeated the final nonzero delta action after each episode.
Because every D0 demonstration moves toward positive X, this incorrectly
trained the policy to continue through the goal. Replaced only the
post-terminal targets with absorbing zero deltas and retrained from random
initialization on the byte-identical 200-demo D0 HDF5. The resulting source
passes the unchanged calibration gate at `1.00 / 0.56 / 0.14`; oracle SR
remains 1.00 and no screen or test layout was inspected.

The prior 150-rollout dynamics model passed action shuffle (`1.93x`) but lost
to latent persistence (`0.000779` versus `0.000716`). A separately requested
geometry-only screen on that same old source changed D0/D1/D2 SR from
`1.00/0.53/0.10` to `1.00/0.58/0.16`, with net paired transitions `0`, `+5`,
and `+6` per 100. It reduced collisions but missed the preregistered `+10/100`
gate. Both conclusions remain valid for the repeated-terminal checkpoint but
cannot be transferred to the new encoder.

Decision: retain the absorbing source and all five intermediate checkpoints.
Archive the repeated-terminal source, its failed dynamics gate, and its
geometry screen as historical pre-fix evidence; delete their raw rollouts and
models. The Toy funnel is no longer stopped by a current dynamics result: no
dynamics collection has yet been made from the absorbing source. Any future
latent comparison must recollect the fixed 150 rollouts; any geometry screen
must retune scale first. Canonical index:
`outputs/toy_recovery_objectives/README.md`.

### OOD success is not OOD elimination

The paper comparison exposed two shifts that had previously been conflated.
LAN-O3DP and DP3 primarily reduce **exogenous environment / observation OOD**
from obstacles, clutter, viewpoints, and instances. Guidance itself can create
**endogenous intervention-induced state OOD** by moving the robot away from
demonstrated task trajectories. Removing distractors from the policy input
does not make those post-intervention robot states part of the source training
distribution.

No reviewed paper demonstrates that OOD is eliminated. LPB tries to steer
future latent states toward the expert manifold; CAPE continues refining the
remaining trajectory under a bounded unseen-clutter family; ReGuide comes
closest to absorbing encountered OOD by training on successful guided recovery
rollouts. Unvisited future OOD remains in every case.

Decision: stop using “eliminate OOD” as an evaluation standard. Distinguish
representation invariance, robustness while OOD, steering back to an expert
manifold, and data aggregation. For this project, the operative test is whether
an intervention can maintain task-valid control until the frozen source is
empirically recoverable. Canonical synthesis: `papers/project-situation.md`.

## 2026-07-30

### Recoverability boundary archived for a controlled recovery-objective prototype

Closed the current manipulation stage without changing its implementation.
Frozen geometric guidance improves local clearance but does not establish
source-policy recovery. Correction training has a positive D1/H600 result
(Task SR `0.67 -> 0.78`) while full-policy mixing and single-chunk correction
remain insufficient in D2. Persistent correction establishes non-monotonic
safe-recoverability windows, but the existing SDP generator covers only 7/20
strict states versus the waypoint teacher's 18/20 upper bound.

Decision: retain the frozen source, SDP-6k, LAN-350, verified C1 correction
raw/pairs, and all audit/config/report artifacts. Remove superseded
checkpoints and duplicate/rejected correction data under the retention
manifest. Continue recovery-objective comparison in an independent PointNav
rapid prototype; it is mechanism isolation, not a replacement manipulation
claim. The next manipulation policy remains a correction-only temporally
extended option trained before a handoff estimator.

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
