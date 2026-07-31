# Project Situation

This document describes the current research problem, negative results, and
open questions. It is the reference for paper-reading agents to evaluate
whether a new paper addresses a gap this project has already hit.

## Core Problem

Distillation of a single-task diffusion imitation policy to multi-distractor
environments, with a frozen epoch-260 delta-EEF source policy and an oracle
waypoint teacher that can provide corrective demonstrations.

**Environments**: PickPlaceCan (0 distractors) → PickPlaceBreadCan (1) →
BreadCerealCan (2) → BreadCerealMilkCan (3). D0 is source-competent (Task SR
0.907). D2 (Task SR 0.253) is the primary capacity boundary; D1 (0.707) is
the intermediate transfer test.

**Action interface**: executable 7-D delta EEF pose through `OSC_POSE`.

## What Is the Distribution Shift?

At deployment time, the policy encounters states where a distractor object
(new geometrical constraint) occludes the direct approach to the target. The
source policy was trained only on clean single-object demonstrations. Its
action distribution for these obstructed states either (a) produces actions
that collide with the distractor, or (b) diverges into non-task-completing
behavior. The shift is **geometric-constraint out-of-distribution**: the
observation space contains novel object configurations, and the required
collision-avoidance behavior lies outside the policy's effective support.

## OOD Terminology and Evidence Boundary (2026-07-31)

The phrase "eliminating OOD" is not a valid description of any reviewed
method. OOD is defined relative to a policy's training distribution: if an
intervention moves the closed-loop system into states not covered by that
distribution, those states remain OOD even when the controller happens to act
successfully in them. A method can reduce the observable shift, remain robust
under a specified shift, steer back toward the training manifold, or absorb
visited recovery states into an updated training set. None of these eliminates
all present or future OOD states.

Two different shifts must therefore be kept separate:

1. **Exogenous environment / observation OOD**: the scene changes independently
   of the policy, for example through unseen obstacles, clutter, viewpoints,
   object instances, language, or embodiment. LAN-O3DP and DP3 primarily test
   this kind of generalization. A target-only point cloud can remove distractor
   geometry from the policy observation and thereby reduce obstacle-induced
   observation shift. It does **not** make the robot states reached after an
   avoidance intervention part of the source demonstration distribution.
2. **Endogenous intervention-induced state OOD**: guidance or a corrective
   controller deliberately changes the source action, moves the robot away from
   demonstrated task trajectories, and subsequently queries the source policy
   at novel states. This is the failure exposed by our guided-denoising,
   single-chunk handoff, and persistent-correction experiments. Target-only
   perception cannot remove it because it is a shift in the robot/task state
   and required continuation behavior, not merely a distractor pixel or point
   cloud shift.

The reviewed papers occupy different points on this distinction:

| Paper | OOD evidence actually provided | Correct interpretation |
|---|---|---|
| [LAN-O3DP](lan-o3dp/paper.md), [DP3](dp3/paper.md) | Generalization to unseen obstacles, clutter, viewpoints, appearances, or instances | Representation reduces exogenous observation shift; it does not establish source-policy recovery after an action intervention. |
| [CAPE](cape/paper.md) | Obstacle-free training followed by evaluation in unseen clutter | Persistent suffix refinement expands avoidance modes and preserves task endpoints. It operates under a bounded OOD family rather than eliminating OOD or handing off immediately to an unchanged source policy. |
| [VLS](vls/paper.md), [Feedback World Model](feedback-world-model/paper.md) | Explicit observation/language or environment-distribution shifts | Closed-loop reward regeneration or feedback correction improves adaptation in the tested shifts; the underlying states do not thereby cease to be OOD. |
| [LPB](lpb/paper.md) | Inference perturbations and behavior-cloning covariate shift | Predicts future latent states and steers them toward the expert manifold. It attempts to prevent or reverse drift, not to eliminate OOD. |
| [ReGuide](reguide/paper.md) | Drifted-but-recoverable behavior-cloning states | Uses phase-conditioned recovery guidance and then trains on successful guided rollouts. Of the reviewed methods, this comes closest to *absorbing* encountered OOD: those visited recovery states can become covered by the updated policy, but unvisited future OOD remains. |

The strict literature conclusion is therefore:

> No reviewed paper demonstrates that OOD has been eliminated. Several papers
> demonstrate success on bounded OOD benchmarks; LPB and ReGuide most directly
> address intervention-induced covariate shift; only data aggregation or
> retraining can make specific encountered recovery states part of an updated
> training distribution.

This changes the standard used in this project. The relevant question is not
whether a method "eliminates OOD," but whether it can maintain task-valid
control after intervention until the system enters a state from which the
frozen source policy is empirically recoverable. Success in an unseen-obstacle
benchmark, or reduced obstacle content in the policy observation, is not
evidence of that handoff property. LAN-O3DP Figure 7 should be cited only as
evidence that continued closed-loop geometric guidance can succeed on some
unseen obstacle layouts, not as evidence that intervention-induced state OOD
was removed.

## Three Layers of Observed Failure

We have tested methods at three levels of intervention. Each level revealed a
distinct failure mode that localizes a deeper requirement.

### Layer 1 — Inference-Only Guidance (2026-07-27, NEGATIVE)

**What we tried**: guided denoising, multi-sample refinement, task-constrained
projection, expanded-boundary obstacle costs — all purely at deployment time
with the frozen source policy.

**What happened**: guidance can produce locally collision-free action chunks in
a given state. However, those modified actions push the closed-loop system
into states that lie outside the source policy's demonstration support. The
frozen source policy has never seen these states during training and does not
know how to complete the task from them. Local feasibility does not translate
to task success.

**Key numbers**: at 20 strict pre-collision D2 states, 0/640 policy samples
were both safe and progress-preserving at a 1 cm expanded boundary.
Support-constrained projection was feasible in 1/20 states.

**Conclusion**: inference-time changes to the action distribution are not
sufficient. The policy's effective support does not contain task-completing
avoidance trajectories. Training-side intervention is required.

### Layer 2 — Full-Policy Correction Training (2026-07-28 / 2026-07-29, NEGATIVE)

**What we tried**: positive-only BC fine-tuning (50/50 clean/correction),
set-supervised SDP (paper-aligned `r=0.1`, `N=16`, `K_A=16`), and lower-exposure
90/10 positive-BC control — all fine-tuned from the epoch-260 checkpoint.

**What happened**: correction training successfully injects local avoidance
support into the policy's action distribution. At fixed D2 states, strict 1 cm
safe-progress coverage rises from source 0/20 direct hits and 1/20 feasible
projections to 10/20 and 13/20. However, in closed-loop rollouts, the
fine-tuned policies fail to reliably decide **when to enter, sustain, and exit
the corrective mode**. They either avoid the distractor but lose task
completion (NCR rises) or remain in source-like collision trajectories (CR
persists). D0 stays near 1.00, so clean-skill forgetting is not the cause.

**Key numbers**: best D2 Task SR across all correction-training variants is
0.25 versus source 0.35. Positive BC produces 2 failure→success and 4
success→failure transitions. SDP: 2 failure→success, 4 success→failure.
Oracle-gated SDP selector: Task SR falls to 0.15 with 0 failure→success.

**Conclusion**: support injection ≠ reliable closed-loop mode switching. The
problem is not that the policy cannot produce safe actions — it can, locally.
The problem is that the policy does not know **when** to produce them and
**when to return control** to the task-completing behavior.

### Layer 3 — Single-Chunk Oracle Correction + Recovery (2026-07-29, NEGATIVE)

**What we tried**: an oracle branch-value upper bound. At a fixed
pre-collision state, execute one chunk of a corrective policy (SDP or
waypoint), then hand back to the frozen source policy for the remaining
episode. A hindsight oracle selects the branch that yields Task success.

**What happened**: on 3 held-out states, 96 corrective chunks all avoid
immediate contact. Their 96 full frozen-source continuations produce 32 NCRs
and 64 collision failures, with **zero Can contacts and zero Task successes**.
There is no successful branch for even an oracle to select. On the same 3
states, the waypoint teacher solves 3/3 without collision using 121--144
controlled steps before handoff.

**Conclusion**: a single corrective chunk (8 actions) is insufficient to
transition the system into a state where the source policy can complete the
task. The corrective generator must produce a temporally extended avoidance +
grasp + handoff-behavior. The necessary duration is on the order of tens of
actions, not a single chunk.

## Persistent Correction Boundary (2026-07-30)

**What we tried**: run the SDP-6k policy as a persistent corrective option,
replanned every 8 actions for up to 20 chunks, on all 20 held-out strict D2
states. Every collision-free prefix boundary was cloned and handed to the
frozen source under the fixed continuation protocol.

**Result**: 7/20 states contain at least one safe recoverable handoff. Within
those seven states, 27/28 correction trajectories contain a handoff; the
remaining thirteen states have 0/52. Five recoverable states require multiple
correction chunks, the median earliest state-level handoff is five chunks, and
no correction trajectory directly completes the task. Recoverability is
non-monotonic — a trajectory can enter, leave, and re-enter a safe handoff
window.

**Conclusion**: temporal persistence generates source-recoverable states that
a single-chunk candidate set does not contain. This supports the hypothesis
that correction must be a **temporally persistent option terminated by source
recoverability**. However, the current SDP generator covers only 7/20 states
even with an oracle handoff, well below the waypoint teacher's 18/20 upper
bound. A handoff estimator cannot repair missing corrective behavior. The next
manipulation method must first learn a correction-only, temporally extended
route generator while keeping the source frozen.

## D1 Transfer — Positive Result (2026-07-29)

SDP-6k in D1 at horizon 600: Task SR 0.78 vs source 0.67, 12 failure→success
vs 1 success→failure (p=0.0034). D1/H600 is the positive targeted-transfer
regime. D2 remains the capacity boundary.

## Summary: What Is the Gap?

| Method level | Can it avoid collision locally? | Can it preserve task completion? | Can it return control to source? |
|---|---|---|---|
| Inference guidance | Yes (local) | **No** — states leave support | N/A |
| Full-policy BC/SDP | Yes (local) | **No** — mode switching fails | N/A |
| Single-chunk + source recovery | Yes (local) | **No** — no successful branch exists | **No** |
| Persistent correction | Yes (multi-chunk) | **No** (direct) | **Yes in 7/20 states with oracle handoff** |

The remaining task is to build a corrective behavior generator that transitions
the system into a state where the frozen source policy can safely resume and
complete the task. Persistent SDP establishes that such states exist but
covers only 7/20 strict states. The next manipulation step is a dedicated,
temporally extended correction policy (object-relative waypoints or residual)
trained before any handoff gating model. A separate PointNav rapid prototype
may isolate recovery objectives, but it does not replace this manipulation
evidence.

## PointNav Rapid-Prototype Status (2026-07-31)

The active image-based PointNav source uses 200 D0 demonstrations and
absorbing zero-delta terminal padding. It passes the frozen D0/D1/D2
calibration gate at success `1.00 / 0.56 / 0.14`. This supersedes a
repeated-terminal source whose final nonzero delta was incorrectly copied into
post-episode action slots.

The old source's latent predictor passed action shuffle (`1.93x`) but failed to
beat latent persistence. Its geometry-only screen improved net paired
transitions by `+5/100` in D1 and `+6/100` in D2, below the preregistered
`+10/100` gate. Both results are archived as pre-fix evidence and cannot be
combined with the active encoder. No dynamics, guidance screen, or test rollout
has yet been run with the absorbing source. A new latent comparison must begin
with the fixed 150-rollout collection from the five retained source
checkpoints; geometry requires fresh scale selection. See
`../outputs/toy_recovery_objectives/README.md`.

## Available Tools

- **Frozen source policy**: epoch-260 delta-EEF Diffusion Policy
- **Oracle waypoint teacher**: discrete-yaw Cartesian waypoint search; 18/20
  collision-free Task success on D2 strict states
- **SDP-6k checkpoint**: trained SDP policy from the Layer 2 comparison
- **Paired correction dataset**: 2,849 (O, A-, A+) pairs from 25 verified D2
  episodes
- **Metrics**: Task SR, Safe SR, CR (collision rate), NCR (non-completing safe
  rate), four-way outcome partition
- **Fixed evaluation banks**: seed-600 paired D2 layouts; seed-601/602
  untouched D1/D3 layouts; 20 fixed strict pre-collision D2 states for
  diagnostic

## Questions for Paper Evaluation

When reading a new paper, answer these questions against the project situation
above:

1. **What distribution shift does this paper address?** Is it a
   geometric-constraint shift in the observation/state space, a different
   embodiment, a goal generalization, etc.? Explicitly distinguish exogenous
   environment / observation OOD from endogenous intervention-induced state
   OOD. Do not treat success on the former as evidence that the latter has
   been removed.

2. **What supervision does it add?** Corrections, ranking labels, a world
   model, human feedback, constraint specifications, etc. Does it require an
   oracle teacher at training time, deployment time, both, or neither?

3. **What must hold for the method to work?** E.g., the policy's support must
   already contain task-completing avoidance behavior; the distribution shift
   must be resolved within a single action chunk; a handoff criterion must be
   known; the world model must be accurate over the required horizon; etc.

4. **Does our environment violate any of these assumptions?** Check against the
   three layers of observed failure above. For example: if a method assumes
   single-chunk correction is sufficient, our Layer 3 result contradicts it.

5. **What does the paper mean by OOD success?** Determine whether it reduces a
   representation shift, acts robustly while still OOD, steers back toward an
   expert manifold, or retrains on encountered recovery states. Do not describe
   any of these as eliminating OOD without direct distributional evidence.

6. **If applied, what observable failure would we expect?** Based on the
   assumption that is violated, predict: (a) local avoidance succeeds but task
   fails; (b) mode switching fails; (c) source recovery fails; (d) something
   else. Map to our existing four-way partition (Task SR / Safe SR / CR / NCR).

7. **What is the minimal decisive experiment?** Define the smallest evaluation
   that would falsify the method in our setting: which environment, which
   metric, which comparison baseline, and what threshold gate.

## Decision Log Quick Reference

| Date | Decision |
|---|---|
| 2026-07-27 | Stop frozen-policy inference-only methods |
| 2026-07-28 | Reject 50/50 positive-only BC |
| 2026-07-29 | Reject paper-aligned SDP (r=0.1, N=16) |
| 2026-07-29 | Reject 90/10 lower-exposure BC control |
| 2026-07-29 | Reject oracle-gated SDP selector |
| 2026-07-29 | Reject oracle branch-value world model |
| 2026-07-29 | D1 SDP positive at horizon 600 |
| 2026-07-30 | Persistent correction expansion: 7/20-state oracle coverage |
| 2026-07-30 | Freeze current manipulation boundary; prototype recovery objectives in PointNav |
