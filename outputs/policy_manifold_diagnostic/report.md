# Policy-Manifold Feasibility Diagnostic

Date: 2026-07-27

## Question

Can phase activation be defined without hand-labelled task phases by asking
whether the frozen source policy currently contains a safe,
progress-preserving action direction?

This diagnostic treats that condition as an automatic gate. At each of 20
saved pre-target states, captured one eight-action chunk before a real
collision in `PickPlaceBreadCerealCan`, it:

1. draws 32 chunks from the frozen delta-EEF diffusion policy;
2. fits a regularized Gaussian metric to their 16-D XY waypoint trajectories;
3. defines policy support as the empirical 95% Mahalanobis ellipsoid; and
4. projects the nominal chunk subject to swept-clearance, target-progress,
   action-bound, and policy-support constraints.

The target-progress constraint allows at most 0.5 mm less progress than the
nominal chunk. No corrected action is executed.

## Results

| condition | states with a raw safe+progress sample | support-constrained projection |
|---|---:|---:|
| original object boundary | 12/20 | 13/20 |
| obstacle expanded by 1 cm, empirical 95% support | 0/20 | 1/20 |
| obstacle expanded by 1 cm, 2x squared support radius | 0/20 | 1/20 |
| obstacle expanded by 1 cm, support effectively removed | 0/20 | 14/20 |

At the 1 cm boundary, none of the 640 sampled chunks was both safe and
progress-preserving. The only default-support projection success was a
borderline state whose best raw sample already reached 0.993 cm clearance; the
projection moved a waypoint by at most 0.075 cm.

When the support constraint was effectively removed, the solver found 14
safe, progress-preserving paths. Their median maximum waypoint displacement
was 1.01 cm. The returned paths had a median squared Mahalanobis support ratio
of 8.20 relative to the empirical 95% threshold, corresponding to 2.86 times
its Mahalanobis radius. This ratio describes the returned control solutions;
it is not a proof of the globally minimum required radius.

## Decision

This is a negative gate for policy-support-based phase gating with the current
source policy. A state-dependent gate based on the policy's own support would
almost never activate at the collision states where intervention is needed.
The limiting factor is not simply the choice of a 95% threshold: doubling the
squared support radius adds no feasible states. Removing support recovers many
geometric solutions, so safe paths often exist but are not represented by the
local source-policy distribution.

Do not proceed to rollout implementation of this gate for the current
checkpoint. A useful next method must first create or supply avoidance support,
for example through obstacle-conditioned demonstrations, targeted action-space
augmentation, or a planner/residual allowed to leave the source-policy
manifold. A learned or automatic gate can then decide when to use that support.

## Scope and limitations

- This is a 20-state mechanism diagnostic, not a success-rate evaluation.
- Geometry uses oracle object positions and the existing swept XY EEF-point /
  object-circle proxy.
- The tested 1 cm expansion is favorable compared with a boundary sized to the
  widest part of the full gripper.
- Z, orientation, and gripper channels remain nominal.
- The support model is a local regularized Gaussian approximation from 32
  policy samples.
- Feasibility does not guarantee OSC tracking or rollout success; infeasibility
  here is specific to the selected collision regime and eight-step horizon.

See `fixed_state_d2/clearance_by_state.png` for the per-state clearance plot and
the three result directories for machine-readable state metrics.
