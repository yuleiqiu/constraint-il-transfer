# Frozen-Policy Inference Limitations

Status: stage conclusion, 2026-07-27.

## Conclusion

For the current epoch-260 clean-image delta-EEF source policy, the tested
deployment-time methods do not provide reliable single-to-multi transfer while
the policy remains frozen. Oracle obstacle information can change predicted
and executed trajectories and improve local clearance, but it does not create
task-preserving avoidance modes that are absent from the learned action
distribution.

This is a conclusion about the current checkpoint and tested inference-only
method family, not a proof that every possible frozen-policy planner is
mathematically incapable of solving the task.

## Progressive evidence

1. **The source policy and action interface are healthy.** On
   `PickPlaceCan`, the policy reaches Task SR 0.907. Delta-EEF chunks reconstruct
   executed trajectories with sub-centimeter mean error and no observed action
   clipping.
2. **There is a substantial single-to-multi gap.** Task SR falls to 0.253 /
   0.200 with two / three distractors, while CR rises to 0.647 / 0.680.
3. **Inference information reaches the robot.** LAN-style oracle-geometry
   guidance improves predicted and short-prefix executed clearance in fixed
   states. Its paired rollout effects are nevertheless inconsistent across
   clutter levels and do not pass the scaling gate.
4. **More inference-time computation does not resolve the gap.** Four-candidate
   sampling plus masked refinement leaves all 20 paired task outcomes unchanged
   and adds 76% inference time. An oracle task-constrained optimizer recovers
   two task successes among nine feasible layouts, but all nine corrected
   continuations still collide. In the easier one-distractor environment, a
   5 cm expanded-boundary guide removes the only baseline collision while Task
   SR falls 0.70 to 0.65 and NCR rises 0.25 to 0.35.
5. **The action support is the limiting factor in the selected collision
   regime.** At 20 saved pre-collision states, none of 640 policy samples is
   both progress-preserving and safe at a 1 cm expanded boundary. A 95%
   policy-support projection is feasible in only 1/20 states, unchanged after
   doubling its squared radius. When the support constraint is effectively
   removed, geometric solutions are found in 14/20 states.
6. **A single local geometry objective cannot cover all failures.** The
   600-episode intent diagnostic finds a mixed partition of
   target-directed-obstructed, non-target-directed, and target-directed-clear
   failures, with no dominant class in both hard environments.

The evidence chain is therefore:

```text
competent single-object policy and valid action interface
    -> large degradation under distractors
    -> oracle inference information changes local motion
    -> local clearance improves without reliable task-level gains
    -> safe paths often exist outside the source-policy support
    -> the frozen policy lacks task-preserving avoidance behavior
```

## Implication

Further scale, margin, or denoising-schedule tuning of the same local
point-clearance guidance is not justified. The next method should first create
or supply avoidance action support, for example through targeted training-side
adaptation or a planner / residual controller that is explicitly allowed to
leave the source-policy manifold. A gate can then decide when to use that
capability; gating alone cannot create it.

## Canonical evidence

- `outputs/eef_pose_osc_policy/multienv_eval_report.md`
- `docs/guided_denoising/lan_o3dp_mechanism_baseline.md`
- `docs/guided_denoising/sample_refine_pilot.md`
- `outputs/task_constrained_guidance/stage1_report.md`
- `outputs/guided_denoising/breadcan_expanded_5cm_paired_pilot/report.md`
- `outputs/policy_manifold_diagnostic/report.md`
- `outputs/guided_denoising/intent_diagnostic/report.md`
