# Targeted Fine-Tuning Upper-Bound Plan

Status: positive BC, lower-exposure control, and SDP successor completed; all
miss the closed-loop gate, 2026-07-29.

Archive note, 2026-07-30: the selected SDP-6k checkpoint subsequently produced
a positive D1 result at the equally applied 600-step horizon (Task SR
`0.67 -> 0.78`) but remained capacity-limited in D2. Persistent use created
oracle safe-recoverability windows in 7/20 strict D2 states, below the waypoint
teacher's 18/20. This plan is closed; do not resume full-policy mixing or
single-chunk correction sweeps.

## 1. Question

Can a small number of successful, targeted multi-object corrections inject
task-preserving avoidance behavior into the reliable epoch-260
`delta_eef_pose_action` Diffusion Policy while preserving its clean
single-object skill?

This is a learnability upper bound. Full policy weights may change. A positive
result establishes that the transfer gap can be repaired with targeted action
support; it does not by itself define the final paper method.

## 2. Fixed Starting Point

- source checkpoint:
  `outputs/robomimic/train/dp_can_delta_pose_osc/20260707222943/models/model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth`
- observations: two RGB cameras plus current proprioception, unchanged;
- action: executable 7-D world-frame `delta_eef_pose_action`;
- horizons: observation / prediction / execution = `2 / 16 / 8`;
- scheduler: DDIM, 100 training timesteps and 10 inference timesteps;
- source evaluation: Task SR 0.907 / 0.707 / 0.253 / 0.200 over 0/1/2/3
  distractors;
- mandatory rollout metrics: Task SR, Safe SR, CR, NCR, and the four-way
  outcome partition.

The original 200 clean demonstrations remain the rehearsal dataset. Do not
replace them with correction-only training.

## 3. Correction-Data Contract

Use `PickPlaceBreadCerealCan`, the two-distractor environment, as the primary
adaptation domain. It is the lowest-clutter setting with a large source-policy
collapse (Task SR 0.253, CR 0.647), while the waypoint teacher still provides
a feasible correction upper bound. Treat `PickPlaceCan` as source-skill
retention, the one-distractor environment as lower-clutter interpolation, and
the three-distractor environment as held-out higher-clutter extrapolation.

Collect complete successful hybrid episodes as the replay and quality-control
layer:

1. start from a fresh multi-object training layout;
2. let the frozen epoch-260 policy act normally;
3. trigger only in the selected
   `pre-target + target-directed + predicted-obstruction` regime;
4. clone the simulator state and test waypoint yaw candidates;
5. select the smallest safe yaw;
6. restore the original state and execute only the selected waypoint route;
7. grasp and lift Can;
8. hand control back to the frozen policy for placement;
9. save the episode for training only when the complete task succeeds and the
   complete selected branch has no distractor contact, including before the
   trigger and after policy handoff.

Candidate-search branches are diagnostic queries, not demonstrations. Save
only the final continuous selected branch.

During every teacher-controlled step, query but do not execute the frozen
policy's first freshly replanned action at the same two-frame observation.
Record these robot actions together with the realized teacher actions, then
derive SDP-style paired windows

$$
(\mathcal O_t,A_t^-,A_t^+).
$$

The full raw schema, exact `2 / 16 / 8` time alignment, pair-extraction rule,
and invariants are fixed in
[`paired_correction_data.md`](paired_correction_data.md). In particular, a
single 16-step policy chunk queried only at the trigger is not a valid negative
after the teacher changes the visited states.

Required raw-episode contents include:

- simulator states and environment model metadata sufficient for exact replay;
- policy observations and next observations;
- `delta_eef_pose_action`, reward, done, and success;
- exact executed commands and a controller / intervention label per action;
- frozen-policy query action, validity flag, and diffusion seed at every
  teacher step;
- trigger step, intervention interval, selected yaw, and teacher configuration;
- target and distractor contact/collision metadata;
- collection seed and policy/checkpoint identity.

Use fresh training seeds. Keep all seeds and states used by the strict
20-state diagnostic bank held out; in particular, do not collect training
corrections from seeds 600--607.

## 4. Collection Stages

### C0: ten-episode format gate

Collect 10 successful episodes, then require:

- exact initial simulator-state restoration followed by open-loop command
  replay with bounded EEF / gripper trajectory error;
- recomputed delta-EEF actions match saved actions;
- observations have the same keys, shapes, dtypes, and camera conventions as
  the clean dataset;
- every episode succeeds when replayed open loop under the recorded controller;
- every train-eligible selected branch is collision-free, and its intervention
  metadata and contact labels reproduce;
- no candidate-search branch appears in the saved trajectory.
- pair extraction exactly reproduces robomimic's two-observation, 16-action,
  and executed `1:9` alignment;
- per-step frozen-policy queries do not mutate simulator, observation-history,
  or executed-policy state;
- all paired windows satisfy the contiguous-intervention and shared-history
  invariants in the paired-correction data contract.

Do not start training if this gate fails.

### C1: small upper-bound dataset

After C0 passes, collect approximately 25 successful correction episodes in
total. Record attempted layouts and rejected episodes separately so collection
success is auditable, but train only on the successful selected branches.

Use a fixed train/validation assignment decided before training. Held-out
closed-loop rollout layouts, rather than demonstration loss alone, decide the
experiment.

## 5. Action-Normalization Gate

The source policy uses checkpointed min-max action normalization. Before
training, normalize every correction action using the epoch-260 statistics and
report per dimension:

- minimum and maximum normalized action;
- number and fraction outside `[-1, 1]`;
- maximum absolute excess;
- comparison with the original clean train split.

The primary dataset must have no material out-of-range correction actions.
Reduce teacher translation or rotation increments and recollect if needed.
Do not silently widen min-max ranges, because that changes the action
coordinate system learned by the source policy.

If every correction action stays inside the original clean extrema, combining
the original clean dataset with correction data leaves min-max normalization
unchanged. Assert this equality before the first optimization step and store
the statistics in every new checkpoint.

## 6. Training Comparisons

All learned variants initialize from the same epoch-260 checkpoint:

1. **Frozen source** -- no additional optimization.
2. **Clean-only continuation** -- continue training on the original clean data
   for the same number of gradient updates as the treatment.
3. **Targeted mixed fine-tuning** -- train on clean replay plus correction
   pairs, using the positive action chunk as the ordinary diffusion target.

The treatment uses the ordinary Diffusion Policy noise-prediction loss. No
reward model, Q function, world model, teacher mask input, or deployment-time
oracle is added to the policy.

For the initial upper bound:

- update the full observation encoder and diffusion UNet;
- use a fresh optimizer;
- initial learning rate `1e-5`;
- batch size 64;
- approximately 10,000 gradient updates;
- save checkpoints every 1,000--2,000 updates;
- evaluate EMA weights.

These are starting values, not a license for a broad sweep. Inspect checkpoint
trajectories before changing the learning rate or update budget.

## 7. Sampling

Keep the clean, raw-correction, and derived pair HDF5 files separate. The
initial implementation uses balanced clean and paired-positive batches:

```text
50% original clean sequences
50% paired-correction sequences using positive_actions
```

Every correction-pair sample is intervention-centered by construction, so no
intervention mask or intervention-aware temporal sampler is used. Overlapping
windows are allowed, but split and report data by episode and layout rather
than treating windows as independent collection units.

## 8. Evaluation and Advance Gate

Training loss and demonstration validation loss are diagnostics only. Evaluate
saved checkpoints on:

1. the held-out strict 20-state support diagnostic;
2. clean `PickPlaceCan`;
3. the one-distractor environment as lower-clutter interpolation;
4. paired held-out two-distractor layouts as in-distribution adaptation;
5. the unchanged three-distractor environment as higher-clutter extrapolation.

The initial advance gate is:

- at least 8/20 held-out states admit a safe, progress-preserving action inside
  the updated empirical support proxy; also report direct policy-sample hit
  rate;
- clean Task SR remains approximately at or above 0.85;
- paired two-distractor evaluation contains at least four
  failure-to-success transitions and at most one success-to-failure transition;
- NCR does not increase as a substitute for collision reduction;
- the same checkpoint and training configuration produce a positive
  three-distractor effect;
- no result depends on runtime action clipping.

Select the checkpoint that maximizes held-out multi-object Task SR subject to
the clean-preservation and failure-mode constraints. Do not select by training
loss.

## 9. Interpretation

If the upper bound passes, targeted corrective support is sufficient in
principle. The immediate next comparison is SDP set supervision on the same
fixed $(\mathcal O,A^-,A^+)$ pairs. Only after separating the benefit of
paired-positive BC from the benefit of set supervision should the project
decide whether to use full fine-tuning, an adapter, a planner / residual
controller, or another architecture.

If it fails despite verified data, balanced exposure, and stable clean
behavior, investigate perception, temporal structure, policy capacity, or
teacher consistency before collecting substantially more demonstrations.

Do not claim that the waypoint teacher is a deployment method, that 25
corrections solve general multi-object transfer, or that world models are
unnecessary.

## 10. Completed Initial Result

The 50/50 paired-positive BC treatment fails the closed-loop development gate
at every inspected checkpoint from 2,000 through 10,000 updates. Its best D2
Task SR is 0.25 versus 0.35 for both source and the update-matched clean-only
control. The 10,000-update model has two failure-to-success and four
success-to-failure transitions, while D0 remains 1.00 and no action clipping
occurs.

This is not a failure to inject action support. At the strict 1 cm fixed-state
margin, direct safe-progress sample coverage changes from source / clean-only
0/20 to mixed 10/20, and support-projection feasibility changes from 1/20 to
13/20. The evidence instead points to unstable mode preservation or selection
under positive-only correction BC.

Decision: do not advance this treatment to the untouched D1/D3 final matrix
and do not collect more identical corrections. Reuse the same immutable pair
file for an SDP-style correction objective while retaining ordinary BC for
clean rehearsal. See
`outputs/targeted_finetuning/upper_bound_report.md`.

## 11. Completed Successor Result

The preregistered SDP successor and 90/10 positive-BC exposure control also
fail every inspected D2 checkpoint. Their best Task SR is 0.25 versus source
0.35, while D0 stays between 0.95 and 1.00. SDP at 6,000 updates nevertheless
retains 10/20 direct safe-progress states and 11/20 support-feasible states.

## 12. Completed Localization Result

The source-default oracle-gated successor also fails. A strict selector found
no SDP sample in 20/20 states that both cleared the 1 cm boundary and matched
the source chunk's target progress. A no-substantial-retreat relaxation found
eligible samples in 11/20 states and was advanced to paired rollout.

On 20 paired D2 layouts, the selector changed Task SR / CR / NCR from
0.25 / 0.80 / 0.05 to 0.15 / 0.75 / 0.15, with zero
failure-to-success and two success-to-NCR transitions. Thus explicitly
localizing the SDP policy does not solve task-preserving entry and recovery.
Do not tune or scale this selector. See
`outputs/gated_support_selector/report.md`.

## 13. Completed Oracle Branch-Value Gate

Before fitting a learned world/value model, source and 32 SDP chunks were
branched through the real simulator on three held-out pre-collision states.
Every candidate executed for eight steps and then handed control to the
frozen source until the original episode horizon.

All 96 corrective chunks avoided immediate physical contact, but their full
continuations yielded 32 NCRs and 64 collision failures. None contacted Can
or completed the task, so even a hindsight oracle had no successful branch
to select. The same three states remain solvable: the explicit waypoint
teacher reaches 3/3 collision-free Task success with 121--144 controlled
steps.

Decision: stop the 20-state expansion and do not train a model that only ranks
the current one-chunk SDP candidates. A successor must generate or represent
multi-step corrective behavior, not merely score the existing local samples.
See `outputs/oracle_branch_value/pilot_3state/result_report.md`.

This rules out correction exposure and pointwise positive targets as complete
explanations. The active issue is now explicit localization and handoff of the
corrective mode. A next upper bound should freeze the source and oracle-gate a
small residual or avoidance expert before learning the gate. See
`outputs/targeted_finetuning/sdp_result_report.md`.
