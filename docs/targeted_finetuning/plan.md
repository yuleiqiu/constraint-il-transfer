# Targeted Fine-Tuning Upper-Bound Plan

Status: active experiment contract, 2026-07-28.

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

Collect complete successful hybrid episodes:

1. start from a fresh multi-object training layout;
2. let the frozen epoch-260 policy act normally;
3. trigger only in the selected
   `pre-target + target-directed + predicted-obstruction` regime;
4. clone the simulator state and test waypoint yaw candidates;
5. select the smallest safe yaw;
6. restore the original state and execute only the selected waypoint route;
7. grasp and lift Can;
8. hand control back to the frozen policy for placement;
9. save the episode only when the complete task succeeds.

Candidate-search branches are diagnostic queries, not demonstrations. Save
only the final continuous selected branch.

Required per-episode contents:

- simulator states and environment model metadata sufficient for exact replay;
- policy observations and next observations;
- `delta_eef_pose_action`, reward, done, and success;
- `intervention_mask` aligned with every saved action;
- trigger step, intervention interval, selected yaw, and teacher configuration;
- target and distractor contact/collision metadata;
- collection seed and policy/checkpoint identity.

Use fresh training seeds. Keep all seeds and states used by the strict
20-state diagnostic bank held out; in particular, do not collect training
corrections from seeds 600--607.

## 4. Collection Stages

### C0: ten-episode format gate

Collect 10 successful episodes, then require:

- exact simulator-state replay;
- recomputed delta-EEF actions match saved actions;
- observations have the same keys, shapes, dtypes, and camera conventions as
  the clean dataset;
- every episode succeeds when replayed open loop under the recorded controller;
- intervention metadata and contact labels reproduce;
- no candidate-search branch appears in the saved trajectory.

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
   data.

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

Keep the clean and correction HDF5 files separate. The initial implementation
uses dataset-level balanced replay:

```text
50% original clean sequences
50% successful correction-episode sequences
```

In robomimic, set `normalize_weights_by_ds_size=true` and use equal dataset
weights so the ratio is not determined by dataset length.

Before training, calculate the expected fraction of sampled sequences whose
executed action slice overlaps `intervention_mask`. If fewer than roughly 25%
of treatment samples expose an intervention, add an intervention-aware sampler
with:

```text
50% clean sequences
20% correction context / handoff sequences
30% intervention-overlapping sequences
```

A sequence is intervention-overlapping when the deployment-executed portion of
its action target intersects the saved intervention interval. Do not condition
the policy on the intervention mask.

## 8. Evaluation and Advance Gate

Training loss and demonstration validation loss are diagnostics only. Evaluate
saved checkpoints on:

1. the held-out strict 20-state support diagnostic;
2. clean `PickPlaceCan`;
3. paired two-distractor rollouts;
4. the unchanged three-distractor environment for confirmation.

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
principle. The next research decision is where to store or supply that
knowledge: full fine-tuning, a small adapter, a world model, a residual
controller, or a hybrid gate.

If it fails despite verified data, balanced exposure, and stable clean
behavior, investigate perception, temporal structure, policy capacity, or
teacher consistency before collecting substantially more demonstrations.

Do not claim that the waypoint teacher is a deployment method, that 25
corrections solve general multi-object transfer, or that world models are
unnecessary.
