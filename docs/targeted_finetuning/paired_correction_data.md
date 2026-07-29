# Paired-Correction Data Contract

Status: active design contract, 2026-07-28.

## 1. Purpose

Represent each accepted waypoint-teacher intervention as paired supervision

$$
(\mathcal O_t, A_t^-, A_t^+),
$$

where $A_t^-$ contains actions queried from the frozen source policy and
$A_t^+$ contains the actions realized under the selected teacher route. The
same pair file must support:

1. the initial positive-correction BC upper bound, which trains on $A_t^+$;
2. a later SDP comparison, which uses both $A_t^-$ and $A_t^+$.

The original clean HDF5 remains unchanged. Full accepted episodes are retained
as an append-only replay and audit layer; overlapping correction pairs are a
deterministic derived dataset rather than independently managed
demonstrations. The representation follows the online intervention loop in
[`papers/sdp/paper.md`](../../papers/sdp/paper.md), with the time indexing
adapted to this repository's robomimic implementation.

## 2. Source-Policy Time Convention

For a raw transition

$$
o_k \xrightarrow{u_k} o_{k+1},
$$

robomimic `SequenceDataset` with `frame_stack=2` and `seq_length=16` returns,
at sequence index $t$,

$$
\mathcal O_t=[o_{t-1},o_t],
\qquad
A_t=[u_{t-1},u_t,\ldots,u_{t+14}].
$$

The Diffusion Policy conditions on the two observations, predicts all 16
actions, and executes prediction indices `1:9`, corresponding to
$u_t,\ldots,u_{t+7}$. Prediction index 0 is the already executed previous
action. This convention follows directly from
`SequenceDataset.get_sequence_from_demo` and
`DiffusionPolicyUNet._get_action_trajectory`; the correction extractor must
not introduce a different alignment.

## 3. Online Collection Semantics

Use fresh `PickPlaceBreadCerealCan` layouts and keep seeds 600--607 and every
state in the strict 20-state bank held out.

At each accepted intervention:

1. the frozen epoch-260 policy runs until the preregistered
   `pre-target + target-directed + predicted-obstruction` trigger;
2. yaw candidates are evaluated from independent copies of the trigger state;
3. the original state is restored and only the selected safe route is
   executed;
4. at every teacher-controlled step $k$, query the frozen policy
   *without executing its output* using the current two-frame observation
   history and save its first executable action as $u_k^-$;
5. execute the teacher command, save the command for replay, and recompute the
   positive `delta_eef_pose_action` $u_k^+$ from the realized
   `obs[k] -> next_obs[k]` transition, using the executed gripper command;
6. continue the intervention for at least 16 consecutive steps, then complete
   the grasp-and-lift route and hand control back to the frozen policy;
7. mark the episode train-eligible only if the selected teacher interval is a
   safe pick, the complete selected branch has no distractor contact, and the
   complete PickPlace task succeeds. This whole-branch rule also rejects
   triggers reached only after an earlier collision.

The robot query is repeated at each teacher state. A single open-loop chunk
queried at the trigger is not a valid negative sequence after the teacher has
changed the state distribution. Each query uses a recorded diffusion seed and
fresh replanning, bypassing the rollout action queue; querying must not mutate
the executed policy or environment state. As in SDP, a derived pair later
conditions the whole chunk only on its starting observation history even
though its future single-step pairs were collected along the teacher
trajectory; report this action-chunk approximation as a method limitation.

## 4. Raw Episode File

Use one append-only `correction_raw.hdf5`. It is robomimic-compatible at the
transition level and adds intervention/query metadata.

```text
/
├── attrs
│   ├── schema_version = "paired-correction-raw-v1"
│   ├── source_checkpoint
│   ├── source_checkpoint_sha256
│   ├── action_key = "delta_eef_pose_action"
│   ├── observation_horizon = 2
│   ├── prediction_horizon = 16
│   └── action_horizon = 8
└── data/
    ├── attrs
    │   ├── env_args
    │   └── total
    └── demo_<episode_id>/
        ├── attrs
        │   ├── num_samples
        │   ├── model_file
        │   ├── ep_meta
        │   ├── env_seed
        │   ├── layout_id
        │   ├── task_success
        │   ├── selected_teacher_safe_pick
        │   ├── train_eligible
        │   ├── selected_yaw_deg
        │   └── teacher_config_json
        ├── states
        ├── obs/<policy keys + robot0_eef_quat_site>
        ├── next_obs/<policy keys + robot0_eef_quat_site>
        ├── actions
        ├── delta_eef_pose_action
        ├── rewards
        ├── dones
        ├── controller
        ├── intervention_id
        ├── robot_query_action
        ├── robot_query_valid
        ├── robot_query_seed
        ├── target_contact
        └── distractor_contact
```

`actions[k]` is the exact command sent to the environment and is used for
replay. `delta_eef_pose_action[k]` follows the clean dataset definition:
realized world-frame EEF position delta, realized site-quaternion rotation
delta, and the executed gripper command. `controller` is an integer enum
(`0=policy`, `1=teacher`). `robot_query_action` is meaningful only where
`robot_query_valid` is true.

Candidate-search traces must not appear under `data/`. Attempted layouts,
rejected routes, and compact candidate summaries belong in a separate
append-only collection log keyed by `layout_id`.

## 5. Derived Pair File

Generate `correction_pairs.hdf5` deterministically from one immutable raw-file
fingerprint. Store raw, unnormalized actions; apply the epoch-260 checkpoint
normalization at load time.

```text
/
├── attrs
│   ├── schema_version = "paired-correction-pairs-v1"
│   ├── raw_file_sha256
│   ├── source_checkpoint_sha256
│   ├── source_action_stats_sha256
│   ├── observation_horizon = 2
│   ├── prediction_horizon = 16
│   └── action_horizon = 8
├── pairs/
│   ├── obs/<policy observation keys>       [P, 2, ...]
│   ├── negative_actions                    [P, 16, 7]
│   ├── positive_actions                    [P, 16, 7]
│   ├── episode_id                          [P]
│   ├── anchor_step                         [P]
│   ├── intervention_id                     [P]
│   ├── layout_id                           [P]
│   └── selected_yaw_deg                    [P]
└── mask/
    ├── train_episode_ids
    └── valid_episode_ids
```

For an anchor $t$, construct:

$$
A_t^+
=
[u_{t-1}^{\mathrm{hist}},
  u_t^+,\ldots,u_{t+14}^+],
$$

$$
A_t^-
=
[u_{t-1}^{\mathrm{hist}},
  u_t^-,\ldots,u_{t+14}^-].
$$

Here $u_{t-1}^{\mathrm{hist}}$ is the source-compatible realized
`delta_eef_pose_action[t-1]`, not the raw controller command. The shared first
action is historical context, not a correction target. An anchor is eligible
only when:

- the episode is train-eligible;
- transition $t-1$ exists;
- every transition $t,\ldots,t+14$ belongs to the same contiguous teacher
  interval;
- all 15 robot queries are valid;
- no sequence padding is required.

The collector guarantees at least 16 teacher steps, although the project
alignment requires 15 current/future corrected actions per pair. Multiple
overlapping windows are allowed, but train/validation assignment is by
episode and layout before extraction. Never split overlapping windows from
one episode across train and validation.

## 6. C0 Verification Gate

After 10 successful accepted episodes, stop collection and require:

1. an index-alignment test on an existing clean demo exactly reproduces
   robomimic's two observations, 16 actions, and executed `1:9` slice;
2. every train-eligible raw episode is collision-free in its recorded contact
   labels, exactly restores its initial simulator state, then
   open-loop replays the recorded commands with bounded EEF / gripper
   trajectory error and reproduces episode-level task success and contact
   outcomes; bitwise intermediate-state or rendered-pixel equality is not
   required because OSC desired-goal state is not contained in MuJoCo's
   flattened simulator state;
3. recomputed positive actions match saved `delta_eef_pose_action`;
4. robot queries leave simulator state, observation history, and executed
   policy state unchanged;
5. every derived pair satisfies the shared-history action and contiguous
   intervention invariants;
6. policy observation keys plus `robot0_eef_quat_site`, their shapes, dtypes,
   camera orientation, and controller metadata match the clean dataset; unused
   environment-specific object-state vectors are not part of this contract;
7. positive and negative actions are finite and their normalized per-dimension
   ranges are reported using the epoch-260 statistics;
8. raw and pair fingerprints, source checkpoint identity, seeds, and
   train/validation assignment reproduce exactly.

Do not train or collect the remaining approximately 15 accepted episodes
until this gate passes.

## 7. Training Interface

The initial upper bound draws balanced clean and correction batches:

```text
50% original clean SequenceDataset samples
50% paired-correction samples using positive_actions
```

The positive-BC batch presented to the existing Diffusion Policy has the same
`obs` and normalized `actions` tensors as a normal robomimic batch. No
intervention mask or intervention-aware temporal sampler is needed.

If positive BC passes the task-level gate, reuse the identical pair file for
SDP. The method comparison then changes only the supervision applied to
correction pairs, not the collected states, splits, or teacher actions.
