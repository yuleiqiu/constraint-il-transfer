# Targeted paired-correction collection

This directory implements the pair-first correction-data contract in
`docs/targeted_finetuning/paired_correction_data.md`.

The collector runs the frozen epoch-260 policy on fresh
`PickPlaceBreadCerealCan` layouts, triggers only in the preregistered
pre-target / target-directed / predicted-obstruction regime, searches the
waypoint teacher's yaw from cloned states, and records only the selected
continuous branch. During every teacher step it freshly queries but does not
execute the frozen policy's first action.

## CPU tests

```bash
uv run python \
  scripts/2026-07-28_targeted_correction/test_paired_correction.py
uv run python \
  scripts/2026-07-27_semi_manual_waypoint_teacher/test_waypoint_teacher.py
```

The tests include an exact comparison with robomimic `SequenceDataset`:
`[o[t-1], o[t]]`, `[a[t-1], ..., a[t+14]]`, and executed slice `1:9`.

## Collection

The fixed C0 configuration uses a maximum yaw increment of 0.015 rad. Do not
reuse the waypoint diagnostic's historical 0.04-rad value: it exceeds the
epoch-260 rotation-z normalization range.

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-07-28_targeted_correction/collect_corrections.py \
  --accepted-episodes 10
```

Resume from a new layout index without overwriting the append-only raw file:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-07-28_targeted_correction/collect_corrections.py \
  --seeds 700 \
  --episode-start 28 \
  --accepted-episodes 25
```

Seeds 600--607 are rejected by the collector because they belong to the held-out
diagnostic bank.

## Extraction and verification

```bash
uv run python \
  scripts/2026-07-28_targeted_correction/extract_correction_pairs.py \
  --overwrite

MUJOCO_GL=egl uv run python \
  scripts/2026-07-28_targeted_correction/verify_correction_data.py \
  --replay \
  --out outputs/targeted_finetuning/verification.json
```

Verification checks:

- raw and pair schema / checkpoint fingerprints;
- deterministic raw-to-pair reconstruction;
- source normalization ranges for both positive and negative chunks;
- observation keys, shapes, and dtypes;
- contiguous intervention and shared-history action invariants;
- exact initial simulator-state restoration;
- bounded open-loop EEF / orientation / gripper replay error;
- episode-level target / distractor contact and final task success.

Raw HDF5 and JSONL artifacts are ignored by Git. Verification JSON and
human-readable reports remain trackable.

## C0 result

The corrected 10-episode C0 gate passed. Ten train-eligible episodes yield
1,135 pairs (1,024 train / 111 validation), with no positive or negative
action outside the source normalization range. One separately retained raw
episode failed open-loop task-success replay and is marked ineligible. See
`outputs/targeted_finetuning/c0_report.md`.

## C1 result

The final C1 gate passed with 25 train-eligible, successful, whole-branch
collision-free episodes and 2,849 pairs (2,120 train / 729 validation).
All eligible episodes pass replay and all positive and negative actions remain
inside the source normalization range. Four excluded raw audit episodes are
retained but do not appear in the pair file. See
`outputs/targeted_finetuning/c1_report.md`.

## Positive-BC upper-bound result

The initial 50/50 clean / paired-positive BC treatment was trained for 10,000
updates together with an update-matched clean-only control. Use the paired
deterministic D0/D2 development screen with:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-07-28_targeted_correction/screen_checkpoints.py
```

No mixed checkpoint passed the closed-loop gate. The 10,000-update mixed
policy did, however, change the strict 1 cm fixed-state support result from
0/20 direct-hit states and 1/20 projection-feasible states to 10/20 and 13/20;
the clean-only control remained at 0/20 and 1/20. Corrections therefore inject
safe-progress support, but positive-only BC fails to preserve and select it
without disrupting successful D2 behavior. See
`outputs/targeted_finetuning/upper_bound_report.md`.

## SDP successor result

The paper-aligned set-supervision successor uses the same immutable pair file
with `r=0.1`, `N=16`, and `K_A=16`. A separate 90/10 clean / positive-BC run
controls for correction exposure. Both 10,000-update runs completed, but no
checkpoint passed the fixed seed-600 D0/D2 gate.

The best SDP D2 Task SR is 0.25 versus source 0.35. Its 6,000-update checkpoint
still reaches 10/20 direct strict safe-progress states and 11/20
support-projection-feasible states, confirming that the method preserves
corrective support but does not solve closed-loop mode selection. The 90/10
endpoint similarly reaches 6/20 and 8/20 while its D2 Task SR falls to 0.10.
Do not advance either treatment to D1/D3. See
`outputs/targeted_finetuning/sdp_result_report.md`.
