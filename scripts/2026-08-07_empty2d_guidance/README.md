# Empty2D Guidance Runbook

All commands run from the repository root with `uv`.

Generate and verify the fixed dataset:

```bash
uv run python scripts/2026-08-07_empty2d_guidance/generate_data.py
uv run python scripts/2026-08-07_empty2d_guidance/verify_data.py
```

Run the real batch-256 preflight:

```bash
ROBOMIMIC_GPU_ID=0 uv run python \
  scripts/2026-08-07_empty2d_guidance/preflight.py
```

Train seed 42. The launcher refuses to start if the data gate is false or CUDA
is unavailable. W&B project `empty2d_guidance` is required in online mode;
authentication or logging failure aborts the run.

```bash
ROBOMIMIC_GPU_ID=0 uv run python \
  scripts/2026-08-07_empty2d_guidance/train.py --seed 42
```

Screen the timestamped run, then calibrate and test the selected checkpoint:

```bash
uv run python scripts/2026-08-07_empty2d_guidance/screen_checkpoints.py \
  --run-dir outputs/robomimic/train/empty2d_guidance_sample_seed42/<timestamp> \
  --device cuda:1

ROBOMIMIC_GPU_ID=0 uv run python \
  scripts/2026-08-07_empty2d_guidance/paired_evaluation.py \
  --checkpoint /path/to/selected.pth --training-seed 42 \
  --out-dir outputs/empty2d_guidance/eval_seed42
```

Circle and Square can instead be sharded across the two GPUs without changing
their fixed seeds or selection rule. Run these two evaluation commands in
parallel, then merge their disjoint artifacts:

```bash
uv run python scripts/2026-08-07_empty2d_guidance/paired_evaluation.py \
  --checkpoint /path/to/selected.pth --training-seed 42 \
  --out-dir outputs/empty2d_guidance/eval_seed42 \
  --environment circle --device cuda:0
uv run python scripts/2026-08-07_empty2d_guidance/paired_evaluation.py \
  --checkpoint /path/to/selected.pth --training-seed 42 \
  --out-dir outputs/empty2d_guidance/eval_seed42 \
  --environment square --device cuda:1
uv run python scripts/2026-08-07_empty2d_guidance/merge_geometry_evaluations.py \
  --eval-dir outputs/empty2d_guidance/eval_seed42
```

Only if seed 42 passes or partially passes, train 43 and 44 with the same
launcher. Their evaluations must pass seed 42's
`eval_seed42/locked_parameters.json` via `--locked-parameters`; they do not
recalibrate. Render typical held-out pairs after evaluation:

```bash
uv run python scripts/2026-08-07_empty2d_guidance/render_videos.py \
  --eval-dir outputs/empty2d_guidance/eval_seed42 \
  --out-dir outputs/empty2d_guidance/videos_seed42
```

After all three locked-parameter evaluations, aggregate the replication gate:

```bash
uv run python scripts/2026-08-07_empty2d_guidance/aggregate_seeds.py \
  outputs/empty2d_guidance/eval_seed{42,43,44}/result.json
```
