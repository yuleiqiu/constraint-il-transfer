# Empty2D smooth-guidance horizon ablation

This successor experiment leaves the frozen H320 paper-scale result unchanged.
It reuses the selected seed-42 epoch-100 checkpoint and fixes:

- one continuous rollout through at most H600, with first-passage outcomes
  reported at H400, H500, and H600;
- `rho=1`, `Q=0.12`;
- either the original hinge or a Huberized hinge with `epsilon=0.03`;
- centered obstacles or deterministic mirrored offsets with
  `x in [-0.25, 0.25]` and `|y| in [0.08, 0.18]`;
- a separate frozen layout seed, with every direct start-to-goal segment still
  obstructed.

Run the default ten-pair Circle ablation from the project root:

```bash
ROBOMIMIC_GPU_ID=0 uv run python \
  scripts/2026-08-12_empty2d_smooth_guidance/paired_ablation.py \
  --checkpoint outputs/robomimic/train/empty2d_guidance_sample_seed42/20260807210551/models/model_epoch_100.pth \
  --environment circle --offset-pairs 10 --horizon 600 \
  --out-dir outputs/empty2d_guidance/smooth_h600_circle
```

Use `--offset-pairs 1` only for an implementation smoke rollout. The output
directory is resumable and refuses a changed checkpoint, geometry, layout
count, or frozen contract.

The centered and offset halves can be run concurrently on separate GPUs by
passing `--condition-group centered` and `--condition-group offset` with the
same output directory. After both finish, rerun once with
`--condition-group all`; every rollout resumes and only aggregation runs.
