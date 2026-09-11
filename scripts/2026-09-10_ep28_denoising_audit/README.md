# Layout28 exact denoising comparison

User-authorized2026-09-10. Uses saved full observations/clouds and reconstructs
original CPU/CUDA sampling RNG, validating all33 hashes through step256. Compare
source/forced-on guidance at224/232/240/248/256; reproduce the originally active
branch's recorded action exactly. Stops on an RNG or action mismatch.

```bash
CUDA_VISIBLE_DEVICES=1 ROBOMIMIC_GPU_ID=0 uv run python \
  scripts/2026-09-10_ep28_denoising_audit/audit.py
```

No simulator, rollout, or training. Original source/guidance state is not the
same as the full baseline rollout at the corresponding time. Canonical report:
`outputs/tabletop_guidance_reeval_2026-09-09/ep28_denoising_audit/report.md`.
