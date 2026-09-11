# Offline placement rescore

Authorized 2026-09-10: try release + in-region residence for0.5s, without a
minimum retreat distance. See the operational definition and reconstruction
limits in `outputs/tabletop_guidance_reeval_2026-09-09/placement_rescore/report.md`.
Uses conservative absence of all gripper contact, not just an open command or
absence of bilateral pad contact. Samples are the existing20Hz boundaries.

```bash
MUJOCO_GL=egl uv run python scripts/2026-09-10_tabletop_placement_rescore/rescore.py
```

CPU-only XML/state contact reconstruction, no rendering or physics stepping and
no policy inference. Requires existing local NPZ trajectories and trusted saved
initial-state pickle. Reconstructs Euler pre-integration derived geometry with
per-sample position checks. Fails rather than silently accepting disagreement.
Preserves strict scores. Early old-success terminations without0.5s confirmation
are censored; do not silently count them as failures or resume policy execution.

## Inclusive follow-up (2026-09-11)

The user confirmed old strict success OR released placement held0.5s. Run
`uv run python scripts/2026-09-10_tabletop_placement_rescore/inclusive_rescore.py`.
This resolves all50 pairs at the earliest recorded success, without simulation.
Source/guidance Task46/31 and Safe39/30. See the separate
`outputs/tabletop_guidance_reeval_2026-09-09/inclusive_rescore/report.md`.
