# Full50 baseline / compact-Bezier local planner comparison

Completed2026-09-11, reviewed2026-09-14:100 rollouts and100 videos verified.
Baseline/planner Task46/49, Safe39/49, CR9/0, NCR2/1 out of50.
Start with `conclusion.md` and `video_index.md`.

User authorized2026-09-11. Exactly50 layouts,100 rollouts, no parameter tuning.
GPU0: layouts0--24; GPU1:25--49. Each pair uses the same GPU process and initial
state/observation/policy seed. One execution per condition and layout.

Frozen epoch50 source and current compact-Bezier planner. H600; success is old
strict OR released placement held0.5s. The original geometry margins and all
planner parameters are fixed. No original point-guidance rerun or training.

- `progress.json`: completed rollouts/pairs and descriptive counts on complete pairs only.
- `video_index.md`: MP4s as branches finish; no HTML.
- `shard_00_24/`, `shard_25_49/`: separate per-episode/per-condition artifacts.
- `supervisor.log`, `shard_*.log`: worker/monitor logs.
- On completion: `complete.json`, `conclusion.md`, `summary.json`, `report.md`,
  `artifact_integrity.json`. On failure: explicit failure marker and logs.

Large MP4/NPZ and step traces stay local; compact reports/configurations and
executed source snapshots are retained. The existing bank is a development
comparison, not new independent test scenes.
