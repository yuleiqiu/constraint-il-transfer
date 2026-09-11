# Layout28: sustained backward motion and cloud anomaly

Correction from exact same-noise audit: guidance is OFF from240 to episode end.
Later outlier nearest-point selections below were hypothetical, not applied
guidance. Active guidance before240 also changes Z targets, despite an XY cost.
Read `../ep28_denoising_audit/report.md` for the stronger, corrected evidence.

2026-09-10, offline examination of the user-selected guided video and saved
actions, observations, obstacle clouds and state-bank XML. No new rollout.

- Can projected progress toward the goal peaks at step240 (21.54cm), then ends
  at only0.80cm relative to its start. This includes sustained backward motion,
  not merely small high-frequency jitter.
- At chunk240, mean raw XY action target projects2.6cm behind current EEF;
  actual Can progress over the chunk is-2.8cm. At248 these values are-3.7/-3.8cm.
  Backward commands persist over subsequent chunks. Thus the executed policy
  targets themselves request retreat; it is not just a controller moving
  backward against consistently forward targets.
- First observed obstacle contact is step256, involving gripper base and Milk.
  Retreat starts16 control steps earlier. Collision cannot explain its onset
  as a simple physical rebound. No claim is made that the entire actor had
  already cleared the obstacle at240.
- Recomputed single selected points change from (0.0477,-0.0048,0.8174) at232
  to (0.0377,0.0090,0.8953) at240. Both are within the combined obstacle AABB.
  Which side a radial gradient favors depends on each predicted clean waypoint,
  not merely the current EEF-to-point vector. No source-only denoising sample
  under this same observation/noise was retained, so source versus guidance
  contributions to the backward target are not isolated by this audit.
- A separate later anomaly: selected cloud points at344/376/480/560 are
  respectively about17.0/17.4/19.1/19.9cm outside the combined Milk+support AABB
  reconstructed from static mesh/box geometry. At each of these sampled chunks,
  one of128 points is more than1cm outside that conservative bound. It is that
  outlier which the nearest-point selection chooses near the EEF. The extraction
  call disables `max_geom_distance` filtering. This identifies nonphysical
  obstacle-cloud content, but does not establish the extraction error's origin
  or explain the initial reversal at240, whose audited cloud had no such
  >1cm outlier. A blanket multi-point cost could propagate such outlier errors.

Hypotheses: nearest-point changes/radial repulsion may oppose forward progress;
the source may propose backward continuations after intervention; later cloud
outliers may sustain perturbations. Their causal contributions need same-input
denoising comparisons (source/guided with identical noise) and matched closed-loop
checks, before selecting geometry cleanup, geometric cost changes or route
persistence. Merely smoothing the output does not establish task recovery.

Artifacts: `progress.png`, `chunks.json`. Original video and numerical artifacts
remain under `../shard_25_49/episodes/ep28/guided/`.
