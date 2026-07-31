# PointNav image-source gate result

Date: 2026-07-31
Status: **passed after correcting the source implementation**

## Superseded preflight

The first source run used a custom global-average CNN, zero-padded action
suffixes, the wrong executable slice, and DDIM sampling without the clipping
used by the repository's established image Diffusion Policy. It reached only
`0.76` D0 calibration success after 10,000 updates. This was an implementation
preflight failure, not evidence that image observations or 200 demonstrations
are insufficient. Its checkpoints were removed after the corrected run passed.

## Corrected image policy

The replacement keeps RGB-only policy input, 200 D0 demonstrations, horizons
`2 / 16 / 8`, and DDIM `100 / 10`. It directly reuses robomimic's
`VisualCore(ResNet18Conv + SpatialSoftmax)`, replaces BatchNorm with GroupNorm,
uses the source-aligned executable slice `[1:9]`, repeat-pads sequence
boundaries, and enables the reference DDIM sample clipping.

A 1,000-update preflight reached `47/50` D0 calibration successes. Continuing
the same run to the preregistered initial 5,000-update budget produced the
formal gate result:

| Difficulty | Oracle success | Source success | Required source range |
|---|---:|---:|---:|
| D0 | 1.00 | 1.00 | at least 0.95 |
| D1 | 1.00 | 0.54 | 0.20--0.80 |
| D2 | 1.00 | 0.06 | 0.05--0.50 and at least 0.10 below D1 |

## Calibration-only environment repair

The original obstacle centers sat directly on the D0 route, making every
high-quality straight source trajectory collide. Obstacle scale alone left D1
at zero. On calibration layouts only, the templates were changed to strict
near-tangent blockers:

- D1 uses one radius-`0.07` obstacle offset to `0.85` of its agent-inflated
  radius, with the side fixed deterministically by layout-seed parity.
- D2 uses two radius-`0.07` blockers on the same side at separated x positions,
  offset by `0.78` and `0.82` of their inflated radii. This requires a
  persistent lateral detour without demanding an unseen D0 source execute an
  untrained S-turn.

Every accepted D1 layout retains genuine upper and lower oracle modes. Every
accepted D2 layout has both obstacles blocking the direct segment and at least
eight actions between their closest oracle interaction points. Oracle success
is 1.00. No source rollout on scale-tuning, screen, or test layouts was used
to select these templates.

The passing candidate manifest was copied byte-for-byte to
`layout_manifest.json`; both files had SHA-256
`c970b55e925e6d2ad12bbbdf815851a9a749bfbb602ce410d3ca1cee002f0be6`.

The source gate authorized the fixed 150-rollout dynamics stage. Its separate
result is in `dynamics_gate_report.md`.
