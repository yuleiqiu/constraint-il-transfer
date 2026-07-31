# PointNav source-policy gate result

Date: 2026-07-31
Status: **stopped at the source gate**

## Result

The formal rapid prototype did not advance to dynamics collection or guidance.
The final source policy reached only `0.76` success on 50 D0 calibration
layouts, below the preregistered `0.95` threshold. D1 and D2 success were both
zero. The shortest-path oracle succeeded on every calibration layout in all
three difficulties.

| Difficulty | Oracle success | Source success | Required source range |
|---|---:|---:|---:|
| D0 | 1.00 | 0.76 | at least 0.95 |
| D1 | 1.00 | 0.00 | 0.20--0.80 |
| D2 | 1.00 | 0.00 | 0.05--0.50 and at least 0.10 below D1 |

The failing D0 result cannot be repaired by tuning obstacle geometry: D0 has
no obstacle. Changing its task distribution after observing this result would
also invalidate the intended source-policy gate. No frozen
`layout_manifest.json` was written, and no untouched test rollout was run.

## Executed budget

- Generated 200 D0 demonstrations. Exact replay passed for all 200; all were
  successful and collision-free. Maximum recorded action norm was `0.048479`,
  below the `0.05` limit.
- Generated disjoint candidate splits: 50 calibration, 30 scale-tuning, 100
  screen, and 200 untouched test layouts per difficulty (1,140 unique layout
  seeds). Merely constructing and geometrically validating these layouts did
  not run policy evaluation on test layouts.
- Trained one source seed for the initial 5,000 updates. Its calibration D0
  success was `0.62`; a D0-only 20-layout diagnostic showed an improving but
  noisy checkpoint trend (`3, 6, 4, 8, 14` successes at 1k--5k).
- Used the single allowed resume to the fixed 10,000-update cap. Training loss
  decreased from `0.007243` at 5k to `0.002845` at 10k. The final checkpoint
  then produced the gate result above.
- Audited the 6k, 8k, and 10k checkpoint payloads: model, EMA model, online
  model, and optimizer state are present and all floating tensors are finite.

## Work not authorized by the failed gate

- zero dynamics rollouts collected (the revised maximum would have been 150);
- no latent-dynamics model or D0 expert latent bank trained;
- no scale selection, rapid screen, fixed-state recovery diagnostic, or
  confirmation evaluation run;
- no LPB/ReGuide-style extension implemented or evaluated beyond the generic
  prototype components already covered by unit/smoke tests.

This is a source-policy capacity/training result for the 200-demo setting, not
evidence for or against either recovery objective. A later run would require a
new, explicitly approved source-policy plan rather than silently expanding the
training or data budget.
