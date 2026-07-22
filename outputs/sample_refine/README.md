# Sample-Refine Result

Date: 2026-07-22

The tested sample-refine Diffusion Policy did not pass the two-distractor
method-development gate. With fixed margin `0.02`, guidance scale `0.003`,
`K=4`, and refinement timestep 40, sample+refine matched the baseline four-way
outcome on all 20 paired layouts while adding about 76% inference time.

Do not scale this configuration to held-out clutter levels or the full
evaluation matrix. Sample only produced one additional safe success, while
refine only exchanged three collision failures for three collision-free
non-completions; neither observation validates the combined method.

Technical interpretation and scope:
[`../../docs/guided_denoising/sample_refine_pilot.md`](../../docs/guided_denoising/sample_refine_pilot.md).

Canonical result files:

- [`fixed_state_sweep_final/summary.json`](fixed_state_sweep_final/summary.json)
- [`paired_pilot_d2_n20/report.md`](paired_pilot_d2_n20/report.md)
- [`paired_pilot_d2_n20/summary.json`](paired_pilot_d2_n20/summary.json)
- [`paired_pilot_d2_n20/manifest.json`](paired_pilot_d2_n20/manifest.json)

The per-state and per-episode JSONL files remain local audit artifacts and are
not intended for version control.

Reproduction requires the root and robomimic repositories at their respective
`archive/s2m-sample-refine-20260722` tags. The active robomimic
`exp/guided-dp` baseline intentionally excludes this failed algorithm.
