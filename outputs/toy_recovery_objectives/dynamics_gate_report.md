# PointNav latent-dynamics gate result

Date: 2026-07-31
Status: **failed the persistence baseline; guidance blocked**

## Data contract and budget

The run used the five corrected image-policy checkpoints at 1k, 2k, 3k, 4k,
and 5k updates. It collected exactly 150 unlabeled rollouts, balanced as ten
rollouts for every D0/D1/D2 and checkpoint combination.

At each of 990 replan states, the simulator was cloned and the predicted
15-action future suffix `[1:16]` was executed open loop. Slot zero of the
16-slot Diffusion Policy prediction is the preceding alignment action and was
not re-executed. Terminal states are absorbing. The rollout-level split is
stratified by difficulty and checkpoint: 120 training rollouts / 783 queries
and 30 validation rollouts / 207 queries.

Success and failure rollouts were both retained. Outcome counts were D0
49/1, D1 20/30, and D2 5/45 (success/failure), but outcome, reward, collision,
and correction labels were not consumed by dynamics training or splitting.

## Gate result

| Metric | Value | Requirement |
|---|---:|---:|
| Validation MSE | 0.0007789 | below persistence |
| Persistence MSE | 0.0007164 | baseline |
| Shuffled-action MSE | 0.0015007 | at least 1.2x normal |
| Shuffle / normal ratio | 1.9267 | passed |

The predictor clearly uses action information: shuffling actions nearly
doubles its error. It nevertheless predicts held-out future latents worse than
the identity persistence baseline by about 8.7%. The combined gate therefore
fails.

Per the fixed funnel, the D0 expert latent bank was not built and no guidance
scale selection, rapid screen, fixed-state recovery diagnostic, test rollout,
or confirmation run was executed. This result blocks comparison of geometry
and global latent-manifold guidance; it is not a negative result for either
guidance objective.
