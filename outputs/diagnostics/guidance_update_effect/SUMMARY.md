# Guidance Update Effect Diagnostic

Date: 2026-07-03

Script:

```text
scripts/diagnose_guidance_update_effect.py
```

Purpose:

Check whether obstacle guidance actually improves obstacle clearance in a same-state counterfactual test:

```text
same simulator state + same diffusion seed
  A_before = unguided action chunk
  A_after  = guided action chunk
```

Then compare:

```text
predicted clearance: forward_model(A_after) - forward_model(A_before)
actual clearance:    execute(A_after) - execute(A_before)
```

Scope note: this diagnostic tests the current gradient guidance update rule.
It is not a standalone validation or rejection of the forward model. Forward
model accuracy is evaluated separately through held-out demo windows and
random-action env rollouts.

## Method

For each candidate state:

1. Save simulator state with `env.get_state()`.
2. Generate unguided and guided action chunks using the same `torch.manual_seed`.
3. Use the forward model to predict EEF trajectories for both chunks.
4. Restore the same simulator state and execute each chunk for `H=8` steps.
5. Compare predicted and actual minimum obstacle clearance.

Only active samples are meaningful:

```text
||A_after - A_before|| > threshold
```

## Normal Setting

Settings:

```text
env = PickPlaceBreadCerealCan
seed = 700
guidance_scale = 0.005 and 0.03
xy_clearance = 0.02
backend = forward_model
guidance_horizon = 8
```

Results:

| run | candidates | active samples | conclusion |
|---|---:|---:|---|
| `scale=0.005` | 50 | 0 | no final action chunk change found |
| `scale=0.03` | 50 | 0 | no final action chunk change found |

Interpretation:

For this rollout seed and the default safety margin, the sampled action chunks did not enter the obstacle-cost active region at chunk boundaries. Therefore this setting cannot diagnose whether guidance improves actual clearance.

## Stress Setting

To force active guidance updates, the safety margin was increased:

```text
xy_clearance = 0.08
guidance_scale = 0.005
```

Results:

| metric | value |
|---|---:|
| candidates scanned | 30 |
| active samples | 10 |
| predicted clearance improved | 10/10 |
| actual clearance improved | 6/10 |
| sign agreement | 6/10 |
| mean predicted clearance delta | 0.0736 cm |
| mean actual clearance delta | 0.0553 cm |
| mean action chunk delta L2 | 0.0283 |
| mean first-action delta L2 | 0.0135 |

Per-sample deltas:

| rollout step | predicted delta (cm) | actual delta (cm) | action delta L2 |
|---:|---:|---:|---:|
| 0 | 0.0072 | 0.0047 | 0.0026 |
| 8 | 0.2468 | 0.2110 | 0.0687 |
| 16 | 0.1132 | 0.1019 | 0.0434 |
| 24 | 0.1185 | 0.1060 | 0.0313 |
| 32 | 0.0764 | 0.0916 | 0.0275 |
| 40 | 0.0481 | -0.0002 | 0.0278 |
| 48 | 0.0145 | -0.0056 | 0.0236 |
| 56 | 0.0138 | -0.0090 | 0.0180 |
| 64 | 0.0170 | -0.0069 | 0.0189 |
| 232 | 0.0805 | 0.0594 | 0.0215 |

## Conclusion

The stress test shows that the guidance update does optimize its predicted objective:

```text
predicted clearance improved in 10/10 active cases
```

But this does not reliably transfer to actual execution:

```text
actual clearance improved in only 6/10 active cases
```

So the current bottleneck is not autograd or forward-model differentiability.
The weak point is the reliability of the current guidance update as an
execution-time correction:

```text
forward-model predicted improvement != guaranteed actual clearance improvement
```

For the default setting (`xy_clearance=0.02`), active updates are rare in the sampled chunk-boundary states, which also explains why rollout-level effects are weak and noisy.
