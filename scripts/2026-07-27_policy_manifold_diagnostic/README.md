# Policy-Manifold Feasibility Diagnostic

This experiment tests whether a frozen single-object diffusion policy's local
conditional action distribution contains a progress-preserving collision-free
trajectory at fixed pre-target states from the two-distractor environment.

It estimates a shrinkage covariance from repeated policy samples and solves a
Mahalanobis projection constrained by:

- swept XY EEF-point / obstacle-circle clearance;
- preservation of the nominal chunk's closest target progress;
- the empirical policy-support ellipsoid; and
- executable delta-position action bounds.

The diagnostic is phase-free and does not clone future simulator rollouts,
execute projected actions, or update the policy. Its geometry remains an oracle
EEF-point proxy; whole-gripper / whole-body SDF evaluation is intentionally
deferred until local policy-support feasibility is established.

Run unit tests:

```bash
uv run python \
  scripts/2026-07-27_policy_manifold_diagnostic/test_manifold_feasibility.py
```

Run the fixed-state diagnostic over the existing 20-state bank:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-07-27_policy_manifold_diagnostic/run_fixed_state.py
```

`--support-multiplier` scales the empirical squared Mahalanobis radius for
threshold-sensitivity checks. A value of `2` permits roughly 1.41 times the
default Mahalanobis radius.
