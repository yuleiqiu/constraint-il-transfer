# Policy-Manifold Fixed-State Feasibility Diagnostic

This is an upper-bound diagnostic, not a rollout method. It asks whether
a progress-preserving collision-free eight-step XY trajectory exists
inside a local support ellipsoid estimated from frozen-policy samples.
No semantic task phase, simulator branching, policy update, or execution
of the projected actions is used.

## Setup

- states: 20 pre-target states captured one chunk before a real collision
- policy samples per state: 32
- support: 95th percentile shrinkage-covariance ellipsoid
- support squared-radius multiplier: 1.00
- covariance shrinkage: 0.2; variance floor: 1.00 mm
- target-progress tolerance: 0.50 mm

The geometry is the existing oracle XY EEF-point / object-circle proxy.
The clearance-margin sweep tests support feasibility before implementing
a whole-gripper or whole-body SDF.

## Results

| expanded boundary | raw samples contain safe+progress path | manifold projection feasible | projection-only rescues | median feasible support ratio | median max waypoint shift |
|---:|---:|---:|---:|---:|---:|
| 0.0 cm | 16/20 (80.0%) | 18/20 (90.0%) | 2 | 0.254 | 0.00 cm |
| 1.0 cm | 10/20 (50.0%) | 13/20 (65.0%) | 3 | 0.533 | 0.00 cm |

Median covariance effective rank: 8.33; median condition number: 20.6.

## Interpretation contract

- A raw-sample success means ordinary policy sampling already contains
  a safe, progress-preserving path for that fixed observation.
- A projection-only rescue means the fitted local ellipsoid connects to
  such a path even though the finite sample set did not contain one.
- Feasibility does not imply that OSC will execute the path accurately
  or that a full rollout will succeed.
- Infeasibility is evidence against this local, phase-free
  policy-support mechanism for the selected collision regime.
