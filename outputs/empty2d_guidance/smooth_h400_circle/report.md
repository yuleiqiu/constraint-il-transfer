# Empty2D H400 smooth-guidance ablation

Circle/square geometry: `circle`; H=400; rho=1; Q=0.12; Huber epsilon=0.03.

| Condition | Task SR | Safe SR | CR | NCR | Steps | Backward distance | X reversals | Path TV | Activation switches | Clip rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| centered_baseline | 1.00 | 0.05 | 0.95 | 0.00 | 222.4 | 0.002 | 1.1 | 0.574 | 0.0 | 0.000 |
| centered_hinge | 0.45 | 0.45 | 0.00 | 0.55 | 367.6 | 0.725 | 159.8 | 3.640 | 14.4 | 0.001 |
| centered_huber_hinge | 0.45 | 0.45 | 0.00 | 0.55 | 359.6 | 0.693 | 151.5 | 3.439 | 7.5 | 0.000 |
| offset_baseline | 1.00 | 0.00 | 1.00 | 0.00 | 222.4 | 0.002 | 1.1 | 0.574 | 0.0 | 0.000 |
| offset_hinge | 0.95 | 0.95 | 0.00 | 0.05 | 280.6 | 0.135 | 29.9 | 1.431 | 22.2 | 0.001 |
| offset_huber_hinge | 0.95 | 0.95 | 0.00 | 0.05 | 271.7 | 0.099 | 24.6 | 1.248 | 14.5 | 0.000 |

## Paired comparisons

- `centered_hinge_vs_baseline`: task F→S/S→F = 0/11, safe F→S/S→F = 8/0, mean Δ backward distance = 0.723, mean Δ x reversals = 158.65.
- `centered_huber_vs_baseline`: task F→S/S→F = 0/11, safe F→S/S→F = 8/0, mean Δ backward distance = 0.691, mean Δ x reversals = 150.35.
- `centered_huber_vs_hinge`: task F→S/S→F = 0/0, safe F→S/S→F = 0/0, mean Δ backward distance = -0.032, mean Δ x reversals = -8.30.
- `offset_hinge_vs_baseline`: task F→S/S→F = 0/1, safe F→S/S→F = 19/0, mean Δ backward distance = 0.133, mean Δ x reversals = 28.80.
- `offset_huber_vs_baseline`: task F→S/S→F = 0/1, safe F→S/S→F = 19/0, mean Δ backward distance = 0.097, mean Δ x reversals = 23.50.
- `offset_huber_vs_hinge`: task F→S/S→F = 0/0, safe F→S/S→F = 0/0, mean Δ backward distance = -0.036, mean Δ x reversals = -5.30.
