# Empty2D smooth-guidance horizon ablation

Circle/square geometry: `circle`; H=600; rho=1; Q=0.12; Huber epsilon=0.03.

| Condition | Task SR | Safe SR | CR | NCR | Steps | Backward distance | X reversals | Path TV | Activation switches | Clip rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| centered_baseline | 1.00 | 0.05 | 0.95 | 0.00 | 222.4 | 0.002 | 1.1 | 0.574 | 0.0 | 0.000 |
| centered_hinge | 0.65 | 0.65 | 0.00 | 0.35 | 457.0 | 0.923 | 202.7 | 4.605 | 14.8 | 0.001 |
| centered_huber_hinge | 0.65 | 0.65 | 0.00 | 0.35 | 450.7 | 0.896 | 195.8 | 4.408 | 9.1 | 0.000 |
| offset_baseline | 1.00 | 0.00 | 1.00 | 0.00 | 222.4 | 0.002 | 1.1 | 0.574 | 0.0 | 0.000 |
| offset_hinge | 1.00 | 1.00 | 0.00 | 0.00 | 289.4 | 0.135 | 30.6 | 1.460 | 22.7 | 0.001 |
| offset_huber_hinge | 1.00 | 1.00 | 0.00 | 0.00 | 276.1 | 0.100 | 24.9 | 1.263 | 15.4 | 0.000 |

## Horizon first-passage outcomes

| Condition | Horizon | Task SR | Safe SR | CR | NCR |
|---|---:|---:|---:|---:|---:|
| centered_baseline | 400 | 1.00 | 0.05 | 0.95 | 0.00 |
| centered_baseline | 500 | 1.00 | 0.05 | 0.95 | 0.00 |
| centered_baseline | 600 | 1.00 | 0.05 | 0.95 | 0.00 |
| centered_hinge | 400 | 0.45 | 0.45 | 0.00 | 0.55 |
| centered_hinge | 500 | 0.55 | 0.55 | 0.00 | 0.45 |
| centered_hinge | 600 | 0.65 | 0.65 | 0.00 | 0.35 |
| centered_huber_hinge | 400 | 0.45 | 0.45 | 0.00 | 0.55 |
| centered_huber_hinge | 500 | 0.55 | 0.55 | 0.00 | 0.45 |
| centered_huber_hinge | 600 | 0.65 | 0.65 | 0.00 | 0.35 |
| offset_baseline | 400 | 1.00 | 0.00 | 1.00 | 0.00 |
| offset_baseline | 500 | 1.00 | 0.00 | 1.00 | 0.00 |
| offset_baseline | 600 | 1.00 | 0.00 | 1.00 | 0.00 |
| offset_hinge | 400 | 0.95 | 0.95 | 0.00 | 0.05 |
| offset_hinge | 500 | 0.95 | 0.95 | 0.00 | 0.05 |
| offset_hinge | 600 | 1.00 | 1.00 | 0.00 | 0.00 |
| offset_huber_hinge | 400 | 0.95 | 0.95 | 0.00 | 0.05 |
| offset_huber_hinge | 500 | 1.00 | 1.00 | 0.00 | 0.00 |
| offset_huber_hinge | 600 | 1.00 | 1.00 | 0.00 | 0.00 |

## Paired comparisons

- `centered_hinge_vs_baseline`: task F→S/S→F = 0/7, safe F→S/S→F = 12/0, mean Δ backward distance = 0.921, mean Δ x reversals = 201.55.
- `centered_huber_vs_baseline`: task F→S/S→F = 0/7, safe F→S/S→F = 12/0, mean Δ backward distance = 0.894, mean Δ x reversals = 194.65.
- `centered_huber_vs_hinge`: task F→S/S→F = 0/0, safe F→S/S→F = 0/0, mean Δ backward distance = -0.027, mean Δ x reversals = -6.90.
- `offset_hinge_vs_baseline`: task F→S/S→F = 0/0, safe F→S/S→F = 20/0, mean Δ backward distance = 0.133, mean Δ x reversals = 29.40.
- `offset_huber_vs_baseline`: task F→S/S→F = 0/0, safe F→S/S→F = 20/0, mean Δ backward distance = 0.098, mean Δ x reversals = 23.80.
- `offset_huber_vs_hinge`: task F→S/S→F = 0/0, safe F→S/S→F = 0/0, mean Δ backward distance = -0.035, mean Δ x reversals = -5.60.
