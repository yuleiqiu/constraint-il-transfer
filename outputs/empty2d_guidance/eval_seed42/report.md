# Empty2D guidance paired evaluation

Training seed: 42

| Geometry | rho | Q | Base / guided Task SR | Base / guided Safe SR | Base / guided CR | Base / guided NCR | F→S | S→F | exact p | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| circle | 0.0001 | 0.02 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 / 1.00 | 0.00 / 0.00 | 0 | 0 | 1 | no |
| square | 0.0001 | 0.02 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 / 1.00 | 0.00 / 0.00 | 0 | 0 | 1 | no |

## Conclusion

Current guidance is not validated even in the simplified environments.

The failed two-geometry gate does not validate the current guidance even in idealized point geometry. Post-hoc large-scale diagnostics are reported separately and do not change this preregistered conclusion.
