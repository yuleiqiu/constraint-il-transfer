# Robomimic Training Runs

These directories are local, ignored training artifacts. Configurations and
logs are retained so completed experiments remain auditable, but checkpoints
follow the manifest in `docs/checkpoint_retention.md`.

As of the 2026-07-30 recoverability-boundary archive, keep exactly:

- the epoch-260 clean-image delta-EEF source;
- the SDP 6k-update checkpoint (`model_epoch_12.pth`);
- the LAN epoch-350 checkpoint used by the completed reproduction.

The verified C1 correction raw and paired HDF5 files live under
`outputs/targeted_finetuning/c1/` and are retained independently of this
directory. Superseded checkpoints are not backups and may be regenerated only
from their preserved configs and code. Before any later cleanup, update and
verify the retention manifest.
