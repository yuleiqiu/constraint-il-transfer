# Toy Recovery Archive

This directory stores compact reports and machine-readable summaries from Toy
experiments that are no longer compatible with the active source checkpoint.

`repeated_terminal_source/` used a source trained with repeated final delta
actions. That terminal convention was later identified as incorrect and
replaced by absorbing zero-delta padding. Its source gate, dynamics validation,
and geometry-guidance screen remain historical mechanism evidence only.

Raw rollout HDF5 / JSONL files, the failed latent model, and the five
superseded source checkpoints were deleted after the reports and manifests
were verified. They can be reproduced from Git history and the retained D0
demonstration dataset, but must not be combined with the active absorbing
source.
