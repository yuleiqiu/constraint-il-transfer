"""Generate the fixed Empty2D conditional-RBF dataset and gate manifest."""

import argparse
import json
from pathlib import Path

from empty2d_guidance.data import (
    KERNEL_AMPLITUDE,
    LENGTH_SCALE,
    SEED,
    file_sha256,
    generate_trajectories,
    validate_hdf5,
    validate_trajectories,
    write_hdf5,
    write_support_plot,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/empty2d_guidance/data"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectories, attempts = generate_trajectories()
    validation = validate_trajectories(trajectories)
    if not validation["passed"]:
        raise RuntimeError("Data gate failed: {}".format(validation["gates"]))
    hdf5_path = args.out_dir / "empty2d_guidance.hdf5"
    write_hdf5(hdf5_path, trajectories)
    hdf5_validation = validate_hdf5(hdf5_path)
    if not hdf5_validation["passed"]:
        raise RuntimeError("HDF5 gate failed: {}".format(hdf5_validation["gates"]))
    write_support_plot(args.out_dir / "support.png", trajectories)
    manifest = {
        "format_version": 1,
        "generator": {
            "type": "endpoint-conditioned RBF Gaussian process",
            "length_scale": LENGTH_SCALE,
            "kernel_amplitude": KERNEL_AMPLITUDE,
            "seed": SEED,
            "accepted": len(trajectories),
            "attempts": attempts,
            "rejection": "workspace bounds or adjacent displacement > 0.03 only",
            "amplitude_deviation": "Raised from upstream-script value 1 to 8 to provide counterfactual avoidance support.",
        },
        "split": {"train": 90, "valid": 10},
        "validation": validation,
        "hdf5_validation": hdf5_validation,
        "hdf5": {
            "path": str(hdf5_path.resolve()),
            "bytes": hdf5_path.stat().st_size,
            "sha256": file_sha256(hdf5_path),
        },
        "support_plot": str((args.out_dir / "support.png").resolve()),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
