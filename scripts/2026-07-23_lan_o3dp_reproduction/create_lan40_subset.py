"""Create the deterministic 40-demo LAN-O3DP training subset.

The source HDF5 files are opened read-only. Demo selection and validation
holdout use two independent ``numpy.random.default_rng(42)`` instances. The
initial Can XY / yaw audit is descriptive and never changes the selected IDs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORY_SOURCE = (
    REPO_ROOT
    / "third_party/robomimic/datasets/can/yq/"
    "image_v15_delta_eef_pose_osc_pointcloud.hdf5"
)
DEFAULT_POSE_SOURCE = (
    REPO_ROOT
    / "third_party/robomimic/datasets/can/yq/"
    "image_v15_delta_eef_pose_osc.hdf5"
)
DEFAULT_DEST = (
    REPO_ROOT
    / "third_party/robomimic/datasets/can/yq/"
    "image_v15_delta_eef_pose_osc_pointcloud_lan40_seed42.hdf5"
)
SEED = 42
SUBSET_SIZE = 40
VALID_SIZE = 2
FORMAT_VERSION = 1


def numeric_demo_sort(demo_names):
    return sorted(demo_names, key=lambda name: int(name.split("_")[-1]))


def select_demo_split(demo_names, seed=SEED):
    ordered = numeric_demo_sort(demo_names)
    if len(ordered) < SUBSET_SIZE:
        raise ValueError(
            "Need at least {} demos, found {}".format(SUBSET_SIZE, len(ordered))
        )
    selected = np.random.default_rng(seed).choice(
        ordered, SUBSET_SIZE, replace=False
    )
    selected = numeric_demo_sort([str(name) for name in selected])
    valid = np.random.default_rng(seed).choice(
        selected, VALID_SIZE, replace=False
    )
    valid = numeric_demo_sort([str(name) for name in valid])
    valid_set = set(valid)
    train = [name for name in selected if name not in valid_set]
    return selected, train, valid


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path):
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def dataset_sha256(dataset, rows_per_chunk=32):
    digest = hashlib.sha256()
    if dataset.shape == ():
        digest.update(np.asarray(dataset[()]).tobytes())
        return digest.hexdigest()
    if dataset.shape[0] == 0:
        return digest.hexdigest()
    for start in range(0, dataset.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, dataset.shape[0])
        values = np.asarray(dataset[start:stop])
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def recursive_dataset_metadata(group):
    metadata = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            metadata[name] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "sha256": dataset_sha256(obj),
            }

    group.visititems(visitor)
    return metadata


def quat_xyzw_to_yaw(quaternion):
    x, y, z, w = quaternion
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(values.max()),
    }


def summarize_pose_rows(rows):
    yaw = np.asarray([row["can_yaw_rad"] for row in rows], dtype=np.float64)
    circular_mean = float(np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean()))
    resultant = float(np.hypot(np.sin(yaw).mean(), np.cos(yaw).mean()))
    return {
        "can_x": summarize_values([row["can_x"] for row in rows]),
        "can_y": summarize_values([row["can_y"] for row in rows]),
        "can_yaw_rad": {
            **summarize_values(yaw),
            "circular_mean": circular_mean,
            "mean_resultant_length": resultant,
        },
    }


def initial_can_pose_audit(pose_source, selected):
    rows = []
    with h5py.File(pose_source, "r") as source:
        demos = numeric_demo_sort(source["data"].keys())
        for demo_name in demos:
            object_state = np.asarray(
                source["data/{}/obs/object".format(demo_name)][0]
            )
            if object_state.shape != (14,):
                raise ValueError(
                    "{} obs/object has shape {}; expected the PickPlaceCan "
                    "[rel pos(3), rel quat(4), Can pos(3), Can quat xyzw(4)] "
                    "layout".format(demo_name, object_state.shape)
                )
            rows.append(
                {
                    "demo_id": demo_name,
                    "can_x": float(object_state[7]),
                    "can_y": float(object_state[8]),
                    "can_yaw_rad": quat_xyzw_to_yaw(object_state[10:14]),
                }
            )
    selected_set = set(selected)
    subset_rows = [row for row in rows if row["demo_id"] in selected_set]
    return {
        "source_layout": (
            "obs/object = Can_to_robot0_eef_pos(3), "
            "Can_to_robot0_eef_quat(4), Can_pos(3), Can_quat_xyzw(4)"
        ),
        "all_200": {
            "summary": summarize_pose_rows(rows),
            "per_demo": rows,
        },
        "selected_40": {
            "summary": summarize_pose_rows(subset_rows),
            "per_demo": subset_rows,
        },
        "selection_changed_by_audit": False,
    }


def copy_attributes(source, destination):
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def validate_demo_integrity(demo, demo_name):
    num_samples = int(demo.attrs["num_samples"])
    metadata = recursive_dataset_metadata(demo)
    if not metadata:
        raise ValueError("{} contains no datasets".format(demo_name))
    bad_lengths = {
        key: values["shape"]
        for key, values in metadata.items()
        if not values["shape"] or values["shape"][0] != num_samples
    }
    if bad_lengths:
        raise ValueError(
            "{} datasets do not match num_samples {}: {}".format(
                demo_name, num_samples, bad_lengths
            )
        )
    return num_samples, metadata


def build_subset(trajectory_source, pose_source, dest, manifest_path, overwrite=False):
    trajectory_source = trajectory_source.resolve()
    pose_source = pose_source.resolve()
    dest = dest.resolve()
    manifest_path = manifest_path.resolve()
    for source in (trajectory_source, pose_source):
        if not source.exists():
            raise FileNotFoundError(source)
    if dest in (trajectory_source, pose_source):
        raise ValueError("Destination must differ from both sources")
    if dest.exists() and not overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to regenerate".format(dest)
        )

    source_before = {
        "trajectory_source": file_fingerprint(trajectory_source),
        "pose_audit_source": file_fingerprint(pose_source),
    }

    with h5py.File(trajectory_source, "r") as source:
        source_demos = numeric_demo_sort(source["data"].keys())
        selected, train, valid = select_demo_split(source_demos)
        if valid != ["demo_18", "demo_164"]:
            raise AssertionError("Unexpected fixed validation split: {}".format(valid))
        pose_audit = initial_can_pose_audit(pose_source, selected)

        temp_dest = dest.with_name(dest.name + ".tmp")
        if temp_dest.exists():
            temp_dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        trajectory_manifest = {}
        try:
            with h5py.File(temp_dest, "w") as output:
                copy_attributes(source, output)
                output.attrs["lan40_subset_format_version"] = FORMAT_VERSION
                output.attrs["lan40_rng"] = "numpy.random.default_rng"
                output.attrs["lan40_seed"] = SEED
                output.attrs["lan40_selection_algorithm"] = (
                    "choice(numeric_sorted_demo_ids, 40, replace=False); "
                    "numeric sort; independent choice(selected, 2, replace=False)"
                )
                data = output.create_group("data")
                copy_attributes(source["data"], data)
                data.attrs["total"] = sum(
                    int(source["data/{}".format(name)].attrs["num_samples"])
                    for name in selected
                )

                for demo_name in selected:
                    source.copy(
                        source["data/{}".format(demo_name)],
                        data,
                        name=demo_name,
                    )
                    source_demo = source["data/{}".format(demo_name)]
                    output_demo = data[demo_name]
                    num_samples, source_metadata = validate_demo_integrity(
                        source_demo, demo_name
                    )
                    output_num_samples, output_metadata = validate_demo_integrity(
                        output_demo, demo_name
                    )
                    if (
                        output_num_samples != num_samples
                        or output_metadata != source_metadata
                    ):
                        raise AssertionError(
                            "{} copy does not match source".format(demo_name)
                        )
                    trajectory_manifest[demo_name] = {
                        "split": "valid" if demo_name in set(valid) else "train",
                        "num_samples": num_samples,
                        "datasets": output_metadata,
                        "complete": True,
                        "matches_source": True,
                    }

                mask = output.create_group("mask")
                mask.create_dataset("train", data=np.asarray(train, dtype="S"))
                mask.create_dataset("valid", data=np.asarray(valid, dtype="S"))
                mask.create_dataset("lan40", data=np.asarray(selected, dtype="S"))
                output.flush()
            os.replace(temp_dest, dest)
        finally:
            if temp_dest.exists():
                temp_dest.unlink()

    source_after = {
        "trajectory_source": file_fingerprint(trajectory_source),
        "pose_audit_source": file_fingerprint(pose_source),
    }
    if source_after != source_before:
        raise AssertionError("A source dataset changed while creating the subset")

    manifest = {
        "format_version": FORMAT_VERSION,
        "output_dataset": {
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
        },
        "sources": source_before,
        "source_unchanged_after_copy": True,
        "selection": {
            "rng": "numpy.random.default_rng",
            "seed": SEED,
            "input_sort": "numeric suffix ascending (demo_1 ... demo_200)",
            "subset_algorithm": "choice(ordered_ids, 40, replace=False)",
            "validation_algorithm": (
                "numeric sort selected IDs, then independent "
                "default_rng(42).choice(..., 2, replace=False)"
            ),
            "selected_40": selected,
            "train_38": train,
            "valid_2": valid,
            "selection_conditioning": "none",
        },
        "initial_can_pose_distribution_audit": pose_audit,
        "trajectories": trajectory_manifest,
        "integrity": {
            "train_valid_disjoint": set(train).isdisjoint(valid),
            "train_valid_union_equals_selected": set(train).union(valid)
            == set(selected),
            "all_trajectory_copies_match_source": all(
                item["matches_source"] for item in trajectory_manifest.values()
            ),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory-source", type=Path, default=DEFAULT_TRAJECTORY_SOURCE
    )
    parser.add_argument("--pose-source", type=Path, default=DEFAULT_POSE_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else args.dest.with_suffix(".manifest.json")
    )
    manifest = build_subset(
        trajectory_source=args.trajectory_source,
        pose_source=args.pose_source,
        dest=args.dest,
        manifest_path=manifest_path,
        overwrite=args.overwrite,
    )
    print("Wrote {}".format(manifest["output_dataset"]["path"]))
    print("Manifest: {}".format(manifest_path.resolve()))
    print("Selected: {}".format(manifest["selection"]["selected_40"]))
    print("Validation: {}".format(manifest["selection"]["valid_2"]))


if __name__ == "__main__":
    main()
