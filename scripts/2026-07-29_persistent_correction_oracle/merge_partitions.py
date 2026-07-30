"""Merge disjoint persistent-oracle partitions in global state order."""

import argparse
import json
import shutil
from pathlib import Path

from persistent_oracle import json_dump, summarize_states, write_report


CONFIG_KEYS = (
    "source_agent",
    "corrective_agent",
    "states",
    "trajectory_count",
    "maximum_correction_chunks",
    "maximum_correction_actions",
    "rollout_horizon",
    "target_object_name",
    "correction_seed_formula",
    "continuation_seed_formula",
    "deterministic_torch",
    "gate",
)


def read_partition(path):
    manifest = json.loads((path / "manifest.json").read_text())
    states = [
        json.loads(line)
        for line in (path / "state_results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(states) != manifest["state_limit"]:
        raise ValueError(
            "{} contains {} states, manifest declares {}".format(
                path, len(states), manifest["state_limit"]
            )
        )
    return manifest, states


def merge_partitions(partitions):
    loaded = [read_partition(path) for path in partitions]
    loaded.sort(key=lambda item: item[0]["state_offset"])
    reference = loaded[0][0]
    expected_offset = loaded[0][0]["state_offset"]
    states = []
    partition_records = []
    for manifest, partition_states in loaded:
        for key in CONFIG_KEYS:
            if manifest[key] != reference[key]:
                raise ValueError(
                    "Partition configuration mismatch for {}".format(key)
                )
        if manifest["state_offset"] != expected_offset:
            raise ValueError(
                "Expected partition offset {}, got {}".format(
                    expected_offset, manifest["state_offset"]
                )
            )
        partition_records.append(
            {
                "state_offset": manifest["state_offset"],
                "state_limit": manifest["state_limit"],
            }
        )
        states.extend(partition_states)
        expected_offset += manifest["state_limit"]
    state_ids = [state["state_id"] for state in states]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("Merged partitions contain duplicate state IDs")
    return reference, partition_records, states


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("partitions", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists():
        if not args.overwrite and any(args.out_dir.iterdir()):
            raise FileExistsError(
                "{} exists; pass --overwrite".format(args.out_dir)
            )
        if args.overwrite:
            shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference, partition_records, states = merge_partitions(args.partitions)
    manifest = {
        key: reference[key]
        for key in CONFIG_KEYS
    }
    manifest.update(
        {
            "state_offset": partition_records[0]["state_offset"],
            "state_limit": len(states),
            "partitions": partition_records,
        }
    )
    json_dump(args.out_dir / "manifest.json", manifest)
    with (args.out_dir / "state_results.jsonl").open("w") as stream:
        for state in states:
            stream.write(json.dumps(state) + "\n")
    summary = summarize_states(states)
    json_dump(args.out_dir / "summary.json", summary)
    write_report(args.out_dir / "report.md", states, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
