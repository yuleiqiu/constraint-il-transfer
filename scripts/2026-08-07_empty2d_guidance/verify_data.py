"""Verify Empty2D HDF5 sequence, padding, masks, and manifest hash."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from empty2d_guidance.data import NUM_POINTS, NUM_TRAJECTORIES
from empty2d_guidance.environment import GOAL, START


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("outputs/empty2d_guidance/data/empty2d_guidance.hdf5"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/empty2d_guidance/data/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert sha256(args.dataset) == manifest["hdf5"]["sha256"]
    with h5py.File(args.dataset, "r") as source:
        assert len(source["data"]) == NUM_TRAJECTORIES
        train = [item.decode() for item in source["mask/train"][:]]
        valid = [item.decode() for item in source["mask/valid"][:]]
        assert len(train) == 90 and len(valid) == 10 and set(train).isdisjoint(valid)
        for name in train + valid:
            demo = source[f"data/{name}"]
            obs = demo["obs/agent_pos"][:]
            actions = demo["actions"][:]
            assert obs.shape == actions.shape == (NUM_POINTS, 2)
            np.testing.assert_array_equal(obs[0], START.astype(np.float32))
            np.testing.assert_array_equal(obs[-1], GOAL.astype(np.float32))
            np.testing.assert_array_equal(actions[:-1], obs[1:])
            np.testing.assert_array_equal(actions[-1], GOAL.astype(np.float32))
            assert np.linalg.norm(np.diff(obs, axis=0), axis=1).max() <= 0.0300001
    print("Empty2D dataset verification passed")


if __name__ == "__main__":
    main()
