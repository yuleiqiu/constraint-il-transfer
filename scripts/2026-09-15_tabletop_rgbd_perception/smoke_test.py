"""One-scene end-to-end GPU and repeatability preflight."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from experiment_common import (
    DEFAULT_CHECKPOINT,
    ENVIRONMENTS,
    CorrectedResetEnv,
    dump_json,
    load_state_bank,
    state_hash,
)
from models import load_model_bundle
from oracle_audit import audit_perception_frame, geom_ids_by_role, render_oracle_masks
from perception import PerceptionProvider, ProviderConfig
from runtime_io import RgbdSensor, preserve_global_rng, replace_policy_pointcloud
from weights import force_offline_runtime, verify_manifest


ROOT = Path(__file__).resolve().parents[2]
PAIRED_EVAL_PATH = ROOT / "scripts/2026-08-05_lan_o3dp_paper_aligned/paired_eval.py"
SPEC = importlib.util.spec_from_file_location("perception_smoke_paired", PAIRED_EVAL_PATH)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def run_provider(bundle, config, rgbd, frames: int):
    provider = PerceptionProvider(
        detector=bundle.detector,
        segmenter=bundle.segmenter,
        tracker_factory=bundle.tracker_factory,
        config=config,
    )
    with preserve_global_rng():
        initial = provider.start_episode(rgbd)
    if not initial.valid:
        raise RuntimeError(f"initial perception failed: {initial.diagnostics}")
    tracked = []
    for _ in range(frames):
        with preserve_global_rng():
            tracked.append(provider.track(rgbd))
        if not tracked[-1].valid:
            raise RuntimeError(f"Cutie smoke tracking failed: {tracked[-1].diagnostics}")
    return provider, initial, tracked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--state-bank", type=Path, default=ENVIRONMENTS["raised_milk"]["state_bank"]
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--locked-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--tracking-frames", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    verify_manifest(args.weights / "manifest.json")
    force_offline_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    initial_allocated = torch.cuda.memory_allocated()

    policy, checkpoint = P.paper_guided_policy_from_checkpoint(
        ckpt_path=str(args.checkpoint.resolve()), device=torch.device("cuda")
    )
    bundle = load_model_bundle(args.weights, device="cuda")
    model_allocated = torch.cuda.memory_allocated()
    env, _ = P.FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=ENVIRONMENTS["raised_milk"]["name"],
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    env = CorrectedResetEnv(env)
    env.base_env.target_pointcloud_config["enabled"] = False
    bank = load_state_bank(args.state_bank)
    key = f"{ENVIRONMENTS['raised_milk']['name']}:828:{args.episode}"
    record = bank["states"][key]
    if state_hash(record["state"]) != record["state_hash"]:
        raise RuntimeError("smoke state hash mismatch")
    env.layout_seed = record["layout_seed"]
    observation = env.reset_to(record["state"])
    if "task_pointcloud" in observation:
        raise RuntimeError("oracle provider was active in perceived smoke test")
    sensor = RgbdSensor(env.raw)
    rgbd = sensor.capture()
    locked = json.loads(args.locked_config.read_text())
    config = ProviderConfig(
        prompts=locked["prompts"],
        scene_has_obstacle=True,
        box_threshold=locked["box_threshold"],
        text_threshold=locked["text_threshold"],
    )
    first_provider, first_initial, first_tracks = run_provider(
        bundle, config, rgbd, args.tracking_frames
    )
    second_provider, second_initial, second_tracks = run_provider(
        bundle, config, rgbd, args.tracking_frames
    )
    initial_equal = np.array_equal(first_initial.task_points, second_initial.task_points)
    masks_equal = all(
        np.array_equal(first_initial.masks[role], second_initial.masks[role])
        for role in first_initial.masks
    ) and set(first_initial.masks) == set(second_initial.masks)
    tracks_equal = all(
        np.array_equal(left.task_points, right.task_points)
        and all(np.array_equal(left.masks[role], right.masks[role]) for role in left.masks)
        for left, right in zip(first_tracks, second_tracks)
    )
    if not (initial_equal and masks_equal and tracks_equal):
        raise RuntimeError("identical RGB input did not reproduce perception outputs")

    observation = replace_policy_pointcloud(observation, first_provider)
    torch.manual_seed(828000000 + args.episode)
    torch.cuda.manual_seed_all(828000000 + args.episode)
    policy.start_episode()
    raw_chunk, _ = P.sample_raw_chunk(policy, observation, None)
    if raw_chunk.shape != (8, 7) or not np.isfinite(raw_chunk).all():
        raise RuntimeError(f"invalid frozen-policy smoke chunk: {raw_chunk.shape}")

    oracle_masks = render_oracle_masks(
        env.raw,
        camera_name=sensor.camera_name,
        height=sensor.height,
        width=sensor.width,
        role_geom_ids=geom_ids_by_role(env.raw),
    )
    audit = audit_perception_frame(first_initial, rgbd, oracle_masks)
    peak_allocated = torch.cuda.max_memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    result = {
        "passed": True,
        "episode": args.episode,
        "initial_state_hash": record["state_hash"],
        "dino_sam_initial_valid": first_initial.valid,
        "cutie_tracking_frames": args.tracking_frames,
        "identical_initial_output": initial_equal and masks_equal,
        "identical_tracking_output": tracks_equal,
        "policy_chunk_shape": list(raw_chunk.shape),
        "policy_actions_finite": True,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": {
            "initial_allocated": initial_allocated,
            "models_allocated": model_allocated,
            "peak_allocated": peak_allocated,
            "device_total": total_memory,
            "peak_fraction": peak_allocated / total_memory,
        },
        "initial_perception": first_initial.diagnostics,
        "initial_oracle_audit": audit.diagnostics,
    }
    dump_json(args.out / "result.json", result)
    np.savez_compressed(
        args.out / "frame_and_masks.npz",
        rgb=rgbd.rgb,
        depth_m=rgbd.depth_m,
        **{f"perceived_{role}": mask for role, mask in first_initial.masks.items()},
        **{f"oracle_{role}": mask for role, mask in oracle_masks.items()},
    )
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    del policy, checkpoint, bundle, first_provider, second_provider
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
