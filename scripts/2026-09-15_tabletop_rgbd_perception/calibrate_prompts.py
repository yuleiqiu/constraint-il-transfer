"""Select one prompt per role by oracle recall on the fixed ten layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from experiment_common import (
    DEFAULT_CHECKPOINT,
    ENVIRONMENTS,
    PILOT_EPISODES,
    CorrectedResetEnv,
    dump_json,
    load_state_bank,
    sha256_file,
    state_hash,
)
from models import GroundingDinoDetector, SamVitBSegmenter
from oracle_audit import binary_mask_metrics, geom_ids_by_role, render_oracle_masks
from runtime_io import RgbdSensor, preserve_global_rng
from weights import force_offline_runtime, verify_manifest


CANDIDATES = {
    "target": ("red soda can", "soda can", "can"),
    "goal": ("green tray", "tray", "green container"),
    "milk": ("gray milk carton", "milk carton", "gray box"),
    "support": ("black base", "pedestal", "stand"),
}


def select_prompt(rows, role: str) -> dict:
    summaries = []
    for index, prompt in enumerate(CANDIDATES[role]):
        selected = [row for row in rows if row["role"] == role and row["prompt"] == prompt]
        if len(selected) != len(PILOT_EPISODES):
            raise ValueError(f"incomplete calibration rows for {role}/{prompt}")
        summaries.append(
            {
                "prompt": prompt,
                "candidate_index": index,
                "mean_recall": float(np.mean([row["recall"] for row in selected])),
                "mean_precision": float(np.mean([row["precision"] for row in selected])),
                "mean_iou": float(np.mean([row["iou"] for row in selected])),
                "detected_scenes": int(sum(row["detected"] for row in selected)),
            }
        )
    best = min(
        summaries,
        key=lambda item: (
            -item["mean_recall"],
            -item["mean_precision"],
            item["candidate_index"],
        ),
    )
    return {"selected": best["prompt"], "candidates": summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--state-bank", type=Path, default=ENVIRONMENTS["raised_milk"]["state_bank"]
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)

    manifest = verify_manifest(args.weights / "manifest.json")
    force_offline_runtime()
    detector = GroundingDinoDetector(
        checkpoint=args.weights / "groundingdino_swint_ogc.pth",
        bert_directory=args.weights / "bert-base-uncased",
    )
    segmenter = SamVitBSegmenter(
        checkpoint=args.weights / "sam_vit_b_01ec64.pth", device="cuda"
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    config, _ = FileUtils.config_from_checkpoint(
        algo_name=checkpoint["algo_name"], ckpt_dict=checkpoint, verbose=False
    )
    ObsUtils.initialize_obs_utils_with_config(config)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        env_name=ENVIRONMENTS["raised_milk"]["name"],
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    env = CorrectedResetEnv(env)
    # Calibration uses the explicit audit renderer below; avoid silently
    # executing the policy's oracle point-cloud provider during reset.
    env.base_env.target_pointcloud_config["enabled"] = False
    bank = load_state_bank(args.state_bank)
    rows = []
    captures = args.out / "captures"
    captures.mkdir()
    try:
        for episode in PILOT_EPISODES:
            key = f"{ENVIRONMENTS['raised_milk']['name']}:828:{episode}"
            record = bank["states"][key]
            if state_hash(record["state"]) != record["state_hash"]:
                raise RuntimeError(f"state hash mismatch: {key}")
            env.layout_seed = record["layout_seed"]
            env.reset_to(record["state"])
            sensor = RgbdSensor(env.raw)
            rgbd = sensor.capture()
            role_ids = geom_ids_by_role(env.raw)
            oracle_masks = render_oracle_masks(
                env.raw,
                camera_name=sensor.camera_name,
                height=sensor.height,
                width=sensor.width,
                role_geom_ids=role_ids,
            )
            label_map = np.zeros(rgbd.depth_m.shape, dtype=np.uint8)
            for role_index, role in enumerate(CANDIDATES, start=1):
                if role in oracle_masks:
                    label_map[oracle_masks[role]] = role_index
            np.savez_compressed(
                captures / f"ep{episode:02d}.npz",
                rgb=rgbd.rgb,
                depth_m=rgbd.depth_m,
                intrinsic=rgbd.intrinsic,
                camera_pose=rgbd.camera_pose,
                oracle_labels=label_map,
            )

            selected_boxes = {}
            detection_meta = {}
            with preserve_global_rng():
                for role, prompts in CANDIDATES.items():
                    for candidate_index, prompt in enumerate(prompts):
                        detections = detector.detect(
                            rgbd.rgb,
                            prompt,
                            box_threshold=0.35,
                            text_threshold=0.25,
                        )
                        candidate_key = f"{role}:{candidate_index}"
                        if detections:
                            detection = max(detections, key=lambda item: item.score)
                            selected_boxes[candidate_key] = detection.box_xyxy
                            detection_meta[candidate_key] = detection
                segmented = segmenter.segment(rgbd.rgb, selected_boxes)
            for role, prompts in CANDIDATES.items():
                oracle = oracle_masks[role]
                for candidate_index, prompt in enumerate(prompts):
                    candidate_key = f"{role}:{candidate_index}"
                    predicted = segmented.get(
                        candidate_key, np.zeros_like(oracle, dtype=bool)
                    )
                    scores = binary_mask_metrics(predicted, oracle)
                    detection = detection_meta.get(candidate_key)
                    rows.append(
                        {
                            "episode": episode,
                            "state_hash": record["state_hash"],
                            "role": role,
                            "prompt": prompt,
                            "detected": detection is not None,
                            "detector_score": (
                                None if detection is None else detection.score
                            ),
                            **scores,
                        }
                    )
            print(json.dumps({"calibrated_episode": episode}), flush=True)
    finally:
        env.close()

    selections = {role: select_prompt(rows, role) for role in CANDIDATES}
    locked = {
        "format_version": 1,
        "calibration_environment": ENVIRONMENTS["raised_milk"]["name"],
        "calibration_bank_seed": 828,
        "calibration_episodes": list(PILOT_EPISODES),
        "selection_rule": "maximum mean oracle-mask recall; mean precision tie-break; candidate order final tie-break",
        "box_threshold": 0.35,
        "text_threshold": 0.25,
        "prompts": {role: value["selected"] for role, value in selections.items()},
        "selection_details": selections,
        "weights_manifest_sha256": sha256_file(args.weights / "manifest.json"),
        "weights_manifest": manifest,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "state_bank": str(args.state_bank.resolve()),
        "state_bank_sha256": sha256_file(args.state_bank),
    }
    dump_json(args.out / "calibration_rows.json", rows)
    dump_json(args.out / "locked_config.json", locked)
    print(json.dumps({"selected_prompts": locked["prompts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
