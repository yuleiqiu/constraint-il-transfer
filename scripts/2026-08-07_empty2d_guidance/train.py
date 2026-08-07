"""Fail-fast Empty2D training launcher for seeds 42, 43, or 44."""

import argparse
import json
from pathlib import Path

import torch

import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("third_party/robomimic/robomimic/exps/empty2d_guidance/diffusion_policy_sample_seed42.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(Path("outputs/empty2d_guidance/data/manifest.json").read_text())
    if not manifest["validation"]["passed"]:
        raise RuntimeError("Refusing to train because the data gate did not pass")
    external = json.loads(args.config.read_text())
    external["train"]["seed"] = args.seed
    external["experiment"]["name"] = f"empty2d_guidance_sample_seed{args.seed}"
    config = config_factory(external["algo_name"])
    with config.values_unlocked():
        config.update(external)
    config.lock()
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal Empty2D training requires CUDA")
    train(config, device=device, resume=False)


if __name__ == "__main__":
    main()
