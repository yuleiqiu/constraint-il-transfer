"""Overfit one fixed point-cloud batch as a Phase-1 training-chain smoke test."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from robomimic.algo import algo_factory
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "third_party/robomimic/robomimic/exps/delta_eef_pose_osc/"
    "diffusion_policy_can_pointcloud_ddpm100.json"
)


def fixed_loss(model, batch, seed):
    torch.manual_seed(seed)
    info = model.train_on_batch(batch, epoch=1, validate=True)
    return float(info["losses"]["l2_loss"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--noise-seed", type=int, default=123)
    parser.add_argument("--minimum-relative-drop", type=float, default=0.2)
    args = parser.parse_args()

    with args.config.open() as stream:
        external = json.load(stream)
    config = config_factory(external["algo_name"])
    with config.values_unlocked():
        config.update(external)
        if args.dataset is not None:
            config.train.data = [{"path": str(args.dataset.resolve())}]
        config.experiment.validate = False
        config.train.hdf5_validation_filter_key = None
        config.train.batch_size = args.batch_size
        for optimizer in config.algo.optim_params.values():
            optimizer["num_train_batches"] = args.steps
            optimizer["num_epochs"] = 1
    config.lock()

    ObsUtils.initialize_obs_utils_with_config(config)
    dataset_config = config.train.data[0]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=dataset_config,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
    )
    trainset, _ = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    trainset.get_action_normalization_stats()
    loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    raw_batch = next(iter(loader))
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    batch = model.process_batch_for_training(raw_batch)
    batch = model.postprocess_batch_for_training(batch, obs_normalization_stats=None)

    model.set_train()
    initial = fixed_loss(model, batch, args.noise_seed)
    for step in range(args.steps):
        torch.manual_seed(args.noise_seed)
        model.train_on_batch(batch, epoch=1, validate=False)
        model.on_gradient_step()
    final = fixed_loss(model, batch, args.noise_seed)
    relative_drop = (initial - final) / initial
    if not final < initial or relative_drop < args.minimum_relative_drop:
        raise AssertionError(
            "Fixed-batch loss did not fall enough: initial={} final={} drop={}".format(
                initial, final, relative_drop
            )
        )
    print(
        "one-batch overfit passed: initial={:.6f} final={:.6f} relative_drop={:.1%}".format(
            initial, final, relative_drop
        )
    )


if __name__ == "__main__":
    main()
