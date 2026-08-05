"""Run one real training update, checkpoint reload, and online action smoke."""

import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
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
    / "third_party/robomimic/robomimic/exps/absolute_eef_osc/"
    "lan_o3dp_can_abs_pose_sample_paper256_seed42.json"
)


def load_config(path, batch_size):
    external = json.loads(path.read_text())
    config = config_factory(external["algo_name"])
    with config.values_unlocked():
        config.update(external)
        config.train.batch_size = batch_size
    return config


def close_wrapped_env(env):
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        close = getattr(current, "close", None)
        if callable(close):
            close()
            return
        if not hasattr(current, "env"):
            return
        current = current.env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/lan_o3dp_paper_preflight.pth"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    config = load_config(args.config.resolve(), args.batch_size)
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    if device.type != "cuda":
        raise RuntimeError("The paper architecture preflight requires CUDA")

    ObsUtils.initialize_obs_utils_with_config(config)
    dataset_config = config.train.data[0]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=dataset_config,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_config["path"])
    trainset, _ = TrainUtils.load_data_for_training(
        config,
        obs_keys=shape_meta["all_obs_keys"],
    )
    obs_stats = trainset.get_obs_normalization_stats()
    action_stats = trainset.get_action_normalization_stats()
    loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    with config.unlocked():
        for optimizer in config.algo.optim_params.values():
            optimizer["num_train_batches"] = len(loader)
            optimizer["num_epochs"] = config.train.num_epochs
    config.lock()

    raw_batch = next(iter(loader))
    model = algo_factory(
        config.algo_name,
        config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    batch = model.process_batch_for_training(raw_batch)
    batch = model.postprocess_batch_for_training(batch, obs_normalization_stats=obs_stats)
    model.set_train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    info = model.train_on_batch(batch, epoch=1, validate=False)
    model.on_gradient_step()
    torch.cuda.synchronize(device)
    loss = float(info["losses"]["l2_loss"])
    peak_bytes = torch.cuda.max_memory_allocated(device)
    if not np.isfinite(loss):
        raise AssertionError("Non-finite training loss")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    TrainUtils.save_model(
        model=model,
        config=config,
        env_meta=env_meta,
        shape_meta=shape_meta,
        ckpt_path=str(args.checkpoint),
        obs_normalization_stats=obs_stats,
        action_normalization_stats=action_stats,
    )
    del batch, raw_batch, info, model, loader, trainset
    gc.collect()
    torch.cuda.empty_cache()

    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=device,
    )
    assert policy.policy.__class__.__name__ == "LanO3DPUNet"
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        render=False,
        render_offscreen=True,
    )
    try:
        policy.start_episode()
        observation = env.reset()
        assert observation["agent_pos"].shape[-1:] == (6,)
        assert observation["task_pointcloud"].shape[-2:] == (256, 3)
        action = policy(ob=observation)
        assert action.shape == (7,) and np.isfinite(action).all()
        env.step(action)
    finally:
        close_wrapped_env(env)

    print("prediction_type: {}".format(config.algo.ddpm.prediction_type))
    print("batch_size: {}".format(args.batch_size))
    print("loss: {:.8f}".format(loss))
    print("peak_cuda_memory_gib: {:.3f}".format(peak_bytes / (1024 ** 3)))
    print("checkpoint: {}".format(args.checkpoint))
    print("finite_online_action: {}".format(action.tolist()))


if __name__ == "__main__":
    main()
