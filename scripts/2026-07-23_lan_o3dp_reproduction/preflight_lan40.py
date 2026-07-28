"""Run the fixed batch-64 LAN-40 memory and checkpoint smoke gate."""

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
    / "third_party/robomimic/robomimic/exps/delta_eef_pose_osc/"
    "lan_o3dp_can_delta_pose_eps_residual_40demo_seed42.json"
)
DEFAULT_CHECKPOINT = Path("/tmp/lan_o3dp_40demo_preflight.pth")


def load_config(path):
    with path.open() as stream:
        external = json.load(stream)
    config = config_factory(external["algo_name"])
    with config.values_unlocked():
        config.update(external)
    return config


def first_stacked_observation(dataset_path, demo_name):
    keys = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "task_pointcloud",
    )
    with h5py.File(dataset_path, "r") as source:
        return {
            key: np.repeat(
                np.asarray(source["data/{}/obs/{}".format(demo_name, key)][0])[None],
                2,
                axis=0,
            )
            for key in keys
        }


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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.train.batch_size != 64:
        raise AssertionError(
            "The preregistered memory gate requires batch 64, got {}".format(
                config.train.batch_size
            )
        )
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    if device.type != "cuda":
        raise RuntimeError("The batch-64 preflight requires CUDA")

    ObsUtils.initialize_obs_utils_with_config(config)
    dataset_config = config.train.data[0]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=dataset_config,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path=dataset_config["path"]
    )
    trainset, _ = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    obs_normalization_stats = trainset.get_obs_normalization_stats()
    action_normalization_stats = trainset.get_action_normalization_stats()
    loader = DataLoader(
        dataset=trainset,
        sampler=trainset.get_dataset_sampler(),
        batch_size=64,
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
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.nets.parameters())
    batch = model.process_batch_for_training(raw_batch)
    batch = model.postprocess_batch_for_training(
        batch, obs_normalization_stats=obs_normalization_stats
    )
    model.set_train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    info = model.train_on_batch(batch, epoch=1, validate=False)
    model.on_gradient_step()
    torch.cuda.synchronize(device)
    loss = float(info["losses"]["l2_loss"])
    peak_bytes = torch.cuda.max_memory_allocated(device)
    if not np.isfinite(loss):
        raise AssertionError("Non-finite batch-64 training loss")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    TrainUtils.save_model(
        model=model,
        config=config,
        env_meta=env_meta,
        shape_meta=shape_meta,
        ckpt_path=str(args.checkpoint),
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
    )
    demo_name = trainset.demos[0]
    stacked_observation = first_stacked_observation(
        Path(dataset_config["path"]), demo_name
    )

    del batch, raw_batch, info, model, loader, trainset
    gc.collect()
    torch.cuda.empty_cache()

    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=device,
    )
    if policy.policy.__class__.__name__ != "LanO3DPUNet":
        raise AssertionError("Checkpoint did not reload as LanO3DPUNet")
    policy.start_episode()
    action = policy(ob=stacked_observation)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise AssertionError(
            "Reloaded policy emitted invalid action shape/value: {}".format(action)
        )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=checkpoint,
        render=False,
        render_offscreen=True,
    )
    try:
        policy.start_episode()
        observation = env.reset()
        rollout_action = policy(ob=observation)
        if rollout_action.shape != (7,) or not np.isfinite(rollout_action).all():
            raise AssertionError("Environment rollout action is not finite 7-D")
        env.step(rollout_action)
    finally:
        close_wrapped_env(env)

    print("batch_size: 64")
    print("parameter_count: {}".format(parameter_count))
    print("loss: {:.8f}".format(loss))
    print("peak_cuda_memory_gib: {:.3f}".format(peak_bytes / (1024 ** 3)))
    print("checkpoint: {}".format(args.checkpoint))
    print("reload_class: LanO3DPUNet")
    print("finite_dataset_action: {}".format(action.tolist()))
    print("finite_rollout_action: {}".format(rollout_action.tolist()))


if __name__ == "__main__":
    main()
