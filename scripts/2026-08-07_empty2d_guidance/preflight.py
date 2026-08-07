"""One real batch-256 update, reload, and custom-environment action smoke."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from robomimic.algo import algo_factory
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils

from empty2d_guidance.environment import EnvEmpty2D
from empty2d_guidance.environment import EnvEmpty2DExtraSphere


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("third_party/robomimic/robomimic/exps/empty2d_guidance/diffusion_policy_sample_seed42.json"),
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/empty2d_guidance_preflight.pth"))
    args = parser.parse_args()
    external = json.loads(args.config.read_text())
    config = config_factory(external["algo_name"])
    with config.values_unlocked():
        config.update(external)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    if device.type != "cuda":
        raise RuntimeError("Empty2D batch-256 preflight requires CUDA")
    ObsUtils.initialize_obs_utils_with_config(config)
    dataset_config = config.train.data[0]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=dataset_config,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_config["path"])
    trainset, validset = TrainUtils.load_data_for_training(
        config,
        obs_keys=shape_meta["all_obs_keys"],
    )
    obs_stats = trainset.get_obs_normalization_stats()
    action_stats = trainset.get_action_normalization_stats()
    loader = DataLoader(
        trainset,
        batch_size=config.train.batch_size,
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
    peak = torch.cuda.max_memory_allocated(device)
    if not np.isfinite(loss):
        raise AssertionError("Preflight loss is not finite")
    TrainUtils.save_model(
        model=model,
        config=config,
        env_meta=env_meta,
        shape_meta=shape_meta,
        ckpt_path=str(args.checkpoint),
        obs_normalization_stats=obs_stats,
        action_normalization_stats=action_stats,
    )
    parameter_count = sum(parameter.numel() for parameter in model.nets.parameters())
    del batch, raw_batch, info, model, loader, trainset, validset
    gc.collect()
    torch.cuda.empty_cache()
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(args.checkpoint), device=device)
    policy.start_episode()
    environment = EnvEmpty2D()
    observation = environment.reset()
    torch.manual_seed(42)
    action = policy(
        {"agent_pos": np.stack((observation["agent_pos"], observation["agent_pos"]))}
    )
    assert action.shape == (2,) and np.isfinite(action).all()
    environment.step(action)
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    from robomimic.utils.paper_lan_guidance_utils import (
        point_guided_policy_from_checkpoint,
        point_trajectory_guidance_context_from_rollout_policy,
    )

    guided, _ = point_guided_policy_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=device,
    )
    guided.start_episode()
    sphere = EnvEmpty2DExtraSphere()
    guided_observation = sphere.reset()
    guided.policy.set_guidance_context(
        point_trajectory_guidance_context_from_rollout_policy(
            guided,
            current_position=sphere.position,
            obstacle_points=sphere.oracle_obstacle_points(),
            guidance_scale=1e-4,
            safety_distance=10.0,
            action_key="actions",
            position_indices=(0, 1),
        )
    )
    torch.manual_seed(42)
    guided_action = guided(
        {
            "agent_pos": np.stack(
                (guided_observation["agent_pos"], guided_observation["agent_pos"])
            )
        }
    )
    diagnostics = guided.policy.get_guidance_diagnostics()
    assert guided_action.shape == (2,) and np.isfinite(guided_action).all()
    assert len(diagnostics) == 100
    assert all(np.isfinite(item["applied_update_norm"]) for item in diagnostics)
    print(
        json.dumps(
            {
                "parameter_count": parameter_count,
                "batch_size": config.train.batch_size,
                "loss": loss,
                "peak_cuda_memory_gib": peak / 1024**3,
                "action": action.tolist(),
                "guided_action": guided_action.tolist(),
                "guided_diagnostic_steps": len(diagnostics),
                "checkpoint": str(args.checkpoint),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
