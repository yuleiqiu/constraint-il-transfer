"""Run one balanced mixed Diffusion Policy validation or update step."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from robomimic.algo import algo_factory
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils


DEFAULT_CONFIG = Path(
    "third_party/robomimic/robomimic/exps/targeted_finetuning/"
    "diffusion_policy_can_targeted_mixed.json"
)


def load_config(path):
    external = json.loads(path.read_text(encoding="utf-8"))
    config = config_factory(external["algo_name"])
    with config.values_unlocked():
        config.update(external)
    return config


def sequence_metadata_config(config):
    for dataset_config in config.train.data:
        if dataset_config.get("type", "sequence") == "sequence":
            return dataset_config
    raise ValueError("At least one sequence dataset is required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--sdp-num-samples",
        type=int,
        help="Override algo.sdp.num_samples for a preflight.",
    )
    parser.add_argument(
        "--sdp-start-timestep",
        type=int,
        help="Override algo.sdp.start_timestep for a preflight.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Take one optimizer / EMA update instead of validation only.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batch_size % 2:
        raise ValueError("--batch-size must be a positive even integer")

    config = load_config(args.config.resolve())
    with config.values_unlocked():
        if args.sdp_num_samples is not None:
            config.algo.sdp.num_samples = args.sdp_num_samples
        if args.sdp_start_timestep is not None:
            config.algo.sdp.start_timestep = args.sdp_start_timestep
    torch.manual_seed(config.train.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    ObsUtils.initialize_obs_utils_with_config(config)

    metadata_config = sequence_metadata_config(config)
    shape_metadata = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=metadata_config,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
    )
    train_dataset, valid_dataset = TrainUtils.load_data_for_training(
        config, obs_keys=list(shape_metadata["all_obs_keys"])
    )
    if not hasattr(train_dataset, "datasets") or len(train_dataset.datasets) != 2:
        raise ValueError("Smoke test requires one clean and one correction dataset")

    checkpoint = FileUtils.load_dict_from_checkpoint(
        ckpt_path=config.train.action_normalization_reference
    )
    normalization_check = TrainUtils.set_reference_action_normalization(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        reference_stats=checkpoint["action_normalization_stats"],
    )

    clean_dataset, correction_dataset = train_dataset.datasets
    half_batch = args.batch_size // 2
    samples = [
        train_dataset[index % len(clean_dataset)]
        for index in range(half_batch)
    ]
    correction_sample_indices = (
        torch.linspace(
            0,
            len(correction_dataset) - 1,
            steps=half_batch,
        )
        .round()
        .to(torch.long)
        .tolist()
    )
    samples.extend(
        train_dataset[
            len(clean_dataset) + index
        ]
        for index in correction_sample_indices
    )
    batch = default_collate(samples)
    correction_mask = batch.get("is_paired_correction")
    if config.algo.sdp.enabled:
        if correction_mask is None or "negative_actions" not in batch:
            raise ValueError(
                "SDP smoke test requires correction labels and negative "
                "actions; batch_keys={}, component_negative_flags={}, "
                "meta_negative_flag={}".format(
                    sorted(batch),
                    [
                        getattr(dataset, "return_negative_actions", None)
                        for dataset in train_dataset.datasets
                    ],
                    getattr(
                        train_dataset, "return_negative_actions", None
                    ),
                )
            )
        correction_count = int(correction_mask.sum())
        if correction_count != half_batch:
            raise ValueError(
                "Expected {} correction samples, got {}".format(
                    half_batch, correction_count
                )
            )
        correction_indices = torch.nonzero(
            correction_mask > 0.5, as_tuple=False
        ).flatten()
        shared_history_difference = float(
            (
                batch["actions"][correction_indices, 0]
                - batch["negative_actions"][correction_indices, 0]
            )
            .abs()
            .max()
        )
        if shared_history_difference > 1e-6:
            raise ValueError(
                "Positive and negative correction history actions differ by "
                "{}".format(shared_history_difference)
            )
    else:
        correction_count = None
        shared_history_difference = None

    with config.values_unlocked():
        for optimizer_config in config.algo.optim_params.values():
            optimizer_config["num_train_batches"] = (
                config.experiment.epoch_every_n_steps
            )
            optimizer_config["num_epochs"] = config.train.num_epochs
    config.lock()
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_metadata["all_shapes"],
        ac_dim=shape_metadata["ac_dim"],
        device=device,
    )
    model.deserialize(checkpoint["model"])
    if args.update:
        model.set_train()
    else:
        model.set_eval()
    parameter_versions_before = {
        name: parameter._version
        for name, parameter in model.nets.named_parameters()
    }
    ema_versions_before = {
        name: parameter._version
        for name, parameter in model.ema.averaged_model.named_parameters()
    } if model.ema is not None else {}

    torch.manual_seed(config.train.seed + 1)
    input_batch = model.process_batch_for_training(batch)
    input_batch = model.postprocess_batch_for_training(
        input_batch, obs_normalization_stats=None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    step_start = time.perf_counter()
    info = model.train_on_batch(
        input_batch,
        epoch=1,
        validate=not args.update,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - step_start
    loss = float(info["losses"]["l2_loss"])
    if not math.isfinite(loss):
        raise ValueError("Non-finite validation loss")
    changed_parameters = [
        name
        for name, parameter in model.nets.named_parameters()
        if parameter._version != parameter_versions_before[name]
    ]
    changed_ema_parameters = [
        name
        for name, parameter in model.ema.averaged_model.named_parameters()
        if parameter._version != ema_versions_before[name]
    ] if model.ema is not None else []
    if not args.update and (changed_parameters or changed_ema_parameters):
        raise RuntimeError(
            "Validation-only smoke test mutated parameters: {}".format(
                (changed_parameters + changed_ema_parameters)[:10]
            )
        )
    if args.update and not changed_parameters:
        raise RuntimeError("Optimizer smoke test did not mutate online parameters")
    if args.update and model.ema is not None and not changed_ema_parameters:
        raise RuntimeError("Optimizer smoke test did not mutate EMA parameters")

    print(
        json.dumps(
            {
                "status": "pass",
                "device": str(device),
                "clean_samples": half_batch,
                "correction_samples": half_batch,
                "correction_label_count": correction_count,
                "shared_history_maximum_difference": (
                    shared_history_difference
                ),
                "collated_observation_steps": int(
                    batch["obs"][next(iter(batch["obs"]))].shape[1]
                ),
                "consumed_observation_steps": int(
                    input_batch["obs"][
                        next(iter(input_batch["obs"]))
                    ].shape[1]
                ),
                "collated_action_steps": int(batch["actions"].shape[1]),
                "consumed_action_steps": int(
                    input_batch["actions"].shape[1]
                ),
                "normalized_action_minimum": float(
                    input_batch["actions"].min()
                ),
                "normalized_action_maximum": float(
                    input_batch["actions"].max()
                ),
                "normalization_check": normalization_check,
                "loss": loss,
                "step_mode": "update" if args.update else "validation",
                "step_seconds": step_seconds,
                "peak_gpu_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    if device.type == "cuda"
                    else None
                ),
                "mutated_parameter_count": len(changed_parameters),
                "mutated_ema_parameter_count": len(
                    changed_ema_parameters
                ),
                "sdp": {
                    key: float(value)
                    for key, value in info.get("sdp", {}).items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
