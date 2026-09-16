"""RGB-D capture, RNG isolation, and policy-observation replacement."""

from __future__ import annotations

import contextlib
import random
from typing import Iterator

import numpy as np
import torch
from robosuite.utils import camera_utils as CameraUtils

from perception import PerceptionProvider, RgbdFrame, array_sha256


class RgbdSensor:
    """Capture only allowed deployment inputs from one fixed MuJoCo camera."""

    def __init__(
        self,
        raw_env,
        *,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> None:
        self._raw_env = raw_env
        self.camera_name = camera_name
        self.height = int(height)
        self.width = int(width)
        self.intrinsic = CameraUtils.get_camera_intrinsic_matrix(
            raw_env.sim,
            camera_name=camera_name,
            camera_height=self.height,
            camera_width=self.width,
        ).astype(np.float64)
        self.camera_pose = CameraUtils.get_camera_extrinsic_matrix(
            raw_env.sim, camera_name=camera_name
        ).astype(np.float64)

    def capture(self) -> RgbdFrame:
        rgb, normalized_depth = self._raw_env.sim.render(
            camera_name=self.camera_name,
            height=self.height,
            width=self.width,
            depth=True,
            segmentation=False,
        )
        rgb = np.asarray(rgb[::-1], dtype=np.uint8)
        normalized_depth = np.asarray(normalized_depth[::-1])
        depth_m = CameraUtils.get_real_depth_map(
            self._raw_env.sim, normalized_depth
        ).astype(np.float32)
        return RgbdFrame(
            rgb=rgb,
            depth_m=depth_m,
            intrinsic=self.intrinsic.copy(),
            camera_pose=self.camera_pose.copy(),
        )

    def calibration_hashes(self) -> dict[str, str]:
        return {
            "intrinsic": array_sha256(self.intrinsic),
            "camera_pose": array_sha256(self.camera_pose),
        }


def replace_policy_pointcloud(
    observation: dict[str, np.ndarray], provider: PerceptionProvider
) -> dict[str, np.ndarray]:
    """Replace only task_pointcloud; proprioception remains checkpoint-native."""

    result = {key: np.asarray(value).copy() for key, value in observation.items()}
    perceived = provider.policy_task_pointcloud()
    if perceived.shape != (2, 256, 3):
        raise ValueError(f"perceived pointcloud contract changed: {perceived.shape}")
    if "task_pointcloud" in result and np.asarray(result["task_pointcloud"]).shape != perceived.shape:
        raise ValueError(
            "existing task_pointcloud shape disagrees with perceived history"
        )
    result["task_pointcloud"] = perceived.copy()
    return result


@contextlib.contextmanager
def preserve_global_rng() -> Iterator[None]:
    """Keep perception kernels from advancing the frozen policy RNG stream."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
