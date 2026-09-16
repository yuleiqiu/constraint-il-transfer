"""Pinned GroundingDINO, SAM, and Cutie adapters.

Heavy optional imports are intentionally local to constructors so the pure
perception tests run without the ``perception`` uv extra.  Every constructor
accepts explicit local weights; none is allowed to download at runtime.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert
from torchvision.transforms.functional import to_tensor

from perception import Detection


class GroundingDinoDetector:
    """Official Swin-T OGC detector with Lan-o3dp's CPU execution path."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        bert_directory: Path,
        config_path: Path | None = None,
    ) -> None:
        checkpoint = Path(checkpoint).resolve()
        bert_directory = Path(bert_directory).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not bert_directory.is_dir():
            raise FileNotFoundError(bert_directory)
        from groundingdino.datasets import transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig

        if config_path is None:
            package = importlib.import_module("groundingdino")
            config_path = (
                Path(package.__file__).resolve().parent
                / "config"
                / "GroundingDINO_SwinT_OGC.py"
            )
        if not Path(config_path).is_file():
            raise FileNotFoundError(config_path)
        args = SLConfig.fromfile(str(config_path))
        args.device = "cpu"
        # Avoid the otherwise implicit Hugging Face lookup performed while the
        # DINO graph is constructed.  The model checkpoint replaces these
        # weights, but construction still needs a complete local BERT snapshot.
        args.text_encoder_type = str(bert_directory)
        self.model = build_model(args)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(clean_state_dict(payload["model"]), strict=False)
        self.model.eval()
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def detect(
        self,
        rgb: np.ndarray,
        prompt: str,
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> list[Detection]:
        from groundingdino.util.inference import predict

        rgb = np.asarray(rgb, dtype=np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        transformed, _ = self.transform(image, None)
        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=transformed,
                caption=prompt,
                box_threshold=float(box_threshold),
                text_threshold=float(text_threshold),
                device="cpu",
            )
        height, width = rgb.shape[:2]
        boxes_xyxy = box_convert(
            boxes * torch.tensor([width, height, width, height]),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ).numpy()
        return [
            Detection(
                box_xyxy=np.asarray(box, dtype=np.float32),
                score=float(score),
                phrase=str(phrase),
            )
            for box, score, phrase in zip(boxes_xyxy, logits, phrases)
        ]


class SamVitBSegmenter:
    """Official SAM ViT-B predictor, batching one selected box per role."""

    def __init__(self, *, checkpoint: Path, device: str = "cuda") -> None:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        from segment_anything import SamPredictor, sam_model_registry

        self.device = torch.device(device)
        model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        self.predictor = SamPredictor(model.to(device=self.device).eval())

    def segment(
        self,
        rgb: np.ndarray,
        boxes_by_role: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if not boxes_by_role:
            return {}
        roles = list(boxes_by_role)
        boxes = torch.as_tensor(
            np.stack([boxes_by_role[role] for role in roles]),
            dtype=torch.float32,
            device=self.device,
        )
        self.predictor.set_image(np.asarray(rgb, dtype=np.uint8))
        transformed = self.predictor.transform.apply_boxes_torch(
            boxes, np.asarray(rgb).shape[:2]
        )
        with torch.inference_mode():
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )
        values = masks[:, 0].detach().cpu().numpy().astype(bool)
        return {role: values[index] for index, role in enumerate(roles)}


class _CutieTracker:
    def __init__(self, network, cfg, device: torch.device) -> None:
        from cutie.inference.inference_core import InferenceCore

        self.network = network
        self.cfg = cfg
        self.device = device
        self.processor = InferenceCore(network, cfg=cfg)
        self.initialized = False

    def _image_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        return to_tensor(Image.fromarray(np.asarray(rgb, dtype=np.uint8))).to(
            self.device, dtype=torch.float32
        )

    @torch.inference_mode()
    def initialize(self, rgb: np.ndarray, label_map: np.ndarray) -> None:
        labels = np.asarray(label_map, dtype=np.uint8)
        objects = [int(value) for value in np.unique(labels) if value != 0]
        if not objects:
            raise ValueError("Cutie initialization requires at least one instance")
        image = self._image_tensor(rgb)
        mask = torch.from_numpy(labels.astype(np.int64)).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            self.processor.step(image, mask, objects=objects)
        self.initialized = True

    @torch.inference_mode()
    def track(self, rgb: np.ndarray) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Cutie tracker has not been initialized")
        image = self._image_tensor(rgb)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            probabilities = self.processor.step(image)
            labels = self.processor.output_prob_to_mask(probabilities)
        return labels.detach().cpu().numpy().astype(np.uint8)


class CutieTrackerFactory:
    """Load Cutie's official default model once, with downloads disabled."""

    def __init__(self, *, checkpoint: Path, device: str = "cuda") -> None:
        checkpoint = Path(checkpoint).resolve()
        if checkpoint.name != "cutie-base-mega.pth" or not checkpoint.is_file():
            raise FileNotFoundError(
                "Cutie default loading requires an explicit cutie-base-mega.pth"
            )
        target_device = torch.device(device)
        if target_device.type != "cuda":
            raise ValueError("Lan-o3dp's Cutie default model requires CUDA")

        # The official helper otherwise calls download_models_if_needed().
        # Replace only that lookup for the duration of construction, while
        # retaining the official default Hydra config and weight loader.
        from hydra.core.global_hydra import GlobalHydra

        module = importlib.import_module("cutie.utils.get_default_model")
        cutie_resnet = importlib.import_module("cutie.model.utils.resnet")
        original_download = module.download_models_if_needed
        original_load_url = cutie_resnet.model_zoo.load_url

        def load_local_resnet(url, model_dir=None, **kwargs):
            path = checkpoint.parent / Path(url).name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cutie attempted an undeclared backbone download: {path}"
                )
            return torch.load(path, map_location="cpu", weights_only=True)

        GlobalHydra.instance().clear()
        module.download_models_if_needed = lambda: str(checkpoint.parent)
        cutie_resnet.model_zoo.load_url = load_local_resnet
        try:
            self.network = module.get_default_model()
        finally:
            module.download_models_if_needed = original_download
            cutie_resnet.model_zoo.load_url = original_load_url
            GlobalHydra.instance().clear()
        self.network = self.network.to(target_device).eval()
        self.cfg = self.network.cfg
        self.device = target_device

    def create(self) -> _CutieTracker:
        return _CutieTracker(self.network, self.cfg, self.device)


@dataclass(frozen=True)
class ModelBundle:
    detector: GroundingDinoDetector
    segmenter: SamVitBSegmenter
    tracker_factory: CutieTrackerFactory


def load_model_bundle(weights_directory: Path, *, device: str = "cuda") -> ModelBundle:
    weights_directory = Path(weights_directory).resolve()
    return ModelBundle(
        detector=GroundingDinoDetector(
            checkpoint=weights_directory / "groundingdino_swint_ogc.pth",
            bert_directory=weights_directory / "bert-base-uncased",
        ),
        segmenter=SamVitBSegmenter(
            checkpoint=weights_directory / "sam_vit_b_01ec64.pth",
            device=device,
        ),
        tracker_factory=CutieTrackerFactory(
            checkpoint=weights_directory / "cutie-base-mega.pth",
            device=device,
        ),
    )
