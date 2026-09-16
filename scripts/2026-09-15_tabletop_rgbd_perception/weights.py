"""Download once and verify every perception model artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LAN_O3DP_REVISION = "b89c2898d8548f7dfde63c44eeb733a59f0c5222"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
SOURCES = (
    {
        "component": "GroundingDINO Swin-T OGC",
        "repository": "https://github.com/IDEA-Research/GroundingDINO",
        "revision": "856dde20aee659246248e20734ef9ba5214f5e44",
        "relative_path": "groundingdino_swint_ogc.pth",
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    },
    {
        "component": "Segment Anything ViT-B",
        "repository": "https://github.com/facebookresearch/segment-anything",
        "revision": "dca509fe793f601edb92606367a655c15ac00fdf",
        "relative_path": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
    {
        "component": "Cutie default base mega",
        "repository": "https://github.com/hkchengrex/Cutie",
        "revision": "ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a",
        "release": "v1.0",
        "relative_path": "cutie-base-mega.pth",
        "url": "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
    },
    {
        "component": "Cutie pixel encoder ResNet-50 initialization",
        "repository": "https://github.com/pytorch/vision",
        "revision": "3ac97aa9120137381ed1060f37237e44485ac2aa",
        "relative_path": "resnet50-19c8e357.pth",
        "url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    },
    {
        "component": "Cutie mask encoder ResNet-18 initialization",
        "repository": "https://github.com/pytorch/vision",
        "revision": "3ac97aa9120137381ed1060f37237e44485ac2aa",
        "relative_path": "resnet18-5c106cde.pth",
        "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    },
    *(
        {
            "component": "BERT base uncased construction snapshot",
            "repository": "https://huggingface.co/google-bert/bert-base-uncased",
            "revision": BERT_REVISION,
            "relative_path": f"bert-base-uncased/{filename}",
            "url": (
                "https://huggingface.co/google-bert/bert-base-uncased/resolve/"
                f"{BERT_REVISION}/{filename}"
            ),
        }
        for filename in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        )
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "constraint-il-transfer/0.1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as stream:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            stream.write(block)
    temporary.replace(destination)


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def create_manifest(directory: Path, *, download: bool) -> dict[str, Any]:
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in SOURCES:
        path = directory / source["relative_path"]
        if not path.is_file():
            if not download:
                raise FileNotFoundError(
                    f"missing {path}; run weights.py --download before formal execution"
                )
            print(f"Downloading {source['component']} -> {path}", flush=True)
            _atomic_download(source["url"], path)
        entry = dict(source)
        entry.update(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        entries.append(entry)
    packages = {}
    for name in ("groundingdino", "segment-anything", "cutie", "torch", "torchvision"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "format_version": 1,
        "formal_runtime_network_policy": "offline; every file must match this manifest",
        "lan_o3dp_repository": "https://github.com/qqianfeng/lan-o3dp",
        "lan_o3dp_revision": LAN_O3DP_REVISION,
        "root_git_head": _git_head(ROOT),
        "package_versions": packages,
        "files": entries,
    }
    manifest_path = directory / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported perception weight manifest")
    for entry in manifest.get("files", []):
        path = Path(entry["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry["size_bytes"]:
            raise RuntimeError(f"weight size mismatch: {path}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise RuntimeError(f"weight SHA256 mismatch: {path}")
    return manifest


def force_offline_runtime() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest_path = args.directory.resolve() / "manifest.json"
    if args.verify and manifest_path.is_file() and not args.download:
        manifest = verify_manifest(manifest_path)
    else:
        manifest = create_manifest(args.directory, download=args.download)
        verify_manifest(manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
