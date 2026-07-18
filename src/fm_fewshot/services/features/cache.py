"""Feature cache per ADR-001 and ADR-004.

Features are stored raw float32; L2 normalization happens at read time so one
cache serves both settings and the normalization axis stays a config field.
Each split gets its own npz plus a meta json carrying a sha256 of the npz
payload; the checksum makes scp transfer between machines safe.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from fm_fewshot.services.data.datasets import load_split
from fm_fewshot.shared import gatekeeper


class CacheChecksumError(RuntimeError):
    """Raised when a cache file does not match the checksum in its meta."""


def cache_dir(data_root: Path, dataset: str, encoder_name: str) -> Path:
    return Path(data_root) / "features" / f"{dataset}_{encoder_name}"


def _paths(data_root: Path, dataset: str, encoder_name: str, split: str) -> tuple[Path, Path]:
    directory = cache_dir(data_root, dataset, encoder_name)
    return directory / f"{split}.npz", directory / f"{split}_meta.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_features(
    dataset: str,
    split: str,
    encoder,  # noqa: ANN001 - duck-typed: name, weights_tag, dim, encode_images
    *,
    data_root: Path,
    batch_size: int = 256,
    device: str = "auto",
    allow_heavy_on_cpu: bool = False,
) -> Path:
    npz_path, meta_path = _paths(data_root, dataset, encoder.name, split)
    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        keys = ("dataset", "split", "encoder", "weights_tag")
        if all(meta.get(k) == v for k, v in zip(
            keys, (dataset, split, encoder.name, encoder.weights_tag), strict=True
        )):
            return npz_path

    resolved = gatekeeper.check("feature_extraction", device, allow_heavy_on_cpu)
    data = load_split(dataset, split, Path(data_root) / "raw")
    chunks = []
    for start in range(0, len(data.labels), batch_size):
        batch = [data.images[i] for i in range(start, min(start + batch_size, len(data.labels)))]
        chunks.append(encoder.encode_images(batch, resolved))
    features = torch.cat(chunks) if chunks else torch.empty(0, encoder.dim)

    n, d = features.shape
    if n != len(data.labels):
        raise ValueError(f"encoder returned {n} rows for {len(data.labels)} items")
    if d != encoder.dim:
        raise ValueError(f"encoder emitted dim {d}, declared dim {encoder.dim}")
    if not torch.isfinite(features).all():
        raise ValueError("non-finite values in extracted features")
    if data.labels.min() < 0 or data.labels.max() >= len(data.class_names):
        raise ValueError("labels outside [0, n_classes)")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, features=features.numpy().astype(np.float32), labels=data.labels)
    meta = {
        "dataset": dataset,
        "split": split,
        "encoder": encoder.name,
        "weights_tag": encoder.weights_tag,
        "N": n,
        "D": d,
        "class_names": list(data.class_names),
        "normalized": False,
        "sha256": _sha256(npz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path


def read_features(
    dataset: str,
    split: str,
    encoder_name: str,
    *,
    data_root: Path,
    l2_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    npz_path, meta_path = _paths(data_root, dataset, encoder_name, split)
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"no cache for ({dataset}, {split}, {encoder_name}) under {npz_path.parent}; "
            "run build_features first"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual = _sha256(npz_path)
    if actual != meta["sha256"]:
        raise CacheChecksumError(
            f"checksum mismatch for {npz_path}: meta says {meta['sha256'][:12]}, "
            f"file is {actual[:12]}; the cache is corrupt or stale, rebuild it "
            "with build_features"
        )
    payload = np.load(npz_path)
    features = torch.from_numpy(payload["features"]).float()
    labels = torch.from_numpy(payload["labels"]).long()
    if l2_normalize:
        features = F.normalize(features, dim=1)
    return features, labels, meta
