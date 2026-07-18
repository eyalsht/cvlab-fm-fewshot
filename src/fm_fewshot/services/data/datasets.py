"""Dataset loaders: mnist, cifar10, mini_imagenet.

Every loader returns a DatasetSplit whose item order is deterministic:
torchvision datasets are exposed in their native index order, mini_imagenet
in sorted class-directory then sorted filename order. Feature caches index
rows by this order, so it must never change between loads.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets as tv_datasets

# Seam for tests: unit tests replace entries with a stub so no download runs.
TORCHVISION_DATASETS: dict[str, type] = {
    "mnist": tv_datasets.MNIST,
    "cifar10": tv_datasets.CIFAR10,
}

MINI_IMAGENET_SPLITS = ("train", "val", "test")

DATASET_NAMES = ("mnist", "cifar10", "mini_imagenet")


class DatasetMissingError(RuntimeError):
    """Raised when a manually acquired dataset is absent from data/raw."""


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    split: str
    images: Sequence[Image.Image]
    labels: np.ndarray  # int64 [N], aligned with images
    class_names: tuple[str, ...]  # index == global class id


class _TorchvisionImages(Sequence[Image.Image]):
    """Index-only view over a torchvision dataset that drops the label."""

    def __init__(self, dataset) -> None:  # noqa: ANN001 - duck-typed torchvision dataset
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i: int) -> Image.Image:
        return self._dataset[i][0]


class _ImageFiles(Sequence[Image.Image]):
    """Lazy image loading so listing a split never reads pixel data."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, i: int) -> Image.Image:
        return Image.open(self._paths[i]).convert("RGB")


def load_split(dataset: str, split: str, root: Path) -> DatasetSplit:
    root = Path(root)
    if dataset in TORCHVISION_DATASETS:
        return _load_torchvision(dataset, split, root)
    if dataset == "mini_imagenet":
        return _load_mini_imagenet(split, root)
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASET_NAMES}")


def _load_torchvision(dataset: str, split: str, root: Path) -> DatasetSplit:
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split!r} for {dataset}; expected 'train' or 'test'")
    tv = TORCHVISION_DATASETS[dataset](root=str(root), train=split == "train", download=True)
    labels = np.asarray(tv.targets, dtype=np.int64)
    return DatasetSplit(
        name=dataset,
        split=split,
        images=_TorchvisionImages(tv),
        labels=labels,
        class_names=tuple(tv.classes),
    )


def _load_mini_imagenet(split: str, root: Path) -> DatasetSplit:
    if split not in MINI_IMAGENET_SPLITS:
        raise ValueError(
            f"unknown split {split!r} for mini_imagenet; expected one of {MINI_IMAGENET_SPLITS}"
        )
    split_dir = root / "mini_imagenet" / split
    class_dirs = sorted(d for d in split_dir.glob("*") if d.is_dir()) if split_dir.is_dir() else []
    if not class_dirs:
        raise DatasetMissingError(
            f"mini_imagenet split {split!r} not found under {split_dir}. "
            "Mini-ImageNet requires a manual download; see the README section "
            "'Mini-ImageNet acquisition' for instructions and checksums."
        )
    paths: list[Path] = []
    labels: list[int] = []
    for class_id, class_dir in enumerate(class_dirs):
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        paths.extend(files)
        labels.extend([class_id] * len(files))
    return DatasetSplit(
        name="mini_imagenet",
        split=split,
        images=_ImageFiles(paths),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=tuple(d.name for d in class_dirs),
    )
