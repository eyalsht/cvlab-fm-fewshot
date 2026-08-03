"""Dataset loaders: dtd, fgvc_aircraft.

Both come from torchvision at their official train/val/test splits, with the
two arguments the supervisor's protocol fixes: DTD uses partition 1, and
FGVC-Aircraft uses the variant annotation level. Those live in DATASET_SPECS
rather than at the call sites, so no experiment can quietly ask for a different
partition or a coarser label set.

Item order is the torchvision dataset's native index order. Feature caches
index rows by this order, so it must never change between loads.

The combined "trainval" split FGVCAircraft offers is deliberately unreachable:
the write-up forbids merging train and validation.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets as tv_datasets

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    """A torchvision constructor plus the arguments the protocol pins."""

    factory: type
    kwargs: dict = field(default_factory=dict)
    n_classes: int = 0  # 0 disables the check, used by test stubs


# Seam for tests: unit tests replace this mapping so no download runs.
DATASET_SPECS: dict[str, DatasetSpec] = {
    "dtd": DatasetSpec(factory=tv_datasets.DTD, kwargs={"partition": 1}, n_classes=47),
    "fgvc_aircraft": DatasetSpec(
        factory=tv_datasets.FGVCAircraft,
        kwargs={"annotation_level": "variant"},
        n_classes=100,
    ),
}


class DatasetMissingError(RuntimeError):
    """Raised when a dataset is absent from data/raw and cannot be downloaded."""


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


def load_split(dataset: str, split: str, root: Path) -> DatasetSplit:
    if dataset not in DATASET_SPECS:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {tuple(DATASET_SPECS)}"
        )
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r} for {dataset}; expected one of {SPLITS}")

    spec = DATASET_SPECS[dataset]
    tv = _construct(spec, root, split, dataset)

    # DTD and FGVCAircraft expose labels only through the private _labels; they
    # have no .targets. Reading it avoids decoding every image just to get a
    # label, which is why the private access is worth the coupling.
    labels = np.asarray(tv._labels, dtype=np.int64)
    class_names = tuple(tv.classes)

    if spec.n_classes and len(class_names) != spec.n_classes:
        raise ValueError(
            f"{dataset} {split} reports {len(class_names)} classes, expected "
            f"{spec.n_classes}; the download under {root} looks incomplete"
        )

    return DatasetSplit(
        name=dataset,
        split=split,
        images=_TorchvisionImages(tv),
        labels=labels,
        class_names=class_names,
    )


def _construct(spec: DatasetSpec, root: Path, split: str, dataset: str):
    try:
        return spec.factory(root=str(root), split=split, download=True, **spec.kwargs)
    except TypeError:
        # Test stubs take no download flag.
        return spec.factory(root=str(root), split=split, **spec.kwargs)
    except RuntimeError as exc:  # torchvision raises this when a download fails
        raise DatasetMissingError(
            f"{dataset} split {split!r} is not available under {root} and could "
            f"not be downloaded: {exc}"
        ) from exc
