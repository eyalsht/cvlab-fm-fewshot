"""Shared fixtures.

The feature-cache and CLI tests need a dataset that never downloads. They used
to build a fake Mini-ImageNet folder tree, which worked only because that
loader read directories. DTD and FGVC-Aircraft come from torchvision
constructors instead, so the seam is now DATASET_SPECS and the stub goes there.
"""

import numpy as np
import pytest
from PIL import Image

from fm_fewshot.services.data import datasets as ds

STUB_DATASET = "stub_ds"
STUB_CLASSES = ("class_a", "class_b", "class_c")
STUB_PER_CLASS = 10
STUB_N_ITEMS = len(STUB_CLASSES) * STUB_PER_CLASS


class StubImageDataset:
    """Class-blocked items whose pixel value is the item's global index.

    Encoding the index into the pixels is what lets the cache's row-alignment
    test assert that cache row i really is dataset item i.
    """

    def __init__(self, root, split, **kwargs):
        self.classes = list(STUB_CLASSES)
        self._labels = [i // STUB_PER_CLASS for i in range(STUB_N_ITEMS)]
        self._images = [Image.new("L", (4, 4), color=i) for i in range(STUB_N_ITEMS)]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, i: int):
        return self._images[i], self._labels[i]


@pytest.fixture
def stub_dataset(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register the stub under DATASET_SPECS and return its name."""
    monkeypatch.setitem(
        ds.DATASET_SPECS,
        STUB_DATASET,
        ds.DatasetSpec(factory=StubImageDataset, kwargs={}, n_classes=len(STUB_CLASSES)),
    )
    return STUB_DATASET


@pytest.fixture
def stub_labels() -> np.ndarray:
    return np.asarray([i // STUB_PER_CLASS for i in range(STUB_N_ITEMS)], dtype=np.int64)
