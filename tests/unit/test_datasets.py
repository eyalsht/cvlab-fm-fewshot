"""Dataset loader tests: DTD and FGVC-Aircraft at their official splits.

No test downloads anything. The torchvision constructors sit behind a registry
the tests replace with a stub, which also lets them assert that the two
protocol-critical arguments (DTD partition 1, Aircraft variant level) actually
reach torchvision.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fm_fewshot.services.data import datasets as ds

SPLITS = ("train", "val", "test")


class StubTorchvisionDataset:
    """Mimics DTD and FGVCAircraft: private _labels, public classes, no targets."""

    calls: list[dict] = []

    def __init__(self, root, split, **kwargs):
        type(self).calls.append({"root": root, "split": split, **kwargs})
        self._split = split
        # Three classes, two images each, class-blocked.
        self.classes = ["alpha", "beta", "gamma"]
        self._labels = [0, 0, 1, 1, 2, 2]
        self._images = [
            Image.new("RGB", (4, 4), color=(i, i, i)) for i in range(len(self._labels))
        ]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, i: int):
        return self._images[i], self._labels[i]


@pytest.fixture(autouse=True)
def stub_registry(monkeypatch: pytest.MonkeyPatch):
    StubTorchvisionDataset.calls = []
    specs = {
        "dtd": ds.DatasetSpec(
            factory=StubTorchvisionDataset,
            kwargs={"partition": 1},
            n_classes=3,
        ),
        "fgvc_aircraft": ds.DatasetSpec(
            factory=StubTorchvisionDataset,
            kwargs={"annotation_level": "variant"},
            n_classes=3,
        ),
    }
    monkeypatch.setattr(ds, "DATASET_SPECS", specs)
    return specs


class TestProtocolArguments:
    def test_dtd_requests_official_partition_one(self, tmp_path: Path) -> None:
        ds.load_split("dtd", "train", tmp_path)
        assert StubTorchvisionDataset.calls[-1]["partition"] == 1

    def test_aircraft_requests_the_variant_annotation_level(self, tmp_path: Path) -> None:
        ds.load_split("fgvc_aircraft", "train", tmp_path)
        assert StubTorchvisionDataset.calls[-1]["annotation_level"] == "variant"

    @pytest.mark.parametrize("split", SPLITS)
    def test_every_official_split_is_reachable(self, split: str, tmp_path: Path) -> None:
        ds.load_split("dtd", split, tmp_path)
        assert StubTorchvisionDataset.calls[-1]["split"] == split

    def test_splits_are_never_merged(self, tmp_path: Path) -> None:
        """No loader call may ask torchvision for a combined split."""
        for split in SPLITS:
            ds.load_split("fgvc_aircraft", split, tmp_path)
        requested = {call["split"] for call in StubTorchvisionDataset.calls}
        assert requested == set(SPLITS)
        assert "trainval" not in requested


class TestSplitContents:
    def test_labels_come_from_the_private_attribute(self, tmp_path: Path) -> None:
        """DTD and FGVCAircraft expose _labels, not the .targets of MNIST/CIFAR."""
        split = ds.load_split("dtd", "train", tmp_path)
        assert split.labels.dtype == np.int64
        assert split.labels.tolist() == [0, 0, 1, 1, 2, 2]

    def test_class_names_are_indexed_by_global_class_id(self, tmp_path: Path) -> None:
        split = ds.load_split("dtd", "train", tmp_path)
        assert split.class_names == ("alpha", "beta", "gamma")

    def test_item_order_is_deterministic(self, tmp_path: Path) -> None:
        first = ds.load_split("dtd", "test", tmp_path)
        second = ds.load_split("dtd", "test", tmp_path)
        assert first.labels.tolist() == second.labels.tolist()
        assert [np.asarray(im).tolist() for im in first.images] == [
            np.asarray(im).tolist() for im in second.images
        ]

    def test_images_align_with_labels(self, tmp_path: Path) -> None:
        split = ds.load_split("dtd", "train", tmp_path)
        assert len(split.images) == split.labels.shape[0]

    def test_carries_its_identity(self, tmp_path: Path) -> None:
        split = ds.load_split("fgvc_aircraft", "val", tmp_path)
        assert split.name == "fgvc_aircraft"
        assert split.split == "val"


class TestValidation:
    def test_unknown_dataset_raises_listing_the_known_ones(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown dataset"):
            ds.load_split("mnist", "train", tmp_path)

    def test_unknown_split_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown split"):
            ds.load_split("dtd", "trainval", tmp_path)

    def test_declared_class_count_is_enforced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A split whose class list disagrees with the spec is a corrupt download."""
        monkeypatch.setitem(
            ds.DATASET_SPECS,
            "dtd",
            ds.DatasetSpec(factory=StubTorchvisionDataset, kwargs={"partition": 1}, n_classes=47),
        )
        with pytest.raises(ValueError, match="47"):
            ds.load_split("dtd", "train", tmp_path)


class TestRealSplitSizes:
    """Guards the numbers PRD and the alignment note rely on. Downloads; slow."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("dataset", "n_classes", "sizes"),
        [
            ("dtd", 47, {"train": 1880, "val": 1880, "test": 1880}),
            ("fgvc_aircraft", 100, {"train": 3334, "val": 3333, "test": 3333}),
        ],
    )
    def test_official_split_sizes(
        self,
        dataset: str,
        n_classes: int,
        sizes: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ds, "DATASET_SPECS", ds.DATASET_SPECS)
        for split, expected in sizes.items():
            loaded = ds.load_split(dataset, split, Path("data/raw"))
            assert loaded.labels.shape[0] == expected
            assert len(loaded.class_names) == n_classes
