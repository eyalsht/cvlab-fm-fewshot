"""Dataset loader tests on synthetic fixtures; no network, no real downloads."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fm_fewshot.services.data import datasets as datasets_module
from fm_fewshot.services.data.datasets import DatasetMissingError, load_split


class StubTorchvisionDataset:
    """Stands in for torchvision MNIST/CIFAR10: classes, targets, indexable images."""

    classes = ["0 - zero", "1 - one", "2 - two"]

    def __init__(self, root: str, train: bool, download: bool) -> None:
        n = 6 if train else 3
        self.targets = [i % 3 for i in range(n)]
        self._images = [Image.new("L", (8, 8), color=i * 10) for i in range(n)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        return self._images[i], self.targets[i]


@pytest.fixture
def stub_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(datasets_module.TORCHVISION_DATASETS, "mnist", StubTorchvisionDataset)


@pytest.fixture
def mini_imagenet_root(tmp_path: Path) -> Path:
    for class_name in ("n01532829", "n01558993", "n01704323"):
        class_dir = tmp_path / "mini_imagenet" / "test" / class_name
        class_dir.mkdir(parents=True)
        for j in range(4):
            Image.new("RGB", (8, 8), color=j).save(class_dir / f"img_{j}.jpg")
    return tmp_path


class TestTorchvisionBacked:
    def test_split_sizes_follow_train_flag(self, stub_torchvision: None, tmp_path: Path) -> None:
        train = load_split("mnist", "train", tmp_path)
        test = load_split("mnist", "test", tmp_path)
        assert len(train.labels) == len(train.images) == 6
        assert len(test.labels) == len(test.images) == 3

    def test_labels_are_int64_in_dataset_order(self, stub_torchvision: None,
                                               tmp_path: Path) -> None:
        split = load_split("mnist", "train", tmp_path)
        assert split.labels.dtype == np.int64
        assert split.labels.tolist() == [0, 1, 2, 0, 1, 2]

    def test_class_names_come_from_the_dataset(self, stub_torchvision: None,
                                               tmp_path: Path) -> None:
        split = load_split("mnist", "test", tmp_path)
        assert split.class_names == ("0 - zero", "1 - one", "2 - two")

    def test_item_order_is_deterministic(self, stub_torchvision: None, tmp_path: Path) -> None:
        first = load_split("mnist", "train", tmp_path)
        second = load_split("mnist", "train", tmp_path)
        assert np.array_equal(first.labels, second.labels)
        assert first.class_names == second.class_names

    def test_images_are_indexable_pil(self, stub_torchvision: None, tmp_path: Path) -> None:
        split = load_split("mnist", "test", tmp_path)
        assert isinstance(split.images[0], Image.Image)


class TestMiniImagenet:
    def test_loads_sorted_classes_and_files(self, mini_imagenet_root: Path) -> None:
        split = load_split("mini_imagenet", "test", mini_imagenet_root)
        assert split.class_names == ("n01532829", "n01558993", "n01704323")
        assert split.labels.dtype == np.int64
        assert split.labels.tolist() == [0] * 4 + [1] * 4 + [2] * 4
        assert isinstance(split.images[0], Image.Image)

    def test_item_order_is_deterministic(self, mini_imagenet_root: Path) -> None:
        first = load_split("mini_imagenet", "test", mini_imagenet_root)
        second = load_split("mini_imagenet", "test", mini_imagenet_root)
        assert np.array_equal(first.labels, second.labels)
        assert first.class_names == second.class_names

    def test_missing_data_error_names_path_and_readme(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetMissingError) as exc_info:
            load_split("mini_imagenet", "test", tmp_path)
        message = str(exc_info.value)
        assert "mini_imagenet" in message
        assert "README" in message

    def test_val_is_a_valid_split(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetMissingError):
            load_split("mini_imagenet", "val", tmp_path)


class TestValidation:
    def test_unknown_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no_such_dataset"):
            load_split("no_such_dataset", "test", tmp_path)

    def test_unknown_split_raises(self, stub_torchvision: None, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="val"):
            load_split("mnist", "val", tmp_path)
