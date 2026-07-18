"""End-to-end: the features CLI and sdk.build_features on a fixture dataset."""

from pathlib import Path

import pytest
from PIL import Image

from fm_fewshot import sdk
from fm_fewshot.__main__ import main


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    for class_name in ("class_a", "class_b"):
        class_dir = tmp_path / "raw" / "mini_imagenet" / "test" / class_name
        class_dir.mkdir(parents=True)
        for j in range(3):
            Image.new("L", (4, 4), color=j).save(class_dir / f"img_{j}.png")
    return tmp_path


def features_argv(data_root: Path) -> list[str]:
    return [
        "features",
        "--dataset", "mini_imagenet",
        "--split", "test",
        "--encoder", "stub",
        "--data-root", str(data_root),
        "--allow-heavy-on-cpu",
    ]


class TestFeaturesCli:
    def test_builds_a_cache(self, data_root: Path) -> None:
        main(features_argv(data_root))
        npz = data_root / "features" / "mini_imagenet_stub" / "test.npz"
        assert npz.exists()

    def test_repeat_invocation_is_a_noop(self, data_root: Path) -> None:
        main(features_argv(data_root))
        npz = data_root / "features" / "mini_imagenet_stub" / "test.npz"
        mtime = npz.stat().st_mtime_ns
        main(features_argv(data_root))
        assert npz.stat().st_mtime_ns == mtime


class TestSdk:
    def test_build_features_resolves_encoder_by_name(self, data_root: Path) -> None:
        path = sdk.build_features(
            "mini_imagenet", "test", encoder="stub",
            data_root=data_root, allow_heavy_on_cpu=True,
        )
        assert path.exists()
        assert path.parent.name == "mini_imagenet_stub"

    def test_unknown_encoder_name_raises(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="no_such_encoder"):
            sdk.build_features(
                "mini_imagenet", "test", encoder="no_such_encoder",
                data_root=data_root, allow_heavy_on_cpu=True,
            )
