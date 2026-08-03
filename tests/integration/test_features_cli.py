"""End-to-end: the features CLI and sdk.build_features on a stub dataset."""

from pathlib import Path

import pytest

from conftest import STUB_DATASET
from fm_fewshot import sdk
from fm_fewshot.__main__ import main


@pytest.fixture
def data_root(tmp_path: Path, stub_dataset: str) -> Path:
    """A writable root; the stub_dataset fixture supplies the items themselves."""
    return tmp_path


def features_argv(data_root: Path) -> list[str]:
    return [
        "features",
        "--dataset", STUB_DATASET,
        "--split", "test",
        "--encoder", "stub",
        "--data-root", str(data_root),
        "--allow-heavy-on-cpu",
    ]


class TestFeaturesCli:
    def test_builds_a_cache(self, data_root: Path) -> None:
        main(features_argv(data_root))
        npz = data_root / "features" / f"{STUB_DATASET}_stub" / "test.npz"
        assert npz.exists()

    def test_heavy_refusal_exits_with_message_not_traceback(
        self, data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        argv = [a for a in features_argv(data_root) if a != "--allow-heavy-on-cpu"]
        with pytest.raises(SystemExit, match="allow-heavy-on-cpu"):
            main(argv)

    def test_repeat_invocation_is_a_noop(self, data_root: Path) -> None:
        main(features_argv(data_root))
        npz = data_root / "features" / f"{STUB_DATASET}_stub" / "test.npz"
        mtime = npz.stat().st_mtime_ns
        main(features_argv(data_root))
        assert npz.stat().st_mtime_ns == mtime


class TestSdk:
    def test_build_features_resolves_encoder_by_name(self, data_root: Path) -> None:
        path = sdk.build_features(
            STUB_DATASET, "test", encoder="stub",
            data_root=data_root, allow_heavy_on_cpu=True,
        )
        assert path.exists()
        assert path.parent.name == f"{STUB_DATASET}_stub"

    def test_unknown_encoder_name_raises(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="no_such_encoder"):
            sdk.build_features(
                STUB_DATASET, "test", encoder="no_such_encoder",
                data_root=data_root, allow_heavy_on_cpu=True,
            )
