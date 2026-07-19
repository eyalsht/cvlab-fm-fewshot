"""Feature cache tests per PRD_feature_cache section 5, all against a stub encoder."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from fm_fewshot.services.features.cache import (
    CacheChecksumError,
    build_features,
    read_features,
)
from fm_fewshot.services.features.encoders import StubEncoder
from fm_fewshot.shared.gatekeeper import HeavyJobOnCpuError

N_CLASSES = 3
PER_CLASS = 10
N_ITEMS = N_CLASSES * PER_CLASS


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """30 grayscale images whose pixel value is the item's global index."""
    index = 0
    for class_name in ("class_a", "class_b", "class_c"):
        class_dir = tmp_path / "raw" / "mini_imagenet" / "test" / class_name
        class_dir.mkdir(parents=True)
        for j in range(PER_CLASS):
            Image.new("L", (4, 4), color=index).save(class_dir / f"img_{j:02d}.png")
            index += 1
    return tmp_path


def build(data_root: Path, encoder: StubEncoder) -> Path:
    return build_features(
        "mini_imagenet", "test", encoder, data_root=data_root, allow_heavy_on_cpu=True
    )


class TestRoundTrip:
    def test_read_returns_built_values_aligned_with_labels(self, data_root: Path) -> None:
        encoder = StubEncoder()
        build(data_root, encoder)
        features, labels, meta = read_features(
            "mini_imagenet", "test", "stub", data_root=data_root, l2_normalize=False
        )
        assert features.dtype == torch.float32
        assert features.shape == (N_ITEMS, encoder.dim)
        assert labels.dtype == torch.int64
        assert labels.tolist() == [i // PER_CLASS for i in range(N_ITEMS)]
        assert meta["N"] == N_ITEMS
        assert meta["D"] == encoder.dim
        assert meta["normalized"] is False
        assert meta["class_names"] == ["class_a", "class_b", "class_c"]

    def test_row_i_is_dataset_item_i(self, data_root: Path) -> None:
        build(data_root, StubEncoder())
        features, _, _ = read_features(
            "mini_imagenet", "test", "stub", data_root=data_root, l2_normalize=False
        )
        assert features[:, 0].tolist() == [float(i) for i in range(N_ITEMS)]


class TestIdempotence:
    def test_second_build_runs_no_encoder_forward(self, data_root: Path) -> None:
        encoder = StubEncoder()
        npz_path = build(data_root, encoder)
        calls_after_first = encoder.batch_calls
        mtime_after_first = npz_path.stat().st_mtime_ns
        assert build(data_root, encoder) == npz_path
        assert encoder.batch_calls == calls_after_first
        assert npz_path.stat().st_mtime_ns == mtime_after_first


class TestNormalization:
    def test_read_normalized_is_unit_norm_and_matches_functional(self, data_root: Path) -> None:
        build(data_root, StubEncoder())
        raw, _, _ = read_features(
            "mini_imagenet", "test", "stub", data_root=data_root, l2_normalize=False
        )
        normalized, _, meta = read_features(
            "mini_imagenet", "test", "stub", data_root=data_root, l2_normalize=True
        )
        norms = normalized.norm(dim=1)
        assert torch.allclose(norms, torch.ones(N_ITEMS))
        assert torch.equal(normalized, F.normalize(raw, dim=1))
        assert meta["normalized"] is False  # stored raw regardless of read mode


class TestCorruption:
    def test_flipped_byte_fails_read_with_rebuild_message(self, data_root: Path) -> None:
        npz_path = build(data_root, StubEncoder())
        payload = bytearray(npz_path.read_bytes())
        payload[len(payload) // 2] ^= 0xFF
        npz_path.write_bytes(bytes(payload))
        with pytest.raises(CacheChecksumError, match="rebuild"):
            read_features("mini_imagenet", "test", "stub", data_root=data_root, l2_normalize=False)


class TestClipWrapperConfig:
    def test_openai_weights_load_the_quickgelu_config(self) -> None:
        # The openai CLIP weights were trained with QuickGELU. open_clip's
        # plain ViT-B-32 config is quick_gelu=False and open_clip 3.x only
        # warns on the mismatch, silently computing wrong features.
        # Regression for the 2026-07-19 cache build that hit this.
        from fm_fewshot.services.features.encoders import ClipVitB32Encoder

        assert ClipVitB32Encoder.model_name == "ViT-B-32-quickgelu"
        assert ClipVitB32Encoder.weights_tag == "openai"


class TestGuardsAndValidation:
    def test_build_is_gatekept_as_heavy(self, data_root: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(HeavyJobOnCpuError):
            build_features("mini_imagenet", "test", StubEncoder(), data_root=data_root)

    def test_encoder_dim_mismatch_aborts(self, data_root: Path) -> None:
        lying = StubEncoder()
        lying.dim = 16  # encoder still emits 8-dim features
        with pytest.raises(ValueError, match="dim"):
            build(data_root, lying)

    def test_missing_cache_read_says_build_first(self, data_root: Path) -> None:
        with pytest.raises(FileNotFoundError, match="build_features"):
            read_features("mini_imagenet", "test", "stub", data_root=data_root, l2_normalize=False)
