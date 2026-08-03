"""Encoder tests per PRD_encoders section 5.

Tests that need pretrained weights are marked slow: they download once per
machine. Everything else runs offline, including the preprocessing checks,
because both transforms resolve without fetching a checkpoint.
"""

import pytest
import torch
from PIL import Image

from fm_fewshot.services.features.encoders import (
    ENCODERS,
    Dinov2ViTS14Encoder,
    ResNet18Encoder,
    StubEncoder,
    load_encoder,
)

CPU = torch.device("cpu")


def rgb_batch(n: int = 3, size: int = 64) -> list[Image.Image]:
    return [Image.new("RGB", (size, size), color=(i * 10, 20, 30)) for i in range(n)]


class TestRegistry:
    def test_registered_keys_are_exactly_the_selected_encoders(self) -> None:
        """CLIP left the project with Option B; only these two plus the stub remain."""
        assert sorted(ENCODERS) == ["dinov2_vits14", "resnet18", "stub"]

    def test_unknown_encoder_raises_listing_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown encoder"):
            load_encoder("clip_rn50")

    def test_load_encoder_returns_an_instance(self) -> None:
        assert isinstance(load_encoder("resnet18"), ResNet18Encoder)


class TestDeclaredIdentity:
    @pytest.mark.parametrize(
        ("cls", "name", "dim"),
        [(ResNet18Encoder, "resnet18", 512), (Dinov2ViTS14Encoder, "dinov2_vits14", 384)],
    )
    def test_declares_its_name_and_dim(self, cls: type, name: str, dim: int) -> None:
        assert cls.name == name
        assert cls().dim == dim

    def test_every_encoder_declares_a_cache_key(self) -> None:
        """name, model_name and weights_tag all enter the cache key (ADR-013)."""
        for factory in ENCODERS.values():
            encoder = factory()
            assert encoder.name and encoder.model_name and encoder.weights_tag


class TestPreprocessing:
    def test_each_encoder_owns_a_distinct_transform(self) -> None:
        """Reusing one transform across checkpoints would be wrong and non-compliant."""
        resnet = ResNet18Encoder().preprocess
        dinov2 = Dinov2ViTS14Encoder().preprocess
        assert resnet is not dinov2
        assert type(resnet) is not type(dinov2) or repr(resnet) != repr(dinov2)

    def test_dinov2_input_dims_are_multiples_of_its_patch_size(self) -> None:
        """ViT-S/14 cannot accept a size that is not a multiple of 14."""
        out = Dinov2ViTS14Encoder().preprocess(rgb_batch(1)[0])
        assert out.shape[-1] % 14 == 0
        assert out.shape[-2] % 14 == 0

    def test_resnet18_preprocess_produces_a_chw_float_tensor(self) -> None:
        out = ResNet18Encoder().preprocess(rgb_batch(1)[0])
        assert out.ndim == 3
        assert out.shape[0] == 3
        assert out.dtype == torch.float32


class TestStub:
    def test_encodes_the_item_index_into_the_first_component(self) -> None:
        """The cache's row-alignment test depends on this."""
        images = [Image.new("L", (4, 4), color=i) for i in range(5)]
        features = StubEncoder().encode_images(images, CPU)
        assert features[:, 0].tolist() == [float(i) for i in range(5)]

    def test_counts_batches_for_the_idempotence_test(self) -> None:
        encoder = StubEncoder()
        assert encoder.batch_calls == 0
        encoder.encode_images(rgb_batch(2), CPU)
        assert encoder.batch_calls == 1


@pytest.mark.slow
class TestPretrainedWeights:
    """Downloads checkpoints once per machine."""

    @pytest.mark.parametrize(
        "cls", [ResNet18Encoder, Dinov2ViTS14Encoder], ids=["resnet18", "dinov2_vits14"]
    )
    def test_output_width_matches_the_declared_dim(self, cls: type) -> None:
        encoder = cls()
        features = encoder.encode_images(rgb_batch(2), CPU)
        assert features.shape == (2, encoder.dim)
        assert features.dtype == torch.float32
        assert torch.isfinite(features).all()

    @pytest.mark.parametrize(
        "cls", [ResNet18Encoder, Dinov2ViTS14Encoder], ids=["resnet18", "dinov2_vits14"]
    )
    def test_is_frozen_and_in_eval_mode(self, cls: type) -> None:
        """Asserted at load, not assumed."""
        encoder = cls()
        encoder.encode_images(rgb_batch(1), CPU)
        model = encoder.model
        assert not model.training
        assert not any(p.requires_grad for p in model.parameters())

    @pytest.mark.parametrize(
        "cls", [ResNet18Encoder, Dinov2ViTS14Encoder], ids=["resnet18", "dinov2_vits14"]
    )
    def test_encoding_is_deterministic(self, cls: type) -> None:
        encoder = cls()
        images = rgb_batch(2)
        assert torch.equal(
            encoder.encode_images(images, CPU), encoder.encode_images(images, CPU)
        )

    def test_resnet18_drops_the_classification_layer(self) -> None:
        """The write-up asks for the 512-dim representation before the final layer."""
        encoder = ResNet18Encoder()
        encoder.encode_images(rgb_batch(1), CPU)
        assert isinstance(encoder.model.fc, torch.nn.Identity)
