"""Frozen encoder wrappers (ADR-013): ResNet-18, DINOv2 ViT-S/14, and a stub.

Encoders are pure functions of the image: frozen weights, eval mode, no
gradient. Each carries its own preprocessing, because the write-up requires
each checkpoint's associated transform and because DINOv2's patch size 14
makes a shared transform wrong, not merely non-compliant.

name, model_name and weights_tag all enter the cache key. That is the
2026-07-19 contamination lesson made structural: a cache built by a differently
configured model of the same name must never be accepted as up to date.

Weights load lazily inside the first encode, so importing the package costs
nothing and the unit suite never touches a download.
"""

from collections.abc import Sequence

import torch
from PIL import Image


class StubEncoder:
    """Deterministic 8-dim encoder for tests and dry runs.

    The feature's first component is the image's top-left pixel value, which
    fixture datasets use to encode the item index; alignment tests depend on
    this. batch_calls counts encode_images invocations for idempotence tests.
    """

    name = "stub"
    model_name = "stub"
    weights_tag = "stub"

    def __init__(self) -> None:
        self.dim = 8
        self.batch_calls = 0

    @property
    def preprocess(self):  # noqa: ANN201 - duck-typed transform
        return lambda image: image

    def encode_images(self, images: Sequence[Image.Image], device: torch.device) -> torch.Tensor:
        self.batch_calls += 1
        rows = []
        for image in images:
            pixel = image.getpixel((0, 0))
            value = float(pixel[0] if isinstance(pixel, tuple) else pixel)
            rows.append(value + torch.cat([torch.zeros(1), torch.ones(7)]))
        return torch.stack(rows).float()


class ResNet18Encoder:
    """torchvision ResNet-18, ImageNet-1K weights, 512-dim pre-classifier features.

    The write-up asks for "the 512-dim representation before the final
    classification layer", so fc is replaced by Identity and the pooled output
    is the feature.
    """

    name = "resnet18"
    model_name = "resnet18"
    weights_tag = "IMAGENET1K_V1"

    def __init__(self) -> None:
        self.dim = 512
        self._model = None
        self._preprocess = None

    @property
    def preprocess(self):  # noqa: ANN201 - duck-typed transform
        if self._preprocess is None:
            from torchvision.models import ResNet18_Weights

            # Resolves without downloading the checkpoint.
            self._preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
        return self._preprocess

    @property
    def model(self):  # noqa: ANN201 - duck-typed module, used by tests
        if self._model is None:
            raise RuntimeError("model is not loaded until the first encode_images call")
        return self._model

    def _load(self, device: torch.device) -> None:
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        _freeze(model)
        self._model = model.to(device)

    def encode_images(self, images: Sequence[Image.Image], device: torch.device) -> torch.Tensor:
        if self._model is None:
            self._load(device)
        batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(device)
        with torch.no_grad():
            features = self._model(batch)
        return features.float().cpu()


class Dinov2ViTS14Encoder:
    """DINOv2 ViT-S/14 via timm, 384-dim final class token.

    timm rather than torch.hub: a pinned wheel in the lockfile is reproducible
    where a runtime git clone is not. The checkpoint's own data config gives
    518x518 inputs (14 * 37), which is why the transform cannot be shared with
    ResNet-18.
    """

    name = "dinov2_vits14"
    model_name = "vit_small_patch14_dinov2.lvd142m"
    weights_tag = "lvd142m"
    patch_size = 14

    def __init__(self) -> None:
        self.dim = 384
        self._model = None
        self._preprocess = None

    @property
    def preprocess(self):  # noqa: ANN201 - duck-typed transform
        if self._preprocess is None:
            import timm

            # pretrained=False: the architecture alone resolves the data config,
            # so this stays offline until an actual encode happens.
            skeleton = timm.create_model(self.model_name, pretrained=False, num_classes=0)
            config = timm.data.resolve_model_data_config(skeleton)
            self._preprocess = timm.data.create_transform(**config, is_training=False)
        return self._preprocess

    @property
    def model(self):  # noqa: ANN201 - duck-typed module, used by tests
        if self._model is None:
            raise RuntimeError("model is not loaded until the first encode_images call")
        return self._model

    def _load(self, device: torch.device) -> None:
        import timm

        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        _freeze(model)
        self._model = model.to(device)

    def encode_images(self, images: Sequence[Image.Image], device: torch.device) -> torch.Tensor:
        if self._model is None:
            self._load(device)
        batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(device)
        with torch.no_grad():
            # forward_features returns [B, 1 + n_patches, D]; token 0 is the
            # class token the write-up asks for. num_classes=0 would otherwise
            # hand back the pooled representation, which is not the same thing.
            tokens = self._model.forward_features(batch)
        return tokens[:, 0].float().cpu()


def _freeze(model: torch.nn.Module) -> None:
    """Eval mode and no gradients, asserted here rather than assumed downstream."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


ENCODERS = {
    "stub": StubEncoder,
    "resnet18": ResNet18Encoder,
    "dinov2_vits14": Dinov2ViTS14Encoder,
}


def load_encoder(name: str):  # noqa: ANN201 - encoders are duck-typed by the cache
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; expected one of {sorted(ENCODERS)}")
    return ENCODERS[name]()
