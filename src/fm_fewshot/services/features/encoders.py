"""Frozen encoder wrappers: open_clip CLIP ViT-B/32 and the test stub.

Encoders are pure functions of the image: frozen weights, eval mode, no
gradient. The cache records name and weights_tag so a cache built with one
set of weights is never silently read as another.
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
    weights_tag = "stub"

    def __init__(self) -> None:
        self.dim = 8
        self.batch_calls = 0

    def encode_images(self, images: Sequence[Image.Image], device: torch.device) -> torch.Tensor:
        self.batch_calls += 1
        rows = []
        for image in images:
            pixel = image.getpixel((0, 0))
            value = float(pixel[0] if isinstance(pixel, tuple) else pixel)
            rows.append(value + torch.cat([torch.zeros(1), torch.ones(7)]))
        return torch.stack(rows).float()


class ClipVitB32Encoder:
    """open_clip ViT-B-32 image tower, openai weights, loaded lazily.

    Weight download happens once per machine on first use. Untested by the
    unit suite (needs the download); exercised when real caches are built.
    """

    name = "clip_vit_b32"
    weights_tag = "openai"

    def __init__(self) -> None:
        self.dim = 512
        self._model = None
        self._preprocess = None

    def _load(self, device: torch.device) -> None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self._model = model.eval().to(device)
        self._preprocess = preprocess

    def encode_images(self, images: Sequence[Image.Image], device: torch.device) -> torch.Tensor:
        if self._model is None:
            self._load(device)
        batch = torch.stack([self._preprocess(img.convert("RGB")) for img in images]).to(device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
        return features.float().cpu()


ENCODERS = {
    "stub": StubEncoder,
    "clip_vit_b32": ClipVitB32Encoder,
}


def load_encoder(name: str):  # noqa: ANN201 - encoders are duck-typed by the cache
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; expected one of {sorted(ENCODERS)}")
    return ENCODERS[name]()
