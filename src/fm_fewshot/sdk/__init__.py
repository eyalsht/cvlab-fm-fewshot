"""Public API surface: run_experiment, evaluate_head, build_features.

The only import surface for notebooks and the CLI. run_experiment and
evaluate_head arrive with the evaluation harness in Phase 5.
"""

from pathlib import Path

from fm_fewshot.services.features import cache
from fm_fewshot.services.features.encoders import load_encoder


def build_features(
    dataset: str,
    split: str,
    *,
    encoder: str = "clip_vit_b32",
    data_root: Path = Path("data"),
    batch_size: int = 256,
    device: str = "auto",
    allow_heavy_on_cpu: bool = False,
) -> Path:
    return cache.build_features(
        dataset,
        split,
        load_encoder(encoder),
        data_root=data_root,
        batch_size=batch_size,
        device=device,
        allow_heavy_on_cpu=allow_heavy_on_cpu,
    )
