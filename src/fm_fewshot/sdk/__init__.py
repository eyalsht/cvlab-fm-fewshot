"""Public API surface: build_features, run_experiment, run_sweep, make_figures.

The only import surface for notebooks and the CLI.
"""

from pathlib import Path

from fm_fewshot.services.evaluation.loop import run_experiment
from fm_fewshot.services.features import cache
from fm_fewshot.services.features.encoders import load_encoder

__all__ = ["build_features", "run_experiment"]


def build_features(
    dataset: str,
    split: str,
    *,
    encoder: str = "resnet18",
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
