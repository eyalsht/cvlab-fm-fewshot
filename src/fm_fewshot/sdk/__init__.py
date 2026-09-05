"""Public API surface: build_features, run_experiment, run_sweep, the figures,
run_reverse, run_diagnostics.

The only import surface for notebooks and the CLI.
"""

from pathlib import Path

from fm_fewshot.services.evaluation.loop import run_experiment
from fm_fewshot.services.evaluation.make_figures import make_figures
from fm_fewshot.services.evaluation.make_stage2_figures import make_stage2_figures
from fm_fewshot.services.evaluation.make_stage3_figures import make_stage3_figures
from fm_fewshot.services.evaluation.reverse_run import run_reverse
from fm_fewshot.services.evaluation.rowspace_diagnostics import run_rowspace_diagnostics
from fm_fewshot.services.evaluation.scale_diagnostics import run_diagnostics
from fm_fewshot.services.evaluation.sweep import run_sweep
from fm_fewshot.services.features import cache
from fm_fewshot.services.features.encoders import load_encoder

__all__ = [
    "build_features",
    "make_figures",
    "make_stage2_figures",
    "make_stage3_figures",
    "run_diagnostics",
    "run_experiment",
    "run_reverse",
    "run_rowspace_diagnostics",
    "run_sweep",
]


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
