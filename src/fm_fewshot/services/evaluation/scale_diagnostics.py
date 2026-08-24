"""The `diagnose` command: what raw feature scale did to the transported cloud.

ADR-018 chose the literal reading of the write-up: features enter the flow raw
while the prototypes are unit norm by his own formula, so the endpoints of the
coupling sit at different scales and

    u_i = p_{y_i} - z_i

is dominated by -z_i, which every class shares and which carries no class
information. That was recorded as a conditioning risk to instrument rather than
a contradiction to correct, and this module is the instrument. Four numbers per
run:

- the distribution of ||z||, against ||p|| = 1;
- ||z_T - p_{y}|| on the training subset and on the test split;
- cos(z_T, p_y), the quantity the decision rule actually reads;
- the pairwise cosine among transported test points, with the same statistic on
  the raw test split beside it, because a high transported cosine only means
  collapse when the raw one was lower.

Everything is median and interquartile range. Contraction is heavy-tailed, so a
mean would be set by whichever points the field squeezed hardest, and the
question here is where the bulk sits.

Two rules this file lives under, both borrowed from `reverse_run`. ADR-025: the
field is refitted deterministically from the stored `config.yaml`, never loaded
and never retrained on other terms, so the numbers describe the field that
produced the run's accuracy. ADR-027 in spirit: these are diagnostics. They may
read the test split, and in exchange nothing here may inform a head
configuration or reach `TABLE.md`, which is why no module on the report path
imports this one and why `diagnostics.json` is not a file the report knows
about.
"""

import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fm_fewshot.services.evaluation.loop import git_commit
from fm_fewshot.services.evaluation.reverse_run import MissingRunError, find_run, refit_head
from fm_fewshot.services.features.cache import read_features
from fm_fewshot.services.flow.reverse import median_iqr
from fm_fewshot.services.heads.fm_base import FmHead
from fm_fewshot.shared.config import load_config
from fm_fewshot.shared.contracts import ExperimentConfig

# The pairwise cosine is over unordered pairs, so its cost is quadratic in the
# split. The largest test split here is 3333 rows, comfortably inside this cap;
# it exists so a bigger split degrades to a seeded subsample instead of an
# allocation nobody asked for.
DEFAULT_MAX_POINTS = 4096


def feature_norms(x: Tensor) -> Tensor:
    """||z|| per row. Read against ||p|| = 1, which is the whole of ADR-018."""
    return x.detach().norm(dim=-1)


def target_distances(transported: Tensor, targets: Tensor) -> Tensor:
    """||z_T - p_y|| per row: how far the flow stopped short of its target."""
    if transported.shape != targets.shape:
        raise ValueError(
            f"transported {tuple(transported.shape)} and targets {tuple(targets.shape)} "
            "must have the same shape; each row is one point and its own target"
        )
    return (transported - targets).detach().norm(dim=-1)


def target_cosines(transported: Tensor, targets: Tensor) -> Tensor:
    """cos(z_T, p_y) per row, the quantity the cosine decision rule reads.

    Distance and cosine can disagree: a point can stay far from its prototype in
    norm and still point straight at it, and only the direction is scored.
    """
    if transported.shape != targets.shape:
        raise ValueError(
            f"transported {tuple(transported.shape)} and targets {tuple(targets.shape)} "
            "must have the same shape; each row is one point and its own target"
        )
    return F.cosine_similarity(transported.detach(), targets.detach(), dim=-1)


def pairwise_cosines(
    x: Tensor, *, max_points: int = DEFAULT_MAX_POINTS, seed: int = 0
) -> Tensor:
    """Cosine over every unordered pair of distinct rows, flattened.

    This is the collapse detector. If the flow drove every test point into one
    cone, these values pile up near 1 whatever the accuracy says, and the
    classifier is reading a margin that barely exists.
    """
    points = x.detach()
    if points.shape[0] < 2:
        raise ValueError(
            f"pairwise cosine needs at least two points, got {points.shape[0]}"
        )
    if points.shape[0] > max_points:
        generator = torch.Generator().manual_seed(seed)
        keep = torch.randperm(points.shape[0], generator=generator)[:max_points].sort().values
        points = points[keep]
    gram = F.normalize(points, dim=-1) @ F.normalize(points, dim=-1).T
    upper = torch.triu_indices(gram.shape[0], gram.shape[0], offset=1)
    return gram[upper[0], upper[1]]


def run_diagnostics(
    cfg: ExperimentConfig,
    *,
    data_root: Path = Path("data"),
    results_dir: Path = Path("results"),
    run_id: str | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> Path:
    """Write `results/<run_id>/diagnostics.json` for one Stage 2 run."""
    results_dir = Path(results_dir)
    run_id = run_id or find_run(results_dir, cfg)
    run_dir = results_dir / run_id
    if not run_dir.exists():
        raise MissingRunError(f"no run directory {run_dir}")

    head, subset_x, subset_y, n_classes = refit_head(cfg, data_root)
    sample_steps = int(FmHead.shared_params(cfg, n_classes)["sample_steps"])
    # Only now, and for diagnostics only, as in the forward loop.
    test_x, test_y, _ = read_features(
        cfg.dataset,
        cfg.eval_split,
        cfg.encoder,
        data_root=data_root,
        l2_normalize=cfg.l2_normalize,
    )

    train_transported = head.transport(subset_x)
    test_transported = head.transport(test_x)
    train_targets = head.prototypes[subset_y]
    test_targets = head.prototypes[test_y]

    payload = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "config": dataclasses.asdict(cfg),
        "forward_steps": sample_steps,
        "max_points": max_points,
        "prototype_norm": median_iqr(feature_norms(head.prototypes)),
        "feature_norm": {
            "train": median_iqr(feature_norms(subset_x)),
            "test": median_iqr(feature_norms(test_x)),
        },
        "target_distance": {
            "train": median_iqr(target_distances(train_transported, train_targets)),
            "test": median_iqr(target_distances(test_transported, test_targets)),
        },
        "target_cosine": {
            "train": median_iqr(target_cosines(train_transported, train_targets)),
            "test": median_iqr(target_cosines(test_transported, test_targets)),
        },
        "pairwise_cosine": {
            "test_raw": median_iqr(
                pairwise_cosines(test_x, max_points=max_points, seed=cfg.seed)
            ),
            "test_transported": median_iqr(
                pairwise_cosines(test_transported, max_points=max_points, seed=cfg.seed)
            ),
        },
    }

    out = run_dir / "diagnostics.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def diagnose_from_config_file(
    config_path: Path,
    *,
    data_root: Path = Path("data"),
    results_dir: Path = Path("results"),
    run_id: str | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> Path:
    return run_diagnostics(
        load_config(config_path),
        data_root=data_root,
        results_dir=results_dir,
        run_id=run_id,
        max_points=max_points,
    )
