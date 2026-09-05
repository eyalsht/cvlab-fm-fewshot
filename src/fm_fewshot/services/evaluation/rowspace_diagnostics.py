"""The `rowspace` command: split `zhat - z` into row(W) and null(W) (ADR-036).

`W` is `[C, D]` with `C <= D` in every Stage 3 cell, so the frozen probe's
logits `s = W zhat + b` depend on `zhat` only through its projection onto
`row(W)` (`PRD_stage3_prelinear.md` section 1). The displacement a Stage 3
field applies therefore splits into a part that can move the decision and a
part the loss cannot see, and this module measures that split on the test
split of a stored run.

    rowspace.json
    {
      "run_id": str,
      "head": str,                  # registry key
      "sample_steps": int,          # the run's T
      "dim": int,                   # D
      "rank_w": int,                # rank of W, at most C
      "n_points": int,              # points recorded, after subsampling
      "max_points": int,
      "summary": {
        "row_fraction_mean": float,     # mean over points of row_norm / total_norm
        "row_fraction_median": float,
        "null_fraction_mean": float,
        "null_fraction_median": float,
        "total_norm_mean": float,
        "total_norm_median": float
      },
      "points": {
        "total_norm":    [float],   # || zhat - z ||
        "row_norm":      [float],   # || P (zhat - z) ||        P = row_space_projector(W)
        "null_norm":     [float],   # || (I - P) (zhat - z) ||
        "margin_before": [float],   # s_y - max_{c != y} s_c evaluated at z
        "margin_after":  [float]    # the same margin evaluated at zhat
      }
    }

Points are test-split examples, subsampled deterministically from `cfg.seed`
when the split exceeds `max_points`, the same convention `scale_diagnostics`
uses for its pairwise cosines.

Two rules this file lives under, both borrowed from `reverse_run` and
`scale_diagnostics`. ADR-025: the head is refitted deterministically from the
stored `config.yaml`, never loaded and never retrained on other terms, so the
numbers describe the field that produced the run's accuracy. ADR-027 in
spirit: this is a diagnostic. It reads the test labels for the margin, and in
exchange nothing here may inform a head configuration or reach `TABLE.md`,
which is why no module on the report path imports this one.
"""

import json
from pathlib import Path

import torch
from torch import Tensor

from fm_fewshot.services.data.subsets import balanced_subset
from fm_fewshot.services.evaluation.reverse_run import MissingRunError, NotAnFmRunError, find_run
from fm_fewshot.services.features.cache import read_features
from fm_fewshot.services.flow.training.rolled_out_ce import row_space_projector
from fm_fewshot.services.heads import make_head
from fm_fewshot.services.heads.fm_prelinear import FmPreLinearHead
from fm_fewshot.shared.config import load_config
from fm_fewshot.shared.contracts import ExperimentConfig

__all__ = [
    "MissingRunError",
    "NotAnFmRunError",
    "margins",
    "norm_fractions",
    "rowspace_from_config_file",
    "run_rowspace_diagnostics",
    "split_displacement",
]

# Matches scale_diagnostics's DEFAULT_MAX_POINTS: the largest test split here
# is 3333 rows, comfortably inside this cap, so it exists to bound a bigger
# split rather than to trim this one in practice.
DEFAULT_MAX_POINTS = 4096


def split_displacement(displacement: Tensor, projector: Tensor) -> tuple[Tensor, Tensor]:
    """`(row_component, null_component)`, each `[N, D]`, summing to `displacement`.

    `projector` is `row_space_projector(weight)`, symmetric and idempotent, so
    `displacement @ projector` is `P v` for every row `v` written as a row
    vector: `(P v)^T = v^T P^T = v^T P` because `P` is symmetric.
    """
    row_component = displacement @ projector
    null_component = displacement - row_component
    return row_component, null_component


def norm_fractions(
    row_norm: Tensor, null_norm: Tensor, total_norm: Tensor
) -> tuple[Tensor, Tensor]:
    """`(row_norm, null_norm) / total_norm`, with a zero-displacement point scored as zero mass.

    A point whose `total_norm` is zero moved nowhere (the field left it at
    `z`, as it does everywhere before the first optimizer step or whenever
    `n_train_steps=0`). Both components are then zero along with the total, so
    the ratio is a genuine `0/0` rather than a numerical accident of a small
    denominator; `nan` there would silently drag every mean and median in
    `summary` to `nan` the first time a diagnostic ran on an untrained field.
    Scoring it as zero row and zero null mass matches what happened: no
    capacity was spent anywhere at that point.
    """
    safe = total_norm > 0
    denominator = torch.where(safe, total_norm, torch.ones_like(total_norm))
    row_fraction = torch.where(safe, row_norm / denominator, torch.zeros_like(total_norm))
    null_fraction = torch.where(safe, null_norm / denominator, torch.zeros_like(total_norm))
    return row_fraction, null_fraction


def margins(logits: Tensor, y: Tensor) -> Tensor:
    """`s_y - max_{c != y} s_c` per row, the decision margin at whatever features gave `logits`."""
    true_score = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    mask = torch.nn.functional.one_hot(y, logits.shape[1]).bool()
    other_max = logits.masked_fill(mask, float("-inf")).max(dim=1).values
    return (true_score - other_max).detach()


def _subsample_indices(n: int, max_points: int, seed: int) -> Tensor:
    """Every row index when `n <= max_points`, else a seeded, sorted draw of `max_points`.

    Sorted after the draw so row order in `points` still follows the split's
    own order, the same convention `pairwise_cosines` uses.
    """
    if n <= max_points:
        return torch.arange(n)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:max_points].sort().values


def refit_stage3_head(cfg: ExperimentConfig, data_root: Path) -> tuple[FmPreLinearHead, int]:
    """Reproduce the run's fitted Stage 3 head from its config (ADR-025, ADR-031).

    Mirrors `reverse_run.refit_head`'s call order, but the type guard is for a
    head that has a frozen linear probe to project against, since row(W) is
    undefined for a head with no `W`.
    """
    train_x, train_y, train_meta = read_features(
        cfg.dataset, "train", cfg.encoder, data_root=data_root, l2_normalize=cfg.l2_normalize
    )
    n_classes = len(train_meta["class_names"])
    head = make_head(cfg, n_classes)
    if not isinstance(head, FmPreLinearHead):
        raise NotAnFmRunError(
            f"head {cfg.head!r} has no frozen linear probe; row-space diagnostics "
            "are defined for the Stage 3 heads only"
        )
    val_x, val_y, _ = read_features(
        cfg.dataset, "val", cfg.encoder, data_root=data_root, l2_normalize=cfg.l2_normalize
    )
    subset = balanced_subset(train_y, cfg.k, cfg.subset_seed, n_classes, cfg.dataset)
    head.fit(train_x[subset.idx], subset.labels, val_x, val_y)
    return head, n_classes


def run_rowspace_diagnostics(
    cfg: ExperimentConfig,
    *,
    data_root: Path = Path("data"),
    results_dir: Path = Path("results"),
    run_id: str | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> Path:
    """Write `results/<run_id>/rowspace.json` for one Stage 3 run."""
    results_dir = Path(results_dir)
    run_id = run_id or find_run(results_dir, cfg)
    run_dir = results_dir / run_id
    if not run_dir.exists():
        raise MissingRunError(f"no run directory {run_dir}")

    head, n_classes = refit_stage3_head(cfg, data_root)
    sample_steps = int(FmPreLinearHead.shared_params(cfg, n_classes)["sample_steps"])

    # Only now, and for diagnostics only, as in reverse_run and scale_diagnostics.
    test_x, test_y, _ = read_features(
        cfg.dataset, cfg.eval_split, cfg.encoder, data_root=data_root, l2_normalize=cfg.l2_normalize
    )
    idx = _subsample_indices(test_x.shape[0], max_points, seed=cfg.seed)
    sample_x = test_x[idx]
    sample_y = test_y[idx]

    weight = head.probe.weight
    dim = int(weight.shape[1])
    projector = row_space_projector(weight)
    # trace(P) == rank(row(W)) exactly for an orthogonal projector: its
    # eigenvalues are all 0 or 1, one per basis row row_space_projector kept.
    rank_w = int(round(float(projector.trace())))

    transported = head.transport(sample_x)
    displacement = (transported - sample_x).detach()
    row_component, null_component = split_displacement(displacement, projector)

    total_norm = displacement.norm(dim=-1)
    row_norm = row_component.norm(dim=-1)
    null_norm = null_component.norm(dim=-1)
    row_fraction, null_fraction = norm_fractions(row_norm, null_norm, total_norm)

    margin_before = margins(head.probe.predict(sample_x), sample_y)
    margin_after = margins(head.probe.predict(transported), sample_y)

    payload = {
        "run_id": run_id,
        "head": cfg.head,
        "sample_steps": sample_steps,
        "dim": dim,
        "rank_w": rank_w,
        "n_points": int(sample_x.shape[0]),
        "max_points": max_points,
        "summary": {
            "row_fraction_mean": float(row_fraction.mean()),
            "row_fraction_median": float(row_fraction.median()),
            "null_fraction_mean": float(null_fraction.mean()),
            "null_fraction_median": float(null_fraction.median()),
            "total_norm_mean": float(total_norm.mean()),
            "total_norm_median": float(total_norm.median()),
        },
        "points": {
            "total_norm": total_norm.tolist(),
            "row_norm": row_norm.tolist(),
            "null_norm": null_norm.tolist(),
            "margin_before": margin_before.tolist(),
            "margin_after": margin_after.tolist(),
        },
    }

    out = run_dir / "rowspace.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def rowspace_from_config_file(
    config_path: Path,
    *,
    data_root: Path = Path("data"),
    results_dir: Path = Path("results"),
    run_id: str | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> Path:
    return run_rowspace_diagnostics(
        load_config(config_path),
        data_root=data_root,
        results_dir=results_dir,
        run_id=run_id,
        max_points=max_points,
    )
