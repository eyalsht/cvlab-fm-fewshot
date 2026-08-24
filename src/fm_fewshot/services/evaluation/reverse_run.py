"""The `reverse` command: refit a Stage 2 run, integrate backwards, write JSON.

FR18. The metrics themselves live in `services/flow/reverse.py`; this module
only finds the run, reproduces its field, and assembles
`results/<run_id>/reverse.json` in the shape `PRD_reverse_flow.md` section 2
gives.

Two rules the placement of this file enforces.

First, ADR-025: the field is refitted deterministically from the stored
`config.yaml`, never trained separately and never loaded from a checkpoint.
Same seed, same device, bit-identical fit, so the diagnostics describe the
field that produced the run's accuracy and not a near neighbour of it.

Second, ADR-027: these numbers are diagnostics. They read the true test labels,
which is permitted on the same footing as the Stage 1 confusion matrices, and
in exchange no number produced here may inform a head configuration or reach
`TABLE.md`. Nothing in `loop.py`, `report.py` or `sweep.py` imports this
module, and `reverse.json` is not a file the report path knows about.
"""

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from fm_fewshot.services.data.subsets import balanced_subset
from fm_fewshot.services.evaluation.loop import git_commit
from fm_fewshot.services.features.cache import read_features
from fm_fewshot.services.flow.reverse import (
    REVERSE_MODES,
    ReverseConfig,
    basin_alignment,
    basin_samples,
    cycle_relative_error,
    log_volume_change,
    median_iqr,
    meet_gaps,
    residual_sigma,
)
from fm_fewshot.services.heads import make_head
from fm_fewshot.services.heads.fm_base import FmHead
from fm_fewshot.shared.config import load_config
from fm_fewshot.shared.contracts import ExperimentConfig

DEFAULT_MODES = ("cycle", "basin", "meet")

# The fields that identify which stored run a config describes. run_name is
# excluded: two configs that agree on all of these are the same cell and the
# same seed pair whatever they were called.
IDENTIFYING = ("dataset", "encoder", "head", "variant", "k", "subset_seed", "init_seed")


class MissingRunError(FileNotFoundError):
    """Raised when no stored run matches the config the caller asked about."""


class NotAnFmRunError(ValueError):
    """Raised for a head with no velocity field to integrate."""


def find_run(results_dir: Path, cfg: ExperimentConfig) -> str:
    """The stored run this config describes, latest first if several match."""
    results_dir = Path(results_dir)
    wanted = {name: getattr(cfg, name) for name in IDENTIFYING}
    matches = []
    for run_dir in sorted(results_dir.iterdir()) if results_dir.exists() else []:
        stored = run_dir / "config.yaml"
        if not stored.exists():
            continue
        candidate = load_config(stored)
        if all(getattr(candidate, name) == value for name, value in wanted.items()):
            matches.append(run_dir.name)
    if not matches:
        raise MissingRunError(
            f"no stored run under {results_dir} matches {wanted}; reverse flow "
            "annotates a run that already happened, so run the forward cell first"
        )
    return matches[-1]


def refit_head(cfg: ExperimentConfig, data_root: Path):  # noqa: ANN201 - duck-typed head
    """Reproduce the run's fitted head from its config (ADR-025).

    The call order mirrors `loop.run_experiment` exactly, and the test split is
    still opened only after `fit` returns, by the caller.
    """
    train_x, train_y, train_meta = read_features(
        cfg.dataset, "train", cfg.encoder, data_root=data_root, l2_normalize=cfg.l2_normalize
    )
    n_classes = len(train_meta["class_names"])
    head = make_head(cfg, n_classes)
    if not isinstance(head, FmHead):
        raise NotAnFmRunError(
            f"head {cfg.head!r} has no velocity field; reverse flow is defined "
            "for the FM heads only"
        )
    val_x, val_y, _ = read_features(
        cfg.dataset, "val", cfg.encoder, data_root=data_root, l2_normalize=cfg.l2_normalize
    )
    subset = balanced_subset(train_y, cfg.k, cfg.subset_seed, n_classes, cfg.dataset)
    head.fit(train_x[subset.idx], subset.labels, val_x, val_y)
    return head, train_x[subset.idx], subset.labels, n_classes


def run_reverse(
    cfg: ExperimentConfig,
    *,
    reverse_cfg: ReverseConfig | None = None,
    modes: Sequence[str] = DEFAULT_MODES,
    data_root: Path = Path("data"),
    results_dir: Path = Path("results"),
    run_id: str | None = None,
) -> Path:
    """Write `results/<run_id>/reverse.json` for one Stage 2 run."""
    reverse_cfg = reverse_cfg or ReverseConfig()
    unknown = [m for m in modes if m not in REVERSE_MODES]
    if unknown:
        raise ValueError(f"unknown reverse modes {unknown}; expected {REVERSE_MODES}")

    results_dir = Path(results_dir)
    run_id = run_id or find_run(results_dir, cfg)
    run_dir = results_dir / run_id
    if not run_dir.exists():
        raise MissingRunError(f"no run directory {run_dir}")

    head, subset_x, subset_y, n_classes = refit_head(cfg, data_root)
    # One source for T: the same helper both FM heads read their configuration
    # through, so the reverse legs cannot drift from the forward map (ADR-022).
    sample_steps = int(FmHead.shared_params(cfg, n_classes)["sample_steps"])
    # Only now. Everything above this line ran without a test row, as in the
    # forward loop; the labels below are read as diagnostics (ADR-027).
    test_x, test_y, _ = read_features(
        cfg.dataset, cfg.eval_split, cfg.encoder,
        data_root=data_root, l2_normalize=cfg.l2_normalize,
    )

    field = head.field
    was_training = field.training
    field.eval()
    try:
        payload: dict[str, object] = {
            "run_id": run_id,
            "git_commit": git_commit(),
            "config": dataclasses.asdict(cfg),
            "reverse_config": dataclasses.asdict(reverse_cfg),
            "modes": list(modes),
            "forward_steps": sample_steps,
        }
        if "cycle" in modes:
            payload["cycle"] = _cycle_family(
                field, reverse_cfg, sample_steps, test_x, test_y, run_dir
            )
        if "basin" in modes:
            payload["basin"] = _basin_family(
                head, field, reverse_cfg, sample_steps, cfg.seed,
                subset_x, subset_y, test_x, test_y, n_classes,
            )
        if "meet" in modes:
            payload["meet"] = _meet_family(
                head, field, reverse_cfg, sample_steps, subset_x, subset_y
            )
        if "volume" in modes:
            payload["volume"] = _volume_family(
                field, reverse_cfg, sample_steps, test_x, cfg.seed
            )
    finally:
        field.train(was_training)

    out = run_dir / "reverse.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _correctness(run_dir: Path, test_y: torch.Tensor) -> torch.Tensor | None:
    """Which test rows the forward run got right, from its stored predictions."""
    preds_path = run_dir / "preds.npy"
    if not preds_path.exists():
        return None
    predictions = torch.from_numpy(np.load(preds_path)).long()
    if predictions.shape[0] != test_y.shape[0]:
        return None
    return predictions == test_y


def _cycle_family(field, reverse_cfg, sample_steps, test_x, test_y, run_dir):  # noqa: ANN001
    """rel_error over the test split at every (T, return solver) of ADR-026.

    The forward leg is the head's own map, T Euler steps at its `sample_steps`,
    so the start point really is `z_T = psi(z)`. Only the return leg sweeps.
    Reading the numbers means reading the T trend: the solver component falls
    with T and contraction does not.
    """
    correct = _correctness(run_dir, test_y)
    family: dict[str, dict[str, object]] = {}
    for steps in reverse_cfg.reverse_steps:
        per_method: dict[str, object] = {}
        for method in reverse_cfg.reverse_method:
            error = cycle_relative_error(
                field,
                test_x,
                forward_steps=sample_steps,
                reverse_steps=steps,
                forward_method="euler",
                reverse_method=method,
            )
            entry: dict[str, object] = {"rel_error": median_iqr(error)}
            if correct is not None:
                entry["rel_error_correct"] = median_iqr(error[correct])
                entry["rel_error_incorrect"] = median_iqr(error[~correct])
            per_method[method] = entry
        family[str(steps)] = per_method
    return family


def _basin_family(  # noqa: ANN001, PLR0913
    head, field, reverse_cfg, sample_steps, seed, subset_x, subset_y, test_x, test_y,
    n_classes,
):
    """Reverse clouds from every prototype, and how they align with the data.

    The reverse depth is the head's own `sample_steps`, because the basin is
    the pre-image of the map the head applies rather than of an idealized flow.
    """
    prototypes = head.prototypes
    sigma = residual_sigma(head.transport(subset_x), prototypes[subset_y], reverse_cfg.sigma_scale)
    clouds = basin_samples(
        field,
        prototypes,
        sigma=sigma,
        n_samples=reverse_cfg.n_samples,
        n_steps=sample_steps,
        seed=seed,
        method=reverse_cfg.reverse_method[0],
    )
    alignment = basin_alignment(clouds, test_x, test_y, n_classes)
    return {
        "alignment": alignment.tolist(),
        "sigma": sigma,
        "sigma_scale": reverse_cfg.sigma_scale,
        "n_samples": reverse_cfg.n_samples,
        "reverse_steps": sample_steps,
        "reverse_method": reverse_cfg.reverse_method[0],
    }


def _meet_family(head, field, reverse_cfg, sample_steps, subset_x, subset_y):  # noqa: ANN001
    """Where the two legs meet, against the true interpolant, per t*."""
    targets = head.prototypes[subset_y]
    return {
        str(t_star): meet_gaps(
            field, subset_x, targets, t_star=t_star, n_steps=sample_steps
        )
        for t_star in reverse_cfg.meet_times
    }


def _volume_family(field, reverse_cfg, sample_steps, test_x, seed):  # noqa: ANN001
    """Per-point log volume change along the forward trajectory (stretch)."""
    change = log_volume_change(
        field,
        test_x,
        n_steps=sample_steps,
        n_hutchinson=reverse_cfg.n_hutchinson,
        seed=seed,
    )
    return {"log_volume_change": median_iqr(change), "n_hutchinson": reverse_cfg.n_hutchinson}


def reverse_from_config_file(
    config_path: Path,
    *,
    reverse_cfg: ReverseConfig | None = None,
    modes: Sequence[str] = DEFAULT_MODES,
    data_root: Path = Path("data"),
    results_dir: Path = Path("results"),
    run_id: str | None = None,
) -> Path:
    return run_reverse(
        load_config(config_path),
        reverse_cfg=reverse_cfg,
        modes=modes,
        data_root=data_root,
        results_dir=results_dir,
        run_id=run_id,
    )
