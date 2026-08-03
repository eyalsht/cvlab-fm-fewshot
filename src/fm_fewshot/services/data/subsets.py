"""Balanced K-per-class subsets of the official train split (ADR-012).

The supervisor's protocol draws K images per class from the official training
split at subset seeds {0, 1, 2}, and evaluates on the complete official test
split. K = None is the full setting and returns the whole train split.

Seeding is SeedSequence([seed, k]), so the 5-shot and 10-shot draws at one seed
are independent rather than nested. That is deliberate: a nested design would
correlate the two settings' errors and understate the spread across the three
runs the protocol reports.

This module only ever sees the training labels. The signature takes a single
label array precisely so validation or test rows cannot be mixed in.
"""

import numpy as np
import torch
from torch import Tensor

from fm_fewshot.shared.contracts import TrainSubset


class InsufficientClassError(ValueError):
    """Raised when a class holds fewer than k training rows."""


def balanced_subset(
    train_labels: Tensor,
    k: int | None,
    seed: int,
    n_classes: int,
    dataset: str,
) -> TrainSubset:
    """Draw exactly k rows per class, or every row when k is None."""
    _validate(train_labels, k, seed, n_classes)

    if k is None:
        idx = torch.arange(train_labels.shape[0], dtype=torch.int64)
        return TrainSubset(
            dataset=dataset,
            k=None,
            seed=seed,
            n_classes=n_classes,
            idx=idx,
            labels=train_labels[idx].clone(),
        )

    rng = np.random.default_rng(np.random.SeedSequence([seed, k]))
    chosen: list[np.ndarray] = []
    for class_id in range(n_classes):
        rows = torch.nonzero(train_labels == class_id, as_tuple=False).flatten().numpy()
        if rows.size < k:
            raise InsufficientClassError(
                f"class {class_id} has {rows.size} training rows, fewer than the "
                f"{k} requested; k={k} is impossible on this split"
            )
        chosen.append(rng.choice(rows, size=k, replace=False))

    # Ascending and deduplicated: a subset is a set, not a sequence. Batch order
    # is the training loop's business and comes from its own seed.
    idx = torch.from_numpy(np.sort(np.concatenate(chosen))).to(torch.int64)
    return TrainSubset(
        dataset=dataset,
        k=k,
        seed=seed,
        n_classes=n_classes,
        idx=idx,
        labels=train_labels[idx].clone(),
    )


def _validate(train_labels: Tensor, k: int | None, seed: int, n_classes: int) -> None:
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1 or None for the full split, got {k}")
    if train_labels.ndim != 1:
        raise ValueError(f"train_labels must be 1-D, got shape {tuple(train_labels.shape)}")
