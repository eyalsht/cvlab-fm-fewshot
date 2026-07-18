"""Closed-form ridge classifier (PRD FR13), optional fourth Stage 1 baseline.

Ridge regression on one-hot targets, W = (X^T X + lam I)^-1 X^T Y, solved once
per episode. It never collapses where the unregularized linear probe does at low
shot (a lesson from the first author's prior course project) and doubles as a
cross-check on the probe's regularization. No gradient training, no RNG; the fit
is deterministic.
"""

import torch
from torch import Tensor

from fm_fewshot.services.heads.base import (
    FewShotHead,
    HeadContext,
    NotFittedError,
    register,
)
from fm_fewshot.shared.contracts import Episode, ExperimentConfig


@register("ridge")
class RidgeHead(FewShotHead):
    def __init__(self, n_way: int, *, lam: float = 1.0) -> None:
        if lam < 0:
            raise ValueError(f"lam must be >= 0, got {lam}")
        self._n_way = n_way
        self._lam = lam
        self._weight: Tensor | None = None  # [D, n_way]

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, episode: Episode, context: HeadContext
    ) -> "RidgeHead":
        return cls(n_way=episode.n_way, lam=float(cfg.head_params.get("lam", 1.0)))

    @property
    def weight(self) -> Tensor:
        if self._weight is None:
            raise NotFittedError("weight is undefined before fit")
        return self._weight

    def fit(self, support_x: Tensor, support_y: Tensor) -> None:
        present = set(support_y.tolist())
        missing = [j for j in range(self._n_way) if j not in present]
        if missing:
            raise ValueError(f"support is missing class ids {missing}; every class must appear")

        n, dim = support_x.shape
        targets = torch.zeros(n, self._n_way, dtype=support_x.dtype)
        targets[torch.arange(n), support_y] = 1.0
        gram = support_x.T @ support_x + self._lam * torch.eye(dim, dtype=support_x.dtype)
        self._weight = torch.linalg.solve(gram, support_x.T @ targets)

    def predict(self, query_x: Tensor) -> Tensor:
        if self._weight is None:
            raise NotFittedError("predict called before fit")
        return (query_x @ self._weight).float()
