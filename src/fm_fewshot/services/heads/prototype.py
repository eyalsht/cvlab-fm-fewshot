"""Image-derived class prototypes, the supervisor's Stage 1 Option A.

The formula is his, followed literally:

    mu_c = normalize( (1 / |S_c|) * sum_{i in S_c} normalize(z_i) )
    y_hat = argmax_c cos(z, mu_c)

The inner normalization is not redundant with the outer one. Normalizing before
averaging makes every image contribute equally regardless of its feature norm;
averaging raw features and normalizing once would weight high-norm images more.
He specified the first.

The head has no configuration. Its only stochasticity is which images the
subset sampler chose, so fit is closed-form arithmetic with no seed.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fm_fewshot.services.heads.base import (
    FewShotHead,
    HeadContext,
    NotFittedError,
    register,
)
from fm_fewshot.shared.contracts import ExperimentConfig


@register("prototype")
class PrototypeHead(FewShotHead):
    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes
        self._prototypes: Tensor | None = None  # [C, D], unit norm

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, n_classes: int, context: HeadContext
    ) -> "PrototypeHead":
        return cls(n_classes=n_classes)

    @property
    def prototypes(self) -> Tensor:
        if self._prototypes is None:
            raise NotFittedError("prototypes are undefined before fit")
        return self._prototypes

    def fit(self, train_x: Tensor, train_y: Tensor, val_x: Tensor, val_y: Tensor) -> None:
        # val_* are accepted and ignored: there is nothing to select (ADR-014).
        if not torch.isfinite(train_x).all():
            raise ValueError("non-finite values in training features")
        norms = train_x.norm(dim=1)
        if bool((norms == 0).any()):
            zero_row = int((norms == 0).nonzero()[0].item())
            raise ValueError(
                f"training feature row {zero_row} has zero norm and no direction; "
                "a zero feature cannot be normalized"
            )

        normalized = F.normalize(train_x, dim=1)
        prototypes = torch.empty(self._n_classes, train_x.shape[1])
        for c in range(self._n_classes):
            class_x = normalized[train_y == c]
            if class_x.shape[0] == 0:
                raise ValueError(f"class {c} has no training example; every class must appear")
            prototypes[c] = class_x.mean(dim=0)
        self._prototypes = F.normalize(prototypes, dim=1)

    def predict(self, query_x: Tensor) -> Tensor:
        if self._prototypes is None:
            raise NotFittedError("predict called before fit")
        return (F.normalize(query_x, dim=1) @ self._prototypes.T).float()
