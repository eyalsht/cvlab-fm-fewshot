"""Multinomial logistic regression on one episode's support set, Stage 1 baseline.

Also the reference last layer that Stage 2 replaces. Zero initialization and
full-batch Adam make the whole fit deterministic with no RNG: two fits on the
same support give bit-identical parameters. Weight decay is applied as an
explicit L2 term on the weight (not Adam's coupled decay), because
regularization strength is the anti-overfit axis and must read the same as the
loss written in the PRD.
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
from fm_fewshot.shared.contracts import Episode, ExperimentConfig


@register("linear_probe")
class LinearProbeHead(FewShotHead):
    def __init__(
        self,
        n_way: int,
        *,
        n_steps: int = 100,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        optimizer: str = "adam",
    ) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if optimizer != "adam":
            raise ValueError(f"unsupported optimizer {optimizer!r}; only 'adam' is implemented")
        self._n_way = n_way
        self._n_steps = n_steps
        self._lr = lr
        self._weight_decay = weight_decay
        self._weight: Tensor | None = None
        self._bias: Tensor | None = None
        self.loss_history: list[float] = []

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, episode: Episode, context: HeadContext
    ) -> "LinearProbeHead":
        params = cfg.head_params
        return cls(
            n_way=episode.n_way,
            n_steps=int(params.get("n_steps", 100)),
            lr=float(params.get("lr", 1e-2)),
            weight_decay=float(params.get("weight_decay", 1e-2)),
            optimizer=str(params.get("optimizer", "adam")),
        )

    @property
    def weight(self) -> Tensor:
        if self._weight is None:
            raise NotFittedError("weight is undefined before fit")
        return self._weight

    @property
    def bias(self) -> Tensor:
        if self._bias is None:
            raise NotFittedError("bias is undefined before fit")
        return self._bias

    def fit(self, support_x: Tensor, support_y: Tensor) -> None:
        present = set(support_y.tolist())
        missing = [j for j in range(self._n_way) if j not in present]
        if missing:
            raise ValueError(f"support is missing class ids {missing}; every class must appear")

        dim = support_x.shape[1]
        weight = torch.zeros(self._n_way, dim, requires_grad=True)
        bias = torch.zeros(self._n_way, requires_grad=True)
        opt = torch.optim.Adam([weight, bias], lr=self._lr)
        history: list[float] = []
        for step in range(self._n_steps):
            opt.zero_grad()
            logits = support_x @ weight.T + bias
            loss = F.cross_entropy(logits, support_y) + self._weight_decay * weight.pow(2).sum()
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at step {step}; lr or features too large")
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
        self._weight = weight.detach()
        self._bias = bias.detach()
        self.loss_history = history

    def predict(self, query_x: Tensor) -> Tensor:
        if self._weight is None or self._bias is None:
            raise NotFittedError("predict called before fit")
        return (query_x @ self._weight.T + self._bias).float()
