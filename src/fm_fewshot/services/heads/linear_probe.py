"""Multiclass linear probe on frozen features, the supervisor's required baseline.

    s = W z + b,  only W and b trained, softmax cross-entropy

Hyperparameters are his suggested initial config: AdamW, lr 1e-3, weight decay
1e-4, batch size 64, max 200 epochs, checkpoint selected on the highest
validation accuracy. He permits adjusting them from validation results provided
the change is reported, so each is a config field and the run config records
what was used.

Weight decay is AdamW's decoupled decay, not a penalty term added to the loss.
The optimizer he named is the specification.

One RNG stream, seeded from init_seed, drives both the weight initialization
and the batch order, so the full setting's three initialization seeds vary the
classifier while the training subset stays fixed (ADR-012).
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
from fm_fewshot.shared.contracts import EpochRecord, ExperimentConfig


@register("linear_probe")
class LinearProbeHead(FewShotHead):
    def __init__(
        self,
        n_classes: int,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        init_seed: int = 0,
    ) -> None:
        self._n_classes = n_classes
        self._lr = lr
        self._weight_decay = weight_decay
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._init_seed = init_seed
        self._weight: Tensor | None = None
        self._bias: Tensor | None = None
        self._epochs: list[EpochRecord] = []
        self._best_epoch: int | None = None

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, n_classes: int, context: HeadContext
    ) -> "LinearProbeHead":
        params = cfg.head_params
        return cls(
            n_classes=n_classes,
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
            batch_size=int(params.get("batch_size", 64)),
            max_epochs=int(params.get("max_epochs", 200)),
            init_seed=cfg.init_seed,
        )

    @property
    def epochs(self) -> list[EpochRecord]:
        return list(self._epochs)

    @property
    def best_epoch(self) -> int:
        if self._best_epoch is None:
            raise NotFittedError("best_epoch is undefined before fit")
        return self._best_epoch

    def fit(self, train_x: Tensor, train_y: Tensor, val_x: Tensor, val_y: Tensor) -> None:
        self._validate(train_x, train_y)

        generator = torch.Generator().manual_seed(self._init_seed)
        dim = train_x.shape[1]
        # torch.nn.Linear's default bound, drawn from our own generator so the
        # seed alone determines the initial weights.
        bound = (1.0 / dim) ** 0.5
        weight = torch.empty(self._n_classes, dim).uniform_(-bound, bound, generator=generator)
        bias = torch.empty(self._n_classes).uniform_(-bound, bound, generator=generator)
        weight.requires_grad_(True)
        bias.requires_grad_(True)

        optimizer = torch.optim.AdamW([weight, bias], lr=self._lr, weight_decay=self._weight_decay)

        best_accuracy = -1.0
        best_state: tuple[Tensor, Tensor] | None = None
        self._epochs = []

        for epoch in range(1, self._max_epochs + 1):
            order = torch.randperm(train_x.shape[0], generator=generator)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, order.shape[0], self._batch_size):
                rows = order[start : start + self._batch_size]
                loss = F.cross_entropy(train_x[rows] @ weight.T + bias, train_y[rows])
                if not torch.isfinite(loss):
                    raise ValueError(
                        f"non-finite training loss at epoch {epoch}, batch {n_batches}"
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                n_batches += 1

            with torch.no_grad():
                val_logits = val_x @ weight.T + bias
                val_loss = float(F.cross_entropy(val_logits, val_y))
                val_accuracy = float((val_logits.argmax(dim=1) == val_y).float().mean())

            self._epochs.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=total_loss / max(n_batches, 1),
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                )
            )
            # Strictly greater keeps the earlier epoch on a tie (ADR-014).
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_state = (weight.detach().clone(), bias.detach().clone())
                self._best_epoch = epoch

        assert best_state is not None  # max_epochs >= 1 is validated above
        self._weight, self._bias = best_state

    def _validate(self, train_x: Tensor, train_y: Tensor) -> None:
        if self._max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self._max_epochs}")
        if self._batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self._batch_size}")
        if not torch.isfinite(train_x).all():
            raise ValueError("non-finite values in training features")
        missing = sorted(set(range(self._n_classes)) - set(train_y.tolist()))
        if missing:
            raise ValueError(f"classes {missing} have no training example; every class must appear")

    def predict(self, query_x: Tensor) -> Tensor:
        if self._weight is None or self._bias is None:
            raise NotFittedError("predict called before fit")
        with torch.no_grad():
            return (query_x @ self._weight.T + self._bias).float()
