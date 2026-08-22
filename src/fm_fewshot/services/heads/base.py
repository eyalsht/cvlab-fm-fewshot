"""FewShotHead ABC, the head registry, and the make_head factory.

Every method in the study, baseline or FM, implements this one interface so the
evaluation loop never branches per method (ADR-002). Heads register under a key
and are built through make_head, which hands each head the class count and a
HeadContext seam carrying any precomputed tensors the head cannot derive from
its training features alone.

fit takes the validation tensors because checkpoint selection on validation
accuracy is specified, not optional (ADR-014). Heads with nothing to select
accept them and ignore them, so the loop needs no branch. The test split is
absent from this interface on purpose: a head has no route to it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor

from fm_fewshot.shared.contracts import ExperimentConfig


class NotFittedError(RuntimeError):
    """Raised when predict is called before fit."""


@dataclass(frozen=True)
class HeadContext:
    """Construction-time inputs beyond cfg and the class count.

    Empty for the Stage 1 heads. Kept as the seam for tensors a head cannot
    derive from its own training features, which the FM heads will need.
    """


class FewShotHead(ABC):
    @abstractmethod
    def fit(self, train_x: Tensor, train_y: Tensor, val_x: Tensor, val_y: Tensor) -> None:
        """Fit on the training subset; train_y holds global class ids in [0, C)."""

    @abstractmethod
    def predict(self, query_x: Tensor) -> Tensor:
        """Return logits [M, C]; column j scores class j."""

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, n_classes: int, context: HeadContext
    ) -> "FewShotHead":
        raise NotImplementedError(f"{cls.__name__} does not implement from_context")

    @property
    def loss_history(self) -> list[float]:
        """Per-step training loss; empty for heads with no stepwise training loop.

        The default covers the closed-form and checkpoint-selected heads without
        touching them. FM heads override this (ADR-024); the loop writes
        loss_curve.csv only when it is non-empty, so this property alone decides
        whether the file exists, with no branch on which head produced it.
        """
        return []


_REGISTRY: dict[str, type[FewShotHead]] = {}


def register(key: str):  # noqa: ANN201 - decorator returns the class unchanged
    def decorate(cls: type[FewShotHead]) -> type[FewShotHead]:
        if key in _REGISTRY:
            raise ValueError(f"head key {key!r} already registered to {_REGISTRY[key].__name__}")
        _REGISTRY[key] = cls
        return cls

    return decorate


def registered_heads() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def make_head(
    cfg: ExperimentConfig, n_classes: int, context: HeadContext | None = None
) -> FewShotHead:
    if cfg.head not in _REGISTRY:
        raise ValueError(
            f"unknown head {cfg.head!r}; registered heads are {list(registered_heads())}"
        )
    return _REGISTRY[cfg.head].from_context(cfg, n_classes, context or HeadContext())
