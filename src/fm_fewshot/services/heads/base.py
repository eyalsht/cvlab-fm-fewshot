"""FewShotHead ABC, the head registry, and the make_head factory.

Every method in the study, baseline or FM, implements this one interface so the
evaluation loop never branches per method (ADR-002). Heads register under a key
and are built through make_head, which hands each head the Episode (for the
class ids a head like zeroshot_clip needs) and a HeadContext seam carrying any
precomputed tensors the head cannot derive from support features alone. The
seam is how zeroshot_clip receives its class text embeddings without a head
ever loading open_clip inside an experiment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor

from fm_fewshot.shared.contracts import Episode, ExperimentConfig


class NotFittedError(RuntimeError):
    """Raised when predict is called before fit."""


@dataclass(frozen=True)
class HeadContext:
    """Construction-time inputs beyond cfg and episode.

    text_embeddings holds the per-episode class text rows for zeroshot_clip,
    ordered to match episode.class_ids; every other head ignores it.
    """

    text_embeddings: Tensor | None = None


class FewShotHead(ABC):
    @abstractmethod
    def fit(self, support_x: Tensor, support_y: Tensor) -> None:
        """Fit on one episode's support set; support_y holds local labels in [0, n_way)."""

    @abstractmethod
    def predict(self, query_x: Tensor) -> Tensor:
        """Return logits [M, n_way]; column j scores local label j."""

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, episode: Episode, context: HeadContext
    ) -> "FewShotHead":
        raise NotImplementedError(f"{cls.__name__} does not implement from_context")


_REGISTRY: dict[str, type[FewShotHead]] = {}


def register(key: str):  # noqa: ANN201 - decorator returns the class unchanged
    def decorate(cls: type[FewShotHead]) -> type[FewShotHead]:
        if key in _REGISTRY:
            raise ValueError(
                f"head key {key!r} already registered to {_REGISTRY[key].__name__}"
            )
        _REGISTRY[key] = cls
        return cls

    return decorate


def registered_heads() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def make_head(
    cfg: ExperimentConfig, episode: Episode, context: HeadContext | None = None
) -> FewShotHead:
    if cfg.head not in _REGISTRY:
        raise ValueError(
            f"unknown head {cfg.head!r}; registered heads are {list(registered_heads())}"
        )
    return _REGISTRY[cfg.head].from_context(cfg, episode, context or HeadContext())
