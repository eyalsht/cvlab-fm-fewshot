"""FM block as the last layer, trained by standard FM (Stage 2, FR10).

The head is a classifier and a describable one. Stage 1 decides by

    argmax_c cos(z, p_c)

and this head decides by

    argmax_c cos(psi_theta(z), p_c)

where psi_theta is T Euler steps of one weight-tied MLP and the prototypes are
Stage 1's, unchanged. So the question Stage 2 asks is not whether flow matching
works; it is whether that particular constrained classifier generalizes better
than the prototype rule on frozen features.

The decision rule is not reimplemented here. The head holds a fitted
PrototypeHead, transports the query, and calls that head's predict on the
transported features (ADR-020). One cosine rule exists in this repository and
both stages call it.

Features enter the flow raw (ADR-018). The write-up says "a frozen image
feature z_i" and never normalizes it, while the prototypes are unit norm by his
own formula, so the endpoints sit at different scales and u_i = p_{y_i} - z_i
is dominated by -z_i. The literal reading is what runs and what gets reported;
l2_normalize stays available as a one-cell ablation for the phase note.

Validation tensors are accepted and ignored: the write-up trains for a fixed
number of steps against one scalar loss, with nothing to select (ADR-014).
"""

from torch import Tensor

from fm_fewshot.services.flow.training.standard import (
    train_standard_field,
    transport,
    transport_trajectory,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import (
    FewShotHead,
    HeadContext,
    NotFittedError,
    register,
)
from fm_fewshot.services.heads.prototype import PrototypeHead
from fm_fewshot.shared.contracts import ExperimentConfig


@register("fm_standard")
class FmStandardHead(FewShotHead):
    def __init__(
        self,
        n_classes: int,
        *,
        sample_steps: int = 4,
        n_train_steps: int = 2000,
        batch_size: int = 64,
        lr: float = 1e-3,
        hidden_dims: tuple[int, ...] = (512, 512),
        time_conditioning: str = "scalar",
        init_seed: int = 0,
    ) -> None:
        if sample_steps < 1:
            raise ValueError(f"sample_steps must be >= 1, got {sample_steps}")
        self._n_classes = n_classes
        self._sample_steps = sample_steps
        self._n_train_steps = n_train_steps
        self._batch_size = batch_size
        self._lr = lr
        self._hidden_dims = tuple(hidden_dims)
        self._time_conditioning = time_conditioning
        self._init_seed = init_seed
        self._prototype_head = PrototypeHead(n_classes=n_classes)
        self._field: VelocityMLP | None = None
        self._loss_history: list[float] = []

    @classmethod
    def from_context(
        cls, cfg: ExperimentConfig, n_classes: int, context: HeadContext | None = None
    ) -> "FmStandardHead":
        params = cfg.head_params
        return cls(
            n_classes=n_classes,
            sample_steps=int(params.get("sample_steps", 4)),
            n_train_steps=int(params.get("n_train_steps", 2000)),
            batch_size=int(params.get("batch_size", 64)),
            lr=float(params.get("lr", 1e-3)),
            hidden_dims=tuple(params.get("hidden_dims", (512, 512))),
            time_conditioning=str(params.get("time_conditioning", "scalar")),
            init_seed=cfg.init_seed,
        )

    @property
    def field(self) -> VelocityMLP:
        if self._field is None:
            raise NotFittedError("the velocity field is undefined before fit")
        return self._field

    @property
    def prototypes(self) -> Tensor:
        """The Stage 1 prototypes this head transports toward, [C, D] unit norm."""
        return self._prototype_head.prototypes

    @property
    def loss_history(self) -> list[float]:
        """Per-step training loss, written to loss_curve.csv by the loop (ADR-024)."""
        return list(self._loss_history)

    def fit(self, train_x: Tensor, train_y: Tensor, val_x: Tensor, val_y: Tensor) -> None:
        self._prototype_head.fit(train_x, train_y, val_x, val_y)
        targets = self._prototype_head.prototypes[train_y]
        self._field, self._loss_history = train_standard_field(
            train_x,
            targets,
            hidden_dims=self._hidden_dims,
            time_conditioning=self._time_conditioning,
            n_train_steps=self._n_train_steps,
            batch_size=self._batch_size,
            lr=self._lr,
            init_seed=self._init_seed,
        )

    def transport(self, query_x: Tensor) -> Tensor:
        """Where the query lands after T Euler steps. The S3 figures and the
        ADR-018 diagnostics read this."""
        return _detached(self.field, query_x, self._sample_steps, transport)

    def trajectory(self, query_x: Tensor) -> Tensor:
        """Every intermediate state, [T + 1, M, D], for the S4 trajectory figures."""
        return _detached(self.field, query_x, self._sample_steps, transport_trajectory)

    def predict(self, query_x: Tensor) -> Tensor:
        return self._prototype_head.predict(self.transport(query_x))


def _detached(field: VelocityMLP, x: Tensor, sample_steps: int, integrate) -> Tensor:
    import torch

    was_training = field.training
    field.eval()
    try:
        with torch.no_grad():
            return integrate(field, x, sample_steps=sample_steps)
    finally:
        field.train(was_training)
