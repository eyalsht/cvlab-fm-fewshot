"""What the two FM heads share, which is everything except the training scheme.

Stage 1 decides by

    argmax_c cos(z, p_c)

and both Stage 2 heads decide by

    argmax_c cos(psi_theta(z), p_c)

where psi_theta is T Euler steps of one weight-tied MLP and the prototypes are
Stage 1's, unchanged. So the question Stage 2 asks is not whether flow matching
works; it is whether that particular constrained classifier generalizes better
than the prototype rule on frozen features, and then whether the two ways of
fitting it differ.

That second question is only readable if the two heads differ in the fit and
nowhere else, so the fit is the one thing a subclass supplies. The write-up
says as much: keep the network architecture and the main training choices fixed
when comparing the schemes.

The decision rule is not reimplemented here. The head holds a fitted
PrototypeHead, transports the query, and calls that head's predict on the
transported features (ADR-020). One cosine rule exists in this repository and
both stages call it.

Features enter the flow raw (ADR-018). The write-up says "a frozen image
feature z_i" and never normalizes it, while the prototypes are unit norm by his
own formula, so the endpoints sit at different scales and u_i = p_{y_i} - z_i
is dominated by -z_i. The literal reading is what runs and what gets reported;
l2_normalize stays available as a one-cell ablation for the phase note.

Validation tensors are used, and used the same way by both schemes. Every
`eval_every` steps the current field transports the validation split at the
run's own `sample_steps`, the fitted PrototypeHead scores where it lands, and
the best-scoring field is the one `predict` runs on (ADR-028). This is the rule
ADR-014 specified and the linear probe has always followed; the FM heads kept
the final field until the Stage 2 grid showed rolled-out training fitting its
training transport perfectly and generalizing to nothing. `eval_every = 0`
restores the fixed-schedule behaviour exactly, which is how the decision is
reversed if the supervisor prefers it.

The selection machinery itself is not here. It is one class in
`flow/training/base`, constructed here and driven by the shared training loop,
so the two schemes cannot select differently: only the loss differs.
"""

from abc import abstractmethod

import torch
from torch import Tensor

from fm_fewshot.services.flow.training.base import (
    ValidationSelector,
    transport,
    transport_trajectory,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import FewShotHead, NotFittedError
from fm_fewshot.services.heads.prototype import PrototypeHead
from fm_fewshot.shared.contracts import ExperimentConfig

# 40 selection points over the 2000-step schedule. Measured on the grid's
# slowest fit, rolled-out T=12 at K=full on Aircraft/ResNet-18: 40.8 s with
# selection against 33.7 s without, inside NFR1's 60 seconds (ADR-028).
DEFAULT_EVAL_EVERY = 50


class FmHead(FewShotHead):
    """An FM block as the last layer. Subclasses differ only in `_fit_field`."""

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
        eval_every: int = DEFAULT_EVAL_EVERY,
        init_seed: int = 0,
    ) -> None:
        if sample_steps < 1:
            raise ValueError(f"sample_steps must be >= 1, got {sample_steps}")
        if eval_every < 0:
            raise ValueError(f"eval_every must be >= 0, got {eval_every}")
        self._n_classes = n_classes
        self._sample_steps = sample_steps
        self._n_train_steps = n_train_steps
        self._batch_size = batch_size
        self._lr = lr
        self._hidden_dims = tuple(hidden_dims)
        self._time_conditioning = time_conditioning
        self._eval_every = eval_every
        self._init_seed = init_seed
        self._prototype_head = PrototypeHead(n_classes=n_classes)
        self._field: VelocityMLP | None = None
        self._loss_history: list[float] = []
        self._selector: ValidationSelector | None = None

    @staticmethod
    def shared_params(cfg: ExperimentConfig, n_classes: int) -> dict[str, object]:
        """The constructor arguments both schemes read, from one place so they agree."""
        params = cfg.head_params
        return {
            "n_classes": n_classes,
            "sample_steps": int(params.get("sample_steps", 4)),
            "n_train_steps": int(params.get("n_train_steps", 2000)),
            "batch_size": int(params.get("batch_size", 64)),
            "lr": float(params.get("lr", 1e-3)),
            "hidden_dims": tuple(params.get("hidden_dims", (512, 512))),
            "time_conditioning": str(params.get("time_conditioning", "scalar")),
            "eval_every": int(params.get("eval_every", DEFAULT_EVAL_EVERY)),
            "init_seed": cfg.init_seed,
        }

    @abstractmethod
    def _fit_field(
        self, train_x: Tensor, targets: Tensor, selector: ValidationSelector
    ) -> tuple[VelocityMLP, list[float]]:
        """Train v_theta on the coupling (train_x[i] -> targets[i]); return it and its curve.

        The selector is handed to the shared loop untouched. A scheme that
        inspected it, or built its own, would be selecting on its own terms.
        """

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

    @property
    def val_top1_history(self) -> list[tuple[int, float]]:
        """(step, val top-1) on the evaluation grid; empty when eval_every is 0.

        Duck-typed, not part of the FewShotHead contract: the loop reads it for
        the val_top1 column of loss_curve.csv and nowhere else (ADR-024).
        """
        return self._selector.history if self._selector is not None else []

    @property
    def best_epoch(self) -> int | None:
        """The training step whose field was kept, None when selection is off.

        Named for the field it populates in summary.json, which the linear
        probe fills with an epoch. Here the unit is a training step: FM trains
        in steps and never sees an epoch (ADR-028).
        """
        if self._field is None:
            raise NotFittedError("best_epoch is undefined before fit")
        return self._selector.best_step if self._selector is not None else None

    def fit(self, train_x: Tensor, train_y: Tensor, val_x: Tensor, val_y: Tensor) -> None:
        self._prototype_head.fit(train_x, train_y, val_x, val_y)
        targets = self._prototype_head.prototypes[train_y]
        # The classifier the selector scores with is the head's own, so the
        # number selection maximizes is the number the run reports (ADR-020).
        selector = ValidationSelector(
            val_x,
            val_y,
            self._prototype_head,
            sample_steps=self._sample_steps,
            eval_every=self._eval_every,
        )
        self._field, self._loss_history = self._fit_field(train_x, targets, selector)
        self._selector = selector

    def transport(self, query_x: Tensor) -> Tensor:
        """Where the query lands after T Euler steps. The S3 figures and the
        ADR-018 diagnostics read this."""
        return self._detached(query_x, transport)

    def trajectory(self, query_x: Tensor) -> Tensor:
        """Every intermediate state, [T + 1, M, D], for the S4 trajectory figures."""
        return self._detached(query_x, transport_trajectory)

    def predict(self, query_x: Tensor) -> Tensor:
        return self._prototype_head.predict(self.transport(query_x))

    def _detached(self, x: Tensor, integrate) -> Tensor:
        field = self.field
        was_training = field.training
        field.eval()
        try:
            with torch.no_grad():
                return integrate(field, x, sample_steps=self._sample_steps)
        finally:
            field.train(was_training)
