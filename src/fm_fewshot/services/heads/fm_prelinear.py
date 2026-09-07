"""What the two Stage 3 strategies share: everything except the objective.

Stage 1 decides by

    argmax_c (W z + b)_c

and both Stage 3 heads decide by

    argmax_c (W psi_theta(z) + b)_c

where psi_theta is T Euler steps of one weight-tied MLP and (W, b) is the
Stage 1 linear probe, fitted first and then frozen. So the question Stage 3
asks is not whether flow matching works, and not the Stage 2 question either:
it is whether a constrained nonlinear reparameterization of the features,
trained after the linear map is fixed, can beat the best linear map on those
same features.

Two properties this class exists to guarantee.

The classifier is the Stage 1 classifier. It is refitted here on the same
tensors at the same `init_seed` rather than loaded from a stored run, which
makes it bit-identical to the Stage 1 row by construction instead of by care,
and leaves the run self-contained (ADR-031). `subset_check` verifies the
identity across runs.

The untrained system is the probe exactly. The field is built with
`zero_output_init`, so before the first optimizer step `v_theta(z, t) = 0`,
T Euler steps leave every feature where it started, and the head's logits are
bit-identical to the probe's. The write-up asks for "close to identity"
(ADR-030); this is identity, and `test_fm_prelinear` asserts it as an equality.

Note the `_fit_field` signature against Stage 2's. `FmHead._fit_field` receives
a precomputed `targets` tensor because its coupling is fixed before training
starts. Here the trainer receives `train_y` and the frozen probe, because
neither strategy has a target that exists in advance: Strategy 1 has no target
at all, only a decision to optimize, and Strategy 2's target is rebuilt from
the classification gradient as the field moves.

The decision rule is not reimplemented. The head holds the fitted
`LinearProbeHead`, transports the query, and calls that head's `predict` on the
transported features, so one linear rule exists in this repository and Stage 1
and Stage 3 both call it. The selector scores with that same probe, which is
what makes the selection metric the metric the run reports (ADR-020).
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
from fm_fewshot.services.heads.fm_base import DEFAULT_EVAL_EVERY
from fm_fewshot.services.heads.linear_probe import LinearProbeHead
from fm_fewshot.shared.contracts import ExperimentConfig


class FmPreLinearHead(FewShotHead):
    """An FM block before a frozen linear probe. Subclasses differ only in `_fit_field`."""

    # "Also report the change relative to the corresponding linear-probe
    # baseline", which is the probe this block sits in front of (ADR-035).
    dacc_baseline = "linear_probe"

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
        zero_output_init: bool = True,
        probe_params: dict[str, object] | None = None,
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
        self._zero_output_init = zero_output_init
        self._init_seed = init_seed
        # The probe's hyperparameters are Stage 1's defaults unless a config
        # overrides them, and they live in their own namespace because `lr` and
        # `batch_size` mean one thing to the field and another to the probe.
        self._probe = LinearProbeHead(
            n_classes=n_classes, init_seed=init_seed, **_probe_kwargs(probe_params)
        )
        self._field: VelocityMLP | None = None
        self._loss_history: list[float] = []
        self._selector: ValidationSelector | None = None

    @staticmethod
    def shared_params(cfg: ExperimentConfig, n_classes: int) -> dict[str, object]:
        """The constructor arguments both strategies read, from one place so they agree."""
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
            "zero_output_init": bool(params.get("zero_output_init", True)),
            "probe_params": params.get("probe_params"),
            "init_seed": cfg.init_seed,
        }

    @abstractmethod
    def _fit_field(
        self,
        train_x: Tensor,
        train_y: Tensor,
        probe: LinearProbeHead,
        selector: ValidationSelector,
    ) -> tuple[VelocityMLP, list[float]]:
        """Train v_theta ahead of the probe; return it and its loss curve.

        The probe is handed over fitted and is read, never written, unless the
        strategy was explicitly configured for the optional extension (FR23).
        A strategy that updated W or b by default would be answering that
        question instead of the graded one, so the extension is opt-in and the
        probe records that it was reopened.
        """

    @property
    def field(self) -> VelocityMLP:
        if self._field is None:
            raise NotFittedError("the velocity field is undefined before fit")
        return self._field

    @property
    def probe(self) -> LinearProbeHead:
        """The frozen Stage 1 classifier this head transports into."""
        return self._probe

    @property
    def classifier_digest(self) -> str:
        """The probe's fingerprint, recorded so the run store can be checked.

        Duck-typed, like `val_top1_history`: the loop reads it when a head has
        one and records it in the summary, and `subset_check` compares it
        against the Stage 1 probe's at the same setting.

        It is the classifier this head actually predicts with, which for a run
        of the optional extension is the fine-tuned map and not the Stage 1
        probe's. A joint run that published the frozen digest would make the
        guard pass over precisely the case it exists to catch, so the digest
        moves with the classifier and `classifier_frozen` is what tells the
        guard to hold that run to a different rule.
        """
        return self._probe.classifier_digest

    @property
    def classifier_frozen(self) -> bool:
        """Whether the probe this head predicts with is the Stage 1 map, untouched.

        True for every graded Stage 3 row. False for a run of the optional
        extension (FR23), whose classifier was trained alongside the field.
        Duck-typed the same way the digest is.
        """
        return self._probe.classifier_frozen

    @property
    def zero_output_init(self) -> bool:
        return self._zero_output_init

    @property
    def loss_history(self) -> list[float]:
        """Per-step training loss, written to loss_curve.csv by the loop (ADR-024)."""
        return list(self._loss_history)

    @property
    def val_top1_history(self) -> list[tuple[int, float]]:
        """(step, val top-1) on the evaluation grid; empty when eval_every is 0."""
        return self._selector.history if self._selector is not None else []

    @property
    def best_epoch(self) -> int | None:
        """The training step whose field was kept, None when selection is off.

        The field's step, not the probe's epoch. Two checkpoints are selected
        in one Stage 3 run and the one the summary carries is the one Stage 3
        varies; the probe's own selection is Stage 1's and is identical across
        every head at this setting.
        """
        if self._field is None:
            raise NotFittedError("best_epoch is undefined before fit")
        return self._selector.best_step if self._selector is not None else None

    def fit(self, train_x: Tensor, train_y: Tensor, val_x: Tensor, val_y: Tensor) -> None:
        self._probe.fit(train_x, train_y, val_x, val_y)
        # The classifier the selector scores with is the head's own frozen
        # probe, so the number selection maximizes is the number the run
        # reports (ADR-020).
        selector = ValidationSelector(
            val_x,
            val_y,
            self._probe,
            sample_steps=self._sample_steps,
            eval_every=self._eval_every,
        )
        self._field, self._loss_history = self._fit_field(
            train_x, train_y, self._probe, selector
        )
        self._selector = selector

    def transport(self, query_x: Tensor) -> Tensor:
        """Where the query lands after T Euler steps. The P figures read this."""
        return self._detached(query_x, transport)

    def trajectory(self, query_x: Tensor) -> Tensor:
        """Every intermediate state, [T + 1, M, D], for the trajectory figures."""
        return self._detached(query_x, transport_trajectory)

    def predict(self, query_x: Tensor) -> Tensor:
        return self._probe.predict(self.transport(query_x))

    def _detached(self, x: Tensor, integrate) -> Tensor:
        field = self.field
        was_training = field.training
        field.eval()
        try:
            with torch.no_grad():
                return integrate(field, x, sample_steps=self._sample_steps)
        finally:
            field.train(was_training)


def _probe_kwargs(probe_params: object) -> dict[str, object]:
    """Stage 1 hyperparameters for the inner probe, keyed as LinearProbeHead names them."""
    if not probe_params:
        return {}
    if not isinstance(probe_params, dict):
        raise TypeError(f"probe_params must be a mapping, got {type(probe_params).__name__}")
    allowed = {"lr", "weight_decay", "batch_size", "max_epochs"}
    unknown = sorted(set(probe_params) - allowed)
    if unknown:
        raise ValueError(
            f"unknown probe_params {unknown}; the inner probe takes {sorted(allowed)}. "
            "init_seed is the run's and is not settable here (ADR-031)."
        )
    return dict(probe_params)
