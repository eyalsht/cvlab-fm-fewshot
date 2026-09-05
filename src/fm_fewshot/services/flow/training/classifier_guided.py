"""Strategy 2: classifier-guided targets, then a standard FM update (FR20, ADR-033).

Per batch, with the field in its current state and under no_grad:

    zhat  = psi_theta(z)                                  # T Euler steps
    g     = W^T (softmax(W zhat + b) - onehot(y))
    zhat <- zhat - eta * step(g)                          # target_steps times
    zhat' = constrain(zhat, z)

then one standard FM update on the coupling z -> zhat':

    t ~ U(0,1),  z_t = (1-t) z + t zhat',  L = || v_theta(z_t, t) - (zhat' - z) ||^2

The update is `cfm_loss`, the same function Stage 2's standard scheme calls.
What Stage 3 adds is that the coupling moves: `standard.py` knows its endpoints
before the first optimizer step, and here the target is rebuilt from the
classification gradient as the field changes. That is the whole difference, and
it is why this module exists instead of a new argument to that one.

Two properties worth stating because the design depends on them.

`g` is a combination of rows of W, so the target displacement lies in `row(W)`
exactly, whose rank is at most C. Strategy 2 therefore searches a subspace of
what Strategy 1 searches, by construction rather than by tuning, which is the
asymmetry `stage3_alignment.md` section 3.3 measures and controls for.

The target compounds. `zhat` comes from the field that is being trained toward
the previous target, so each recompute adds another eta step and nothing in the
write-up bounds the total. He anticipates this in the knobs he lists, "whether
the target updates should be normalized or otherwise constrained", and chooses
none of them, so the literal unconstrained step is the graded row and the
normalized step with an anchor cap runs beside it in `ablations/`. If the
literal version drifts, that is a measurement of his own open question.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.solver import VelocityField
from fm_fewshot.services.flow.training.base import (
    StepObserver,
    ValidationSelector,
    train_field,
    transport,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP

TARGET_STEPS = ("raw", "normalized")
_EPS = 1e-12


@dataclass(frozen=True)
class GuidedTargetConfig:
    """The knobs the write-up leaves open (ADR-033). Every default is the graded row.

    `raw` with `eta = 1`, one target step, recomputed every optimizer step is
    his step 6 read literally. `normalized` scales the step to the anchor's own
    norm and caps the total displacement at `rho ||z||`, which is the version
    that answers "should the target updates be normalized" with a number.
    """

    target_step: str = "raw"
    eta: float = 1.0
    target_steps: int = 1
    target_every: int = 1
    rho: float = 0.5

    def validate(self) -> "GuidedTargetConfig":
        if self.target_step not in TARGET_STEPS:
            raise ValueError(
                f"unknown target_step {self.target_step!r}; expected one of {TARGET_STEPS}"
            )
        if self.target_steps < 1:
            raise ValueError(f"target_steps must be >= 1, got {self.target_steps}")
        if self.target_every < 1:
            raise ValueError(f"target_every must be >= 1, got {self.target_every}")
        if self.rho <= 0:
            raise ValueError(f"rho must be > 0, got {self.rho}")
        return self


def classification_gradient(zhat: Tensor, y: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """grad_zhat CE(W zhat + b, y), summed over the batch, in closed form.

    Written out rather than taken from autograd because the closed form is what
    makes the row-space claim readable: `softmax - onehot` is [N, C] and right
    multiplying by W lands every row in the span of W's rows.
    """
    residual = F.softmax(zhat @ weight.T + bias, dim=1)
    residual = residual - F.one_hot(y, num_classes=weight.shape[0]).to(residual.dtype)
    return residual @ weight


def guided_target(
    field: VelocityField,
    x0: Tensor,
    y: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    sample_steps: int,
    config: GuidedTargetConfig,
) -> Tensor:
    """The improved target zhat' for the coupling z -> zhat'.

    Detached throughout: the coupling is data for the flow matching update, not
    something the update optimizes through.
    """
    config.validate()
    with torch.no_grad():
        zhat = transport(field, x0, sample_steps=sample_steps)
        anchor = x0.norm(dim=1, keepdim=True)
        for _ in range(config.target_steps):
            gradient = classification_gradient(zhat, y, weight, bias)
            if config.target_step == "normalized":
                gradient = gradient / (gradient.norm(dim=1, keepdim=True) + _EPS) * anchor
            zhat = zhat - config.eta * gradient
        if config.target_step == "normalized":
            zhat = _cap(zhat, x0, anchor, config.rho)
        return zhat


def _cap(zhat: Tensor, x0: Tensor, anchor: Tensor, rho: float) -> Tensor:
    """Clip each row so || zhat' - z || <= rho || z ||, keeping its direction."""
    displacement = zhat - x0
    distance = displacement.norm(dim=1, keepdim=True)
    limit = rho * anchor
    scale = torch.where(distance > limit, limit / (distance + _EPS), torch.ones_like(distance))
    return x0 + displacement * scale


class ClassifierGuidedCoupler:
    """The moving coupling, as the `couple` hook the shared loop calls.

    Holds the whole training set because `target_every > 1` caches the target
    for every row and hands out slices between recomputes, which is the
    cheaper, staler alternative the write-up also allows. At `target_every = 1`,
    the graded setting, there is nothing to cache and only the batch is
    computed: the target for a row is built from the field as it stands at that
    step either way, so the two paths agree on every row they both produce.
    """

    def __init__(
        self,
        x0: Tensor,
        y: Tensor,
        weight: Tensor,
        bias: Tensor,
        *,
        sample_steps: int,
        config: GuidedTargetConfig,
    ) -> None:
        self.x0 = x0
        self.y = y
        self.weight = weight.detach()
        self.bias = bias.detach()
        self.sample_steps = sample_steps
        self.config = config.validate()
        self.recompute_steps: list[int] = []
        self._step = 0
        self._cache: Tensor | None = None

    def __call__(self, field: VelocityField, rows: Tensor, batch_x0: Tensor) -> Tensor:
        self._step += 1
        if self.config.target_every == 1:
            self.recompute_steps.append(self._step)
            return self._target(field, batch_x0, self.y[rows])
        if self._cache is None or (self._step - 1) % self.config.target_every == 0:
            self.recompute_steps.append(self._step)
            self._cache = self._target(field, self.x0, self.y)
        return self._cache[rows]

    def _target(self, field: VelocityField, x0: Tensor, y: Tensor) -> Tensor:
        return guided_target(
            field,
            x0,
            y,
            self.weight,
            self.bias,
            sample_steps=self.sample_steps,
            config=self.config,
        )


def train_classifier_guided_field(
    x0: Tensor,
    y: Tensor,
    weight: Tensor,
    bias: Tensor,
    selector: ValidationSelector | None = None,
    *,
    sample_steps: int,
    config: GuidedTargetConfig | None = None,
    hidden_dims: tuple[int, ...] = (512, 512),
    time_conditioning: str = "scalar",
    n_train_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    init_seed: int = 0,
    zero_output_init: bool = True,
    on_step: StepObserver | None = None,
) -> tuple[VelocityMLP, list[float]]:
    """Fit v_theta by standard FM on a coupling rebuilt from the frozen probe's gradient."""
    if sample_steps < 1:
        raise ValueError(f"sample_steps must be >= 1, got {sample_steps}")
    coupler = ClassifierGuidedCoupler(
        x0,
        y,
        weight,
        bias,
        sample_steps=sample_steps,
        config=(config or GuidedTargetConfig()),
    )

    def batch_loss(
        field: VelocityMLP, batch_x0: Tensor, batch_x1: Tensor, generator: torch.Generator
    ) -> Tensor:
        # The same simulation-free update Stage 2 runs, on a coupling that moved.
        t = torch.rand(batch_x0.shape[0], generator=generator)
        return cfm_loss(field, batch_x0, batch_x1, t)

    return train_field(
        x0,
        y,
        batch_loss,
        selector,
        couple=coupler,
        hidden_dims=hidden_dims,
        time_conditioning=time_conditioning,
        n_train_steps=n_train_steps,
        batch_size=batch_size,
        lr=lr,
        init_seed=init_seed,
        zero_output_init=zero_output_init,
        on_step=on_step,
    )
