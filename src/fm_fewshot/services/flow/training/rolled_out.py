"""Rolled-out training, the write-up's second scheme.

Starting from each training feature zhat_0 = z_i, apply the same full T-step
sequence used at inference and score only where it lands:

    zhat_{k+1} = zhat_k + (1 / T) v_theta(zhat_k, k / T),   k = 0 .. T - 1
    L_roll = mean_i || zhat_T - p_{y_i} ||^2

The gradient goes back through all T velocity predictions. That is the whole
objective. There is no t sampling and no interpolation anywhere in this module,
and no cross-entropy, temperature or CFM term either: the earlier
`PRD_rolled_out_training` design had all three and the write-up has none of
them (ADR-021).

The norm is summed over the feature dimension and only the batch is averaged,
which is what `|| . ||^2` means. `cfm_loss` reduces the same way, so the two
schemes' losses are on one scale, their learning rates mean the same thing and
their curves can be plotted on one axis.

Training depth is inference depth: `sample_steps` is the only step count and
T = 4 and T = 12 are two separately trained models (ADR-022).

Depth costs activation memory, since every intermediate state is held for the
backward pass. `gradient_checkpointing` trades it back for a second forward
pass through the network at each step, recomputing rather than approximating,
so the loss and the gradient are bit-identical either way. It wraps the field
rather than the solver: there is one integrator in this repository and
rolled-out training uses the same one inference does.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from fm_fewshot.services.flow.solver import VelocityField
from fm_fewshot.services.flow.training.base import (
    StepObserver,
    ValidationSelector,
    train_field,
    transport,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP


def rolled_out_loss(
    field: VelocityField,
    x0: Tensor,
    targets: Tensor,
    *,
    sample_steps: int,
    gradient_checkpointing: bool = False,
) -> Tensor:
    """L_roll for one batch: mean squared distance from zhat_T to the prototypes.

    `targets` is detached. The prototypes are fixed points of the space, given
    by Stage 1's closed-form rule, and nothing in Stage 2 fits them.
    """
    endpoint = transport(
        _checkpointed(field) if gradient_checkpointing else field,
        x0,
        sample_steps=sample_steps,
    )
    return ((endpoint - targets.detach()) ** 2).sum(dim=1).mean()


def train_rolled_out_field(
    x0: Tensor,
    x1: Tensor,
    selector: ValidationSelector | None = None,
    *,
    sample_steps: int,
    hidden_dims: tuple[int, ...] = (512, 512),
    time_conditioning: str = "scalar",
    n_train_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    gradient_checkpointing: bool = False,
    init_seed: int = 0,
    on_step: StepObserver | None = None,
) -> tuple[VelocityMLP, list[float]]:
    """Fit v_theta by rolling the solver out over the paired coupling (x0[i] -> x1[i])."""
    if sample_steps < 1:
        raise ValueError(f"sample_steps must be >= 1, got {sample_steps}")

    def batch_loss(
        field: VelocityMLP, batch_x0: Tensor, batch_x1: Tensor, generator: torch.Generator
    ) -> Tensor:
        # generator unused: the rolled-out objective has no random time.
        return rolled_out_loss(
            field,
            batch_x0,
            batch_x1,
            sample_steps=sample_steps,
            gradient_checkpointing=gradient_checkpointing,
        )

    return train_field(
        x0,
        x1,
        batch_loss,
        selector,
        hidden_dims=hidden_dims,
        time_conditioning=time_conditioning,
        n_train_steps=n_train_steps,
        batch_size=batch_size,
        lr=lr,
        init_seed=init_seed,
        on_step=on_step,
    )


def _checkpointed(field: VelocityField) -> Callable[[Tensor, Tensor], Tensor]:
    """The same field, with its internals recomputed in the backward pass."""

    def evaluate(x: Tensor, t: Tensor) -> Tensor:
        return checkpoint(field, x, t, use_reentrant=False)

    return evaluate
