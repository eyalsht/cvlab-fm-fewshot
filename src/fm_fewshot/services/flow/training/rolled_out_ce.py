"""Strategy 1: end-to-end rolled-out classification training (FR19, ADR-034).

    zhat_0 = z,  zhat_{k+1} = zhat_k + (1/T) v_theta(zhat_k, k/T),  k = 0..T-1
    L_cls = CE(W zhat_T + b, y)

The gradient goes back through all T velocity predictions. W and b are the
Stage 1 probe's and are detached, so only theta moves.

This is not `rolled_out.py` with a different target. That module minimizes
|| zhat_T - p_y ||^2 toward a prototype fixed before training; here there is no
prototype and no point in feature space to reach, only a decision to get right.
An endpoint distance and a cross-entropy differ in what they hold the block
responsible for: the first names where to land, the second names only which
side of the frozen map's boundaries to land on. That is the whole difference
between Stage 2's second scheme and this one, and it is why Stage 3 writes a
new loss instead of reusing that one (stage3_alignment section 1.1).

The optional penalties are the two the write-up names, and both carry zero
weight in `results/` (ADR-034):

    L = L_cls + lambda_disp * mean_i || zhat_T - z_i ||^2
              + lambda_vel  * mean_i sum_k || v_theta(zhat_k, k/T) ||^2

`project_velocity` is ours, not his. The classifier is frozen and linear, so
the logits depend on zhat only through its projection onto row(W) and the field
is free in all D dimensions while the loss can see at most C of them. Projecting
the velocity puts Strategy 1 in Strategy 2's search space while keeping its
objective, which is the only way to tell an objective difference from a search
space difference in the comparison the write-up asks for
(stage3_alignment section 3.3). It is an `ablations/` row and never a graded one.

There is one integrator in this repository and this module uses it. The
velocities the penalty needs are collected by recording what the solver asks
for, not by writing a second rollout that could drift from the first.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fm_fewshot.services.flow.solver import VelocityField
from fm_fewshot.services.flow.training.base import (
    StepObserver,
    ValidationSelector,
    train_field,
    transport,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP


@dataclass(frozen=True)
class RolledOutCeConfig:
    """The knobs past the shared training block. Every default is the graded row.

    The write-up's Strategy 1 is the plain cross-entropy, so the penalties are
    off and the velocity is unprojected unless an ablation config says
    otherwise.
    """

    lambda_disp: float = 0.0
    lambda_vel: float = 0.0
    project_velocity: bool = False


def row_space_projector(weight: Tensor) -> Tensor:
    """The orthogonal projector onto row(W), [D, D].

    Built from an orthonormal basis of the row space rather than from
    W^T (W W^T)^-1 W, which is the same matrix but needs W to have full row
    rank. A probe fitted at K=5 on 47 classes can be rank deficient, and a
    singular inverse there would be a crash in an ablation rather than a
    projector onto the space that actually exists.
    """
    _, singular, right = torch.linalg.svd(weight, full_matrices=False)
    tolerance = singular.max() * max(weight.shape) * torch.finfo(weight.dtype).eps
    basis = right[singular > tolerance]  # [rank, D], orthonormal rows
    return basis.T @ basis


class _Recorder(torch.nn.Module):
    """The field, plus the velocities the solver asked it for.

    Wrapping the field rather than reimplementing the rollout keeps the
    training path and the inference path the same integrator, which is what
    makes "trained through the rollout it is scored on" true rather than
    intended. Gradients pass through untouched; the recorded tensors are the
    same objects the solver used.
    """

    def __init__(self, field: VelocityField) -> None:
        super().__init__()
        self.field = field
        self.velocities: list[Tensor] = []

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        velocity = self.field(x, t)
        self.velocities.append(velocity)
        return velocity


def rolled_out_ce_loss(
    field: VelocityField,
    x0: Tensor,
    y: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    sample_steps: int,
    lambda_disp: float = 0.0,
    lambda_vel: float = 0.0,
) -> Tensor:
    """L_cls for one batch, with the two optional penalties.

    `weight` and `bias` are detached here as well as frozen by the head, so a
    caller that hands over tensors still attached to a graph cannot train the
    classifier through this loss by accident.
    """
    recorder = _Recorder(field)
    endpoint = transport(recorder, x0, sample_steps=sample_steps)
    loss = F.cross_entropy(endpoint @ weight.detach().T + bias.detach(), y)
    if lambda_disp:
        loss = loss + lambda_disp * ((endpoint - x0) ** 2).sum(dim=1).mean()
    if lambda_vel:
        velocity = torch.stack([(v**2).sum(dim=1) for v in recorder.velocities])
        loss = loss + lambda_vel * velocity.sum(dim=0).mean()
    return loss


def train_rolled_out_ce_field(
    x0: Tensor,
    y: Tensor,
    weight: Tensor,
    bias: Tensor,
    selector: ValidationSelector | None = None,
    *,
    sample_steps: int,
    hidden_dims: tuple[int, ...] = (512, 512),
    time_conditioning: str = "scalar",
    n_train_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_disp: float = 0.0,
    lambda_vel: float = 0.0,
    project_velocity: bool = False,
    init_seed: int = 0,
    zero_output_init: bool = True,
    on_step: StepObserver | None = None,
) -> tuple[VelocityMLP, list[float]]:
    """Fit v_theta by rolling the solver out and scoring the frozen probe's decision."""
    if sample_steps < 1:
        raise ValueError(f"sample_steps must be >= 1, got {sample_steps}")
    # One projector for the whole fit, built into the field itself: W is
    # frozen, so this is one SVD, and a field that projects only inside the
    # loss would stop projecting the moment inference integrated it.
    projector = row_space_projector(weight) if project_velocity else None

    def batch_loss(
        field: VelocityMLP, batch_x0: Tensor, batch_y: Tensor, generator: torch.Generator
    ) -> Tensor:
        # generator unused: this objective has no random time (ADR-021 carries
        # over; the rollout is the whole path and nothing is sampled along it).
        return rolled_out_ce_loss(
            field,
            batch_x0,
            batch_y,
            weight,
            bias,
            sample_steps=sample_steps,
            lambda_disp=lambda_disp,
            lambda_vel=lambda_vel,
        )

    return train_field(
        x0,
        y,
        batch_loss,
        selector,
        hidden_dims=hidden_dims,
        time_conditioning=time_conditioning,
        n_train_steps=n_train_steps,
        batch_size=batch_size,
        lr=lr,
        init_seed=init_seed,
        zero_output_init=zero_output_init,
        output_projector=projector,
        on_step=on_step,
    )
