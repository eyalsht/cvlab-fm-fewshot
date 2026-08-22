"""Standard FM training, exactly the write-up's equations.

    t ~ U(0, 1),  z_t = (1 - t) z_i + t p_{y_i},  u_i = p_{y_i} - z_i
    L_FM = || v_theta(z_t, t) - u_i ||^2

Note what training does not read: the step count. Standard training supervises
the velocity at points on the ideal path and never solves the ODE, so one
trained field serves every T. T = 4 and T = 12 are that one field read at two
resolutions, and their gap measures the curvature of the field rather than any
difference in capacity (ADR-023). Rolled-out training is the opposite case and
lives in its own module.

The loop, and the inference rollout both schemes classify with, are in `base`.
`transport` and `transport_trajectory` are re-exported here because this is
where callers have always found them.
"""

import torch
from torch import Tensor

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.training.base import (
    train_field,
    transport,
    transport_trajectory,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP

__all__ = ["train_standard_field", "transport", "transport_trajectory"]


def train_standard_field(
    x0: Tensor,
    x1: Tensor,
    *,
    hidden_dims: tuple[int, ...] = (512, 512),
    time_conditioning: str = "scalar",
    n_train_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    init_seed: int = 0,
) -> tuple[VelocityMLP, list[float]]:
    """Fit v_theta simulation-free on the paired coupling (x0[i] -> x1[i])."""

    def batch_loss(
        field: VelocityMLP, batch_x0: Tensor, batch_x1: Tensor, generator: torch.Generator
    ) -> Tensor:
        t = torch.rand(batch_x0.shape[0], generator=generator)
        return cfm_loss(field, batch_x0, batch_x1, t)

    return train_field(
        x0,
        x1,
        batch_loss,
        hidden_dims=hidden_dims,
        time_conditioning=time_conditioning,
        n_train_steps=n_train_steps,
        batch_size=batch_size,
        lr=lr,
        init_seed=init_seed,
    )
