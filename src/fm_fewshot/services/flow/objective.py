"""Conditional flow matching loss per ADR-005.

    L_CFM = E_{t, (x0, x1)} || v_theta(x_t, t) - (x1 - x0) ||^2

Simulation-free: no ODE is solved during training, which is exactly what
distinguishes standard FM training from the rolled-out scheme.
"""

from collections.abc import Callable

import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from fm_fewshot.services.flow.path import conditional_target, interpolate


def cfm_loss(
    field: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
) -> Tensor:
    """Mean squared error between the predicted and conditional velocities."""
    xt = interpolate(x0, x1, t)
    return F.mse_loss(field(xt, t), conditional_target(x0, x1))
