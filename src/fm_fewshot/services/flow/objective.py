"""Conditional flow matching loss per ADR-005.

    L_CFM = E_{t, (x0, x1)} || v_theta(x_t, t) - (x1 - x0) ||^2

Simulation-free: no ODE is solved during training, which is exactly what
distinguishes standard FM training from the rolled-out scheme.

The norm is summed over the feature dimension and only the batch is averaged,
which is what `|| . ||^2` means and what `rolled_out_loss` already did. Until
the fairness guards this was `F.mse_loss`, a per-element mean, so the two
schemes' losses differed in scale by a factor of D and neither their learning
rates nor their loss curves could be read against each other.
"""

from collections.abc import Callable

from torch import Tensor

from fm_fewshot.services.flow.path import conditional_target, interpolate


def cfm_loss(
    field: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
) -> Tensor:
    """Mean over the batch of the squared L2 error in the predicted velocity."""
    xt = interpolate(x0, x1, t)
    return ((field(xt, t) - conditional_target(x0, x1)) ** 2).sum(dim=1).mean()
