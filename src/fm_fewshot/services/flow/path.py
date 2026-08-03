"""Linear conditional optimal-transport path per ADR-005.

    x_t = (1 - t) x0 + t x1,        dx_t/dt = x1 - x0

The straightest per-sample path there is, and its conditional target does not
depend on t. Regressing these per-pair velocities is gradient-equivalent to
regressing the intractable marginal field (Lipman et al.), which is what makes
simulation-free training work.
"""

import torch
from torch import Tensor


def interpolate(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Point on the linear path at time t, one t per row."""
    if bool(((t < 0.0) | (t > 1.0)).any()):
        raise ValueError("t must lie in [0, 1]")
    t_col = t.view(-1, *([1] * (x0.ndim - 1)))
    return (1.0 - t_col) * x0 + t_col * x1


def conditional_target(x0: Tensor, x1: Tensor) -> Tensor:
    """dx_t/dt along the linear path. Constant in t, by construction."""
    return x1 - x0


def sample_times(batch: int, generator: torch.Generator | None = None) -> Tensor:
    """t ~ U[0, 1], one per pair."""
    return torch.rand(batch, generator=generator)
