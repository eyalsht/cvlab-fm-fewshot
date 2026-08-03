"""Fixed-step ODE solvers per ADR-006.

Sampling a flow means integrating dx/dt = v(x, t) from t=0 to t=1 with a fixed
step h = 1/N. Step count is a first-class experiment axis (goal G5), so the
solver never chooses it: adaptive methods would pick N internally and blind the
curve the project is trying to measure.

No no_grad anywhere. The same code path serves inference, where the caller
detaches, and rolled-out training, where gradients flow through all N steps.
"""

from collections.abc import Callable

import torch
from torch import Tensor

VelocityField = Callable[[Tensor, Tensor], Tensor]
METHODS = ("euler", "midpoint")


def solve_ode(
    field: VelocityField,
    x0: Tensor,
    *,
    n_steps: int = 8,
    method: str = "euler",
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Integrate dx/dt = field(x, t) over [0, 1] in n_steps uniform steps."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    step = 1.0 / n_steps
    x = x0
    states = [x0] if return_trajectory else None

    for k in range(n_steps):
        t = torch.full((x.shape[0],), k * step, dtype=x.dtype, device=x.device)
        if method == "euler":
            x = x + step * field(x, t)
        else:
            half = step / 2.0
            k1 = field(x, t)
            x = x + step * field(x + half * k1, t + half)

        if not torch.isfinite(x).all():
            raise ValueError(
                f"non-finite state after step {k} of {n_steps} ({method}); "
                "the field diverged"
            )
        if states is not None:
            states.append(x)

    if states is not None:
        return x, torch.stack(states)
    return x


def straightness(trajectory: Tensor) -> Tensor:
    """Mean deviation from the straight chord, normalized by chord length.

    Zero for a perfectly straight path. Used by reflow to show that
    straightening happened (PRD_rectified_flow); lives here because it is a
    property of a solver trajectory.
    """
    start, end = trajectory[0], trajectory[-1]
    chord = end - start
    length = chord.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    direction = chord / length

    deviations = []
    for k in range(1, trajectory.shape[0] - 1):
        offset = trajectory[k] - start
        projected = (offset * direction).sum(dim=-1, keepdim=True) * direction
        deviations.append((offset - projected).norm(dim=-1) / length.squeeze(-1))
    if not deviations:
        return torch.zeros(trajectory.shape[1], device=trajectory.device)
    return torch.stack(deviations).mean(dim=0)
