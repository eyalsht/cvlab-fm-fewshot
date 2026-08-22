"""Standard FM training and the write-up's inference rule.

Training is simulation-free, exactly the write-up's equations:

    t ~ U(0, 1),  z_t = (1 - t) z_i + t p_{y_i},  u_i = p_{y_i} - z_i
    L_FM = || v_theta(z_t, t) - u_i ||^2

Inference is T Euler steps from the test feature:

    zhat_{k+1} = zhat_k + (1 / T) v_theta(zhat_k, k / T),   k = 0 .. T - 1

Note what training does not read: the step count. Standard training supervises
the velocity at points on the ideal path and never solves the ODE, so one
trained field serves every T. T = 4 and T = 12 are that one field read at two
resolutions, and their gap measures the curvature of the field rather than any
difference in capacity (ADR-023). Rolled-out training is the opposite case and
lives in its own module.

The optimizer is ours, not his: he asks only for stable training, so Adam at
1e-3 carries over from the Phase 7 toy and every setting of it is a config
field recorded in the run.
"""

import torch
from torch import Tensor

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.solver import solve_ode
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP


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
    """Fit v_theta on the paired coupling (x0[i] -> x1[i]); return it and its loss curve."""
    if x0.shape != x1.shape:
        raise ValueError(
            "paired endpoints must match in shape, got "
            f"{tuple(x0.shape)} and {tuple(x1.shape)}"
        )
    if n_train_steps < 0:
        raise ValueError(f"n_train_steps must be >= 0, got {n_train_steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    field = VelocityMLP(
        dim=x0.shape[1],
        hidden_dims=tuple(hidden_dims),
        time_conditioning=time_conditioning,
        seed=init_seed,
    )
    if n_train_steps == 0:
        return field, []

    # One stream for the weights, one for the batches, both from init_seed, so
    # the seed alone reproduces the fit (ADR-012).
    generator = torch.Generator().manual_seed(init_seed)
    optimizer = torch.optim.Adam(field.parameters(), lr=lr)
    history: list[float] = []

    for step in range(n_train_steps):
        rows = torch.randint(0, x0.shape[0], (batch_size,), generator=generator)
        t = torch.rand(batch_size, generator=generator)
        loss = cfm_loss(field, x0[rows], x1[rows], t)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite CFM loss at training step {step}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    return field, history


def transport(field, x: Tensor, *, sample_steps: int) -> Tensor:
    """T Euler steps from x, the write-up's inference rule."""
    return solve_ode(field, x, n_steps=sample_steps, method="euler")


def transport_trajectory(field, x: Tensor, *, sample_steps: int) -> Tensor:
    """The same integration, keeping every state: [T + 1, N, D] for the S4 figures."""
    _, states = solve_ode(field, x, n_steps=sample_steps, method="euler", return_trajectory=True)
    return states
