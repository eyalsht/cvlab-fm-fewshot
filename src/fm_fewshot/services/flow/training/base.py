"""What both training schemes share: the fitting loop and the inference rollout.

The write-up asks us to keep the network architecture and the main training
choices fixed when comparing standard and rolled-out training, so the two
schemes must not own two copies of the loop that could drift apart. They own a
loss and nothing else. Everything around it, the seeded field, the Adam
optimizer, the batch draw and the loss curve, lives here and is called by both.

The per-batch loss is handed the batch generator rather than a pre-drawn time.
Standard training needs t ~ U(0, 1) and draws it from that stream; rolled-out
training has no t anywhere in its objective (ADR-021) and draws nothing, which
is the honest way to say so in code. The row stream is the same call either
way, so the schemes agree on the first batch and the comparison starts fair.

The optimizer is ours, not his: he asks only for stable training, so Adam at
1e-3 carries over from the Phase 7 toy and every setting of it is a config
field recorded in the run.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from fm_fewshot.services.flow.solver import solve_ode
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP

BatchLoss = Callable[[VelocityMLP, Tensor, Tensor, torch.Generator], Tensor]


def train_field(
    x0: Tensor,
    x1: Tensor,
    batch_loss: BatchLoss,
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
        loss = batch_loss(field, x0[rows], x1[rows], generator)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite training loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    return field, history


def transport(field, x: Tensor, *, sample_steps: int) -> Tensor:
    """T Euler steps from x, the write-up's inference rule.

    Both schemes classify this way; rolled-out training also trains through it,
    which is why there is one integrator and not two.
    """
    return solve_ode(field, x, n_steps=sample_steps, method="euler")


def transport_trajectory(field, x: Tensor, *, sample_steps: int) -> Tensor:
    """The same integration, keeping every state: [T + 1, N, D] for the S4 figures."""
    _, states = solve_ode(field, x, n_steps=sample_steps, method="euler", return_trajectory=True)
    return states
