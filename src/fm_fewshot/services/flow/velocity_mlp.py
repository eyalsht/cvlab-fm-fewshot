"""Velocity network v_theta(x, t) per PRD_flow_matching_block.

A small MLP over concat(x, time_embed(t)) with SiLU hidden layers. Small
because the whole point of caching features is that the block operates on
vectors, not images.

Time enters through a sinusoidal embedding rather than as a raw scalar: one
scalar among hundreds of feature dimensions is easy for the network to ignore,
and a field that ignores t still trains and still transports, so the failure
would be silent.
"""

import math

import torch
from torch import Tensor, nn


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Transformer-style embedding of t in [0, 1]; first half sin, second cos."""
    if dim % 2 != 0:
        raise ValueError(f"time_embed_dim must be even, got {dim}")
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=t.dtype, device=t.device) * (-math.log(10000.0) / half)
    )
    angles = t.view(-1, 1) * freqs.view(1, -1) * (2.0 * math.pi)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class VelocityMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        time_embed_dim: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if time_embed_dim % 2 != 0:
            raise ValueError(f"time_embed_dim must be even, got {time_embed_dim}")
        self.dim = dim
        self.time_embed_dim = time_embed_dim

        generator = torch.Generator().manual_seed(seed)
        layers: list[nn.Module] = []
        width = dim + time_embed_dim
        for hidden in hidden_dims:
            linear = nn.Linear(width, hidden)
            _seeded_init(linear, generator)
            layers += [linear, nn.SiLU()]
            width = hidden
        out = nn.Linear(width, dim)
        _seeded_init(out, generator)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        embedded = sinusoidal_time_embedding(t.to(x.dtype), self.time_embed_dim)
        return self.net(torch.cat([x, embedded], dim=1))


def _seeded_init(linear: nn.Linear, generator: torch.Generator) -> None:
    """Initialize from our own generator so the seed alone fixes the weights."""
    bound = 1.0 / math.sqrt(linear.in_features)
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound, generator=generator)
        linear.bias.uniform_(-bound, bound, generator=generator)
