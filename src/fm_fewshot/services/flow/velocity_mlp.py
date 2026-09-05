"""Velocity network v_theta(z, t) per PRD_flow_matching_block.

A small MLP over the feature and the time, with SiLU hidden layers. Small
because the whole point of caching features is that the block operates on
vectors, not images.

Two time conditionings (ADR-019). The default is the write-up's: the scalar t
concatenated to the feature, two hidden layers of width 512. The alternative is
a sinusoidal embedding, which is what Phase 7 built, because one scalar among
hundreds of feature dimensions is easy for the network to ignore and a field
that ignores t still trains and still transports, so the failure would be
silent. His is what runs; ours is the ablation, and
test_velocity_mlp asserts a fitted scalar field is not time-invariant rather
than leaving the risk to argument.

`output_projector` is Stage 3's row-space control (stage3_alignment 3.3): a
fixed [D, D] projection applied to the velocity, so the block can only move a
feature in directions the frozen classifier can see. It belongs to the network
rather than to a training loss because a field projected only while training
would leave the row space the moment it was integrated at inference, and the
ablation would measure nothing. None everywhere but that ablation.

`zero_output_init` is Stage 3's near-identity initialization (ADR-030). Zeroing
the output layer gives v_theta(z, t) = 0 for every z and t, so T Euler steps
leave a feature where it started and the untrained Stage 3 system is exactly
the linear probe it sits in front of. The write-up asks for "close to
identity"; this makes it an equality a test can assert. It defaults to False,
which is bit-identically the Stage 2 initialization.
"""

import math

import torch
from torch import Tensor, nn

TIME_CONDITIONINGS = ("scalar", "sinusoidal")


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
        hidden_dims: tuple[int, ...] = (512, 512),
        time_conditioning: str = "scalar",
        time_embed_dim: int = 64,
        seed: int = 0,
        zero_output_init: bool = False,
        output_projector: Tensor | None = None,
    ) -> None:
        super().__init__()
        if time_conditioning not in TIME_CONDITIONINGS:
            raise ValueError(
                f"unknown time_conditioning {time_conditioning!r}; "
                f"expected one of {TIME_CONDITIONINGS}"
            )
        if time_embed_dim % 2 != 0:
            raise ValueError(f"time_embed_dim must be even, got {time_embed_dim}")
        self.dim = dim
        self.time_conditioning = time_conditioning
        self.time_embed_dim = time_embed_dim
        self.zero_output_init = zero_output_init

        generator = torch.Generator().manual_seed(seed)
        layers: list[nn.Module] = []
        width = dim + (1 if time_conditioning == "scalar" else time_embed_dim)
        for hidden in hidden_dims:
            linear = nn.Linear(width, hidden)
            _seeded_init(linear, generator)
            layers += [linear, nn.SiLU()]
            width = hidden
        out = nn.Linear(width, dim)
        # Drawn from the generator either way, then zeroed, so the flag changes
        # the output layer and not the random stream: a zeroed field's hidden
        # weights are the Stage 2 field's hidden weights at the same seed.
        _seeded_init(out, generator)
        if zero_output_init:
            with torch.no_grad():
                out.weight.zero_()
                out.bias.zero_()
        layers.append(out)
        self.net = nn.Sequential(*layers)
        # A buffer, not a parameter: it is a fixed property of the frozen
        # classifier, and being in the state dict is what makes a selected
        # checkpoint restore the same field it was scored as.
        self.register_buffer("output_projector", output_projector)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        time = t.to(x.dtype)
        if self.time_conditioning == "scalar":
            conditioning = time.view(-1, 1)
        else:
            conditioning = sinusoidal_time_embedding(time, self.time_embed_dim)
        velocity = self.net(torch.cat([x, conditioning], dim=1))
        if self.output_projector is None:
            return velocity
        return velocity @ self.output_projector


def _seeded_init(linear: nn.Linear, generator: torch.Generator) -> None:
    """Initialize from our own generator so the seed alone fixes the weights."""
    bound = 1.0 / math.sqrt(linear.in_features)
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound, generator=generator)
        linear.bias.uniform_(-bound, bound, generator=generator)
