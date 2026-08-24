"""Backward integration of the trained field, per PRD_reverse_flow.

The write-up's optional section: explore the learned flow in the reverse
direction, starting from the class prototypes, and compare samples and
prototypes at intermediate flow times. ADR-026 fixes what "reverse" means
numerically, backward ODE integration with the solver this repository already
has, and ADR-027 fixes what it is for, a diagnostic and a geometry instrument
and never a classifier.

There is one integrator in this repository. Every mode here reaches it by
reparameterizing the segment it wants onto [0, 1] and calling `solve_ode`, so
the reverse leg cannot drift from the forward leg the heads run.

The grid is the thing to get right. Forward evaluates t at 0, 1/T, ..,
(T-1)/T; reverse evaluates it at 1, (T-1)/T, .., 1/T. Neither holds the
other's missing endpoint, and swapping them moves every number without
raising anything.

Everything below reads true test labels and none of it may reach a head
configuration or `TABLE.md` (ADR-027). That is why this module is imported by
the `reverse` command and by nothing on the head or evaluation path.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from fm_fewshot.services.flow.path import interpolate
from fm_fewshot.services.flow.solver import METHODS, solve_ode

REVERSE_MODES = ("cycle", "basin", "meet", "volume")

VelocityField = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class ReverseConfig:
    """The PRD section 3 table. Validated on construction, so a bad request
    fails before any field is refitted."""

    mode: str = "cycle"
    reverse_steps: tuple[int, ...] = (4, 12, 32)
    reverse_method: tuple[str, ...] = ("euler", "midpoint")
    sigma_scale: float = 1.0
    n_samples: int = 128
    meet_times: tuple[float, ...] = (0.25, 0.5, 0.75)
    n_hutchinson: int = 1

    def __post_init__(self) -> None:
        if self.mode not in REVERSE_MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {REVERSE_MODES}")
        if not self.reverse_steps or any(t < 1 for t in self.reverse_steps):
            raise ValueError(f"reverse_steps must all be >= 1, got {self.reverse_steps}")
        if not self.reverse_method or any(m not in METHODS for m in self.reverse_method):
            raise ValueError(
                f"reverse_method must be drawn from {METHODS}, got {self.reverse_method}"
            )
        if self.sigma_scale < 0.0:
            raise ValueError(f"sigma_scale must be >= 0, got {self.sigma_scale}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if not self.meet_times or any(not 0.0 < t < 1.0 for t in self.meet_times):
            raise ValueError(
                f"meet_times must lie strictly inside (0, 1), got {self.meet_times}"
            )
        # The refusal the PRD asks for by name: volume with nothing to
        # estimate the trace with would return zeros that look like a result.
        if self.mode == "volume" and self.n_hutchinson < 1:
            raise ValueError(
                f"mode 'volume' needs n_hutchinson >= 1, got {self.n_hutchinson}"
            )


def integrate_segment(
    field: VelocityField,
    x: Tensor,
    *,
    t_start: float,
    t_end: float,
    n_steps: int,
    method: str = "euler",
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Integrate dx/dt = field(x, t) from t_start to t_end in n_steps.

    Reparameterizes onto s in [0, 1] by t = t_start + (t_end - t_start) s and
    scales the field by the span, then hands the result to the one solver. For
    t_start = 1, t_end = 0 the span is -1, so the step becomes
    z - (1/T) v(z, t) and the solver's own grid s = 0, 1/T, .., (T-1)/T maps to
    t = 1, (T-1)/T, .., 1/T, which is the reverse grid exactly.
    """
    if not torch.isfinite(x).all():
        raise ValueError("the start tensor must be finite before integration begins")
    span = t_end - t_start

    def segment(state: Tensor, s: Tensor) -> Tensor:
        return span * field(state, t_start + span * s)

    return solve_ode(
        segment, x, n_steps=n_steps, method=method, return_trajectory=return_trajectory
    )


def reverse_transport(
    field: VelocityField,
    x: Tensor,
    *,
    n_steps: int,
    method: str = "euler",
    t_start: float = 1.0,
    t_end: float = 0.0,
) -> Tensor:
    """The backward leg: from t_start down to t_end. `cycle` and `basin` use it."""
    return integrate_segment(
        field, x, t_start=t_start, t_end=t_end, n_steps=n_steps, method=method
    )


def cycle_relative_error(
    field: VelocityField,
    x: Tensor,
    *,
    forward_steps: int,
    reverse_steps: int,
    forward_method: str = "euler",
    reverse_method: str = "euler",
) -> Tensor:
    """||z~ - z|| / ||z|| per row, for z~ the round trip of z.

    In the continuous limit with an exactly invertible flow this is zero, so
    what is measured is contraction plus reverse-solver error. ADR-026
    separates them by behaviour: the solver term falls with T and contraction
    does not, which is why the caller sweeps T rather than reading one number.

    The default forward leg is Euler because that is the map the head actually
    applies (`z_T = psi(z)`); the control tests drive both legs with the same
    method so that the measured error is purely the solver's.
    """
    # No gradient anywhere below: the solver deliberately keeps its graph for
    # rolled-out training, and none of these diagnostics is ever trained
    # through, so paying for the graph would only cost memory on real features.
    with torch.no_grad():
        forward = integrate_segment(
            field, x, t_start=0.0, t_end=1.0, n_steps=forward_steps, method=forward_method
        )
        back = integrate_segment(
            field, forward, t_start=1.0, t_end=0.0, n_steps=reverse_steps,
            method=reverse_method,
        )
        return (back - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-12)


def median_iqr(values: Tensor) -> dict[str, float | int | None]:
    """Median and interquartile range, the PRD's reporting rule.

    Not mean and std: cycle error is heavy-tailed, so a handful of points the
    flow squeezed hard would set the mean and hide where the bulk sits.
    """
    flat = values.detach().reshape(-1).to(torch.float64)
    n = int(flat.numel())
    if n == 0:
        return {"median": None, "q1": None, "q3": None, "iqr": None, "n": 0}
    q1, median, q3 = (
        float(torch.quantile(flat, q)) for q in (0.25, 0.5, 0.75)
    )
    return {"median": median, "q1": q1, "q3": q3, "iqr": q3 - q1, "n": n}


def residual_sigma(transported: Tensor, targets: Tensor, sigma_scale: float) -> float:
    """sigma_scale times the median of ||psi(z_i) - p_{y_i}|| over the subset.

    sigma is derived, never a free constant: the perturbation has to match the
    residual spread the forward flow actually leaves, or the clouds stop being
    comparable across cells, encoders and schemes. sigma_scale = 0 degenerates
    to the prototype pre-image, one point per class.
    """
    if sigma_scale < 0.0:
        raise ValueError(f"sigma_scale must be >= 0, got {sigma_scale}")
    residual = (transported - targets).detach().norm(dim=-1)
    return sigma_scale * float(residual.median())


def basin_seed(seed: int, class_id: int) -> int:
    """Derived from (config.seed, "reverse", class_id).

    Per class rather than per call, so a cloud is reproducible on its own and
    does not depend on how many classes were drawn before it. blake2b rather
    than hash(), which is salted per process.
    """
    digest = hashlib.blake2b(f"{seed}:reverse:{class_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def basin_samples(
    field: VelocityField,
    prototypes: Tensor,
    *,
    sigma: float,
    n_samples: int,
    n_steps: int,
    seed: int,
    method: str = "euler",
    class_ids: Sequence[int] | None = None,
) -> Tensor:
    """Reverse clouds from p_c + sigma eps, one [n_samples, D] block per row.

    The result is the set of features that would have flowed to prototype c:
    the head's decision region, drawn in feature space rather than argued for.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    ids = tuple(range(prototypes.shape[0])) if class_ids is None else tuple(class_ids)
    if len(ids) != prototypes.shape[0]:
        raise ValueError(
            f"class_ids has {len(ids)} entries for {prototypes.shape[0]} prototypes"
        )

    clouds = []
    with torch.no_grad():
        for row, class_id in enumerate(ids):
            generator = torch.Generator().manual_seed(basin_seed(seed, class_id))
            noise = torch.randn(
                n_samples, prototypes.shape[1], generator=generator, dtype=prototypes.dtype
            )
            start = prototypes[row].unsqueeze(0) + sigma * noise
            clouds.append(reverse_transport(field, start, n_steps=n_steps, method=method))
    return torch.stack(clouds)


def basin_alignment(
    clouds: Tensor, test_x: Tensor, test_y: Tensor, n_classes: int
) -> Tensor:
    """Row-normalized C x C matrix of mean cosine, cloud c against class c'.

    Diagonal dominance is the claim, and the matrix plots like the Stage 1
    confusion matrix. Negative mean cosines carry no "belongs to" mass, so they
    are clamped away before the row is normalized; a row with nothing positive
    in it stays zero rather than being manufactured into a distribution.
    """
    matrix = torch.zeros(clouds.shape[0], n_classes, dtype=torch.float32)
    normalized_clouds = torch.nn.functional.normalize(clouds, dim=-1)
    for target in range(n_classes):
        points = test_x[test_y == target]
        if points.shape[0] == 0:
            continue
        normalized_points = torch.nn.functional.normalize(points, dim=-1)
        for row in range(clouds.shape[0]):
            matrix[row, target] = (normalized_clouds[row] @ normalized_points.T).mean()
    positive = matrix.clamp_min(0.0)
    return positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-12)


def meet_gaps(
    field: VelocityField,
    x0: Tensor,
    x1: Tensor,
    *,
    t_star: float,
    n_steps: int,
    method: str = "euler",
) -> dict[str, dict[str, float | int | None]]:
    """Where the two legs meet at t*, against the true interpolant.

    The three coincide only if the learned field reproduces the conditional
    path. The gaps say where along the path the marginal averaging departs
    from the conditional targets, which neither endpoint alone can show. Each
    is normalized by ||x1 - x0|| so pairs of different lengths compare.
    """
    if not 0.0 < t_star < 1.0:
        raise ValueError(f"t_star must lie strictly inside (0, 1), got {t_star}")
    with torch.no_grad():
        forward = integrate_segment(
            field, x0, t_start=0.0, t_end=t_star, n_steps=n_steps, method=method
        )
        backward = integrate_segment(
            field, x1, t_start=1.0, t_end=t_star, n_steps=n_steps, method=method
        )
    times = torch.full((x0.shape[0],), t_star, dtype=x0.dtype, device=x0.device)
    target = interpolate(x0, x1, times)
    chord = (x1 - x0).norm(dim=-1).clamp_min(1e-12)
    return {
        "gap_forward": median_iqr((forward - target).norm(dim=-1) / chord),
        "gap_backward": median_iqr((backward - target).norm(dim=-1) / chord),
        "gap_between": median_iqr((forward - backward).norm(dim=-1) / chord),
    }


def log_volume_change(
    field: VelocityField,
    x: Tensor,
    *,
    n_steps: int,
    n_hutchinson: int,
    seed: int,
    method: str = "euler",
) -> Tensor:
    """Per-point log volume change along the forward trajectory (stretch mode).

        d/dt log|det J| = div v_theta

    estimated by Hutchinson with one vector-Jacobian product per probe per
    step. This is the continuous form of what the cycle error measures with a
    round trip: a scalar per point for how much its neighbourhood was squeezed.
    Rademacher probes, so the estimator is exact on a linear field.
    """
    if n_hutchinson < 1:
        raise ValueError(f"n_hutchinson must be >= 1, got {n_hutchinson}")
    _, trajectory = solve_ode(
        field, x, n_steps=n_steps, method=method, return_trajectory=True
    )
    generator = torch.Generator().manual_seed(seed)
    step = 1.0 / n_steps
    total = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    for k in range(n_steps):
        state = trajectory[k].detach().requires_grad_(True)
        t = torch.full((state.shape[0],), k * step, dtype=state.dtype, device=state.device)
        velocity = field(state, t)
        divergence = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
        # A field with no dependence on the state has zero divergence and no
        # graph to differentiate; asking autograd for one would raise.
        for _ in range(n_hutchinson) if velocity.requires_grad else ():
            probe = (
                torch.randint(
                    0, 2, state.shape, generator=generator, dtype=torch.int64
                ).to(state.dtype)
                * 2.0
                - 1.0
            )
            (grad,) = torch.autograd.grad(
                velocity, state, grad_outputs=probe, retain_graph=True, create_graph=False
            )
            divergence = divergence + (grad * probe).sum(dim=-1)
        total = total + step * divergence / n_hutchinson

    return total.detach()
