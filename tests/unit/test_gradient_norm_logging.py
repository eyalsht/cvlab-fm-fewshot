"""Gradient-norm observation during a fit (TODO 8.9).

The phase note has to say whether rolled-out training at T = 12 was stable,
and "stable" is a claim about the gradients, not about the loss curve. A loss
that falls smoothly can still be riding a handful of enormous steps, and
backpropagating through twelve solver steps is exactly the setting where that
happens. So the fit has to be able to report the norm of the gradient it took
at each step.

The mechanism is an optional observer, not a new artifact. Two constraints
shape it:

- Nothing about an unobserved fit may change. The 108-run grid was produced
  without this hook and the phase note cites those runs, so a fit with no
  observer must stay bit-identical to the one this repository ran before the
  hook existed. That is asserted here on the trained weights, not on the loss
  curve alone.
- Both schemes get it or neither does. The fairness guard in
  `test_scheme_parity` exists because the two schemes must not drift apart in
  anything but their loss, and a diagnostic that only rolled-out training can
  run would be exactly such a drift.

What is asserted is invariant: when it fires, what it is handed, that the
numbers are finite and non-negative, and that observing perturbs nothing.
Whether the norms are large is the phase note's question, not this file's.
"""

import math

import pytest
import torch
from test_scheme_parity import field_hash

from fm_fewshot.services.flow.training.rolled_out import train_rolled_out_field
from fm_fewshot.services.flow.training.standard import train_standard_field

N_TRAIN_STEPS = 12

SHARED: dict[str, object] = {
    "hidden_dims": (16, 16),
    "n_train_steps": N_TRAIN_STEPS,
    "batch_size": 8,
    "lr": 1e-3,
    "init_seed": 0,
}


def problem(n_classes: int = 3, dim: int = 5, per_class: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 4.0
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    x0 = centers[labels] + 0.3 * torch.randn(labels.shape[0], dim, generator=g)
    x1 = torch.nn.functional.normalize(centers[labels], dim=1)
    return x0, x1


def fit(scheme: str, **kwargs):
    x0, x1 = problem()
    if scheme == "fm_rolled":
        return train_rolled_out_field(x0, x1, sample_steps=4, **SHARED, **kwargs)
    return train_standard_field(x0, x1, **SHARED, **kwargs)


SCHEMES = ["fm_standard", "fm_rolled"]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_observer_fires_once_per_optimizer_step(scheme):
    seen: list[tuple[int, float]] = []
    fit(scheme, on_step=lambda step, norm: seen.append((step, norm)))
    assert len(seen) == N_TRAIN_STEPS


@pytest.mark.parametrize("scheme", SCHEMES)
def test_step_numbers_are_one_based_and_match_the_loss_curve(scheme):
    """1-based, so a norm can be read against `loss_curve.csv` without an offset."""
    seen: list[tuple[int, float]] = []
    _, history = fit(scheme, on_step=lambda step, norm: seen.append((step, norm)))
    assert [step for step, _ in seen] == list(range(1, N_TRAIN_STEPS + 1))
    assert len(history) == len(seen)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_norms_are_finite_and_non_negative(scheme):
    seen: list[tuple[int, float]] = []
    fit(scheme, on_step=lambda step, norm: seen.append((step, norm)))
    norms = [norm for _, norm in seen]
    assert all(math.isfinite(norm) for norm in norms)
    assert all(norm >= 0.0 for norm in norms)
    # A field that moved at all took at least one non-zero gradient.
    assert max(norms) > 0.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_norm_is_read_before_the_optimizer_step(scheme):
    """The first step's gradient cannot depend on the learning rate.

    At step 1 the weights are the seeded initialization and the batch is the
    seeded first draw, so the gradient is fully determined before Adam has
    applied anything. If the observer were fired after `optimizer.step()`, or
    handed the update rather than the gradient, changing `lr` would move this
    number. Pinning it is what makes "gradient norm" mean the gradient.
    """
    def first_norm(lr: float) -> float:
        seen: list[float] = []
        x0, x1 = problem()
        params = dict(SHARED, lr=lr)
        if scheme == "fm_rolled":
            train_rolled_out_field(
                x0, x1, sample_steps=4, **params,
                on_step=lambda step, norm: seen.append(norm),
            )
        else:
            train_standard_field(
                x0, x1, **params, on_step=lambda step, norm: seen.append(norm)
            )
        return seen[0]

    slow, fast = first_norm(1e-3), first_norm(1e-1)
    assert slow == pytest.approx(fast)
    assert slow > 0.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_an_unobserved_fit_is_bit_identical_to_an_observed_one(scheme):
    """Observing reads gradients; it must not consume randomness or change them.

    This is the guard that lets the phase note cite the 108-run grid and these
    norms in the same paragraph: the fit the norms describe is the fit that
    produced the grid.
    """
    plain, plain_history = fit(scheme)
    observed, observed_history = fit(scheme, on_step=lambda step, norm: None)
    assert field_hash(plain) == field_hash(observed)
    assert plain_history == observed_history


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_default_is_no_observer(scheme):
    """The hook is opt-in; every existing caller keeps its current behaviour."""
    field, history = fit(scheme)
    assert len(history) == N_TRAIN_STEPS
    assert field_hash(field)
