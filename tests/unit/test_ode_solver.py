"""ODE solver tests per PRD_ode_solver section 5.

Empirical, so the assertions are invariants and orders rather than accuracy
numbers: exactness on fields with a known closed form, measured convergence
order, differentiability, and determinism.
"""

import math

import pytest
import torch

from fm_fewshot.services.flow.solver import solve_ode

METHODS = ("euler", "midpoint")


def constant_field(c: torch.Tensor):
    def v(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return c.expand_as(x)

    return v


def linear_decay_field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """dx/dt = -x, exact solution x(1) = x0 * exp(-1)."""
    return -x


def time_only_field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """dx/dt = t, exact solution x(1) = x0 + 1/2."""
    return t.view(-1, 1).expand_as(x)


class TestConstantField:
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_steps", [1, 2, 8, 32])
    def test_returns_x0_plus_c_for_any_step_count(self, method: str, n_steps: int) -> None:
        x0 = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
        c = torch.tensor([1.0, -2.0, 0.5])
        out = solve_ode(constant_field(c), x0, n_steps=n_steps, method=method)
        assert torch.allclose(out, x0 + c, atol=1e-5)


class TestConvergenceOrder:
    @pytest.mark.parametrize(
        ("method", "expected_order"), [("euler", 1.0), ("midpoint", 2.0)]
    )
    def test_measured_order_matches_the_method(
        self, method: str, expected_order: float
    ) -> None:
        """Halving the step should cut the error by 2^order."""
        x0 = torch.ones(1, 1)
        exact = x0 * math.exp(-1.0)

        errors = []
        for n_steps in (8, 16, 32, 64):
            out = solve_ode(linear_decay_field, x0, n_steps=n_steps, method=method)
            errors.append(float((out - exact).abs().max()))

        orders = [
            math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)
        ]
        measured = sum(orders) / len(orders)
        assert measured == pytest.approx(expected_order, abs=0.15), orders

    def test_midpoint_beats_euler_at_equal_steps(self) -> None:
        x0 = torch.ones(1, 1)
        exact = x0 * math.exp(-1.0)
        euler = (solve_ode(linear_decay_field, x0, n_steps=8, method="euler") - exact).abs()
        mid = (solve_ode(linear_decay_field, x0, n_steps=8, method="midpoint") - exact).abs()
        assert float(mid) < float(euler)


class TestTimeGrid:
    @pytest.mark.parametrize("method", METHODS)
    def test_time_dependent_field_integrates_correctly(self, method: str) -> None:
        """dx/dt = t over [0,1] adds exactly 1/2; midpoint is exact here."""
        x0 = torch.zeros(2, 2)
        out = solve_ode(time_only_field, x0, n_steps=64, method=method)
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=2e-2)

    def test_midpoint_is_exact_on_a_linear_time_field(self) -> None:
        out = solve_ode(time_only_field, torch.zeros(1, 1), n_steps=2, method="midpoint")
        assert torch.allclose(out, torch.tensor([[0.5]]), atol=1e-6)

    def test_t_reaches_the_field_as_a_batch_tensor(self) -> None:
        seen = []

        def recorder(x, t):
            seen.append(t.clone())
            return torch.zeros_like(x)

        solve_ode(recorder, torch.zeros(5, 2), n_steps=3, method="euler")
        assert all(t.shape == (5,) for t in seen)
        assert [float(t[0]) for t in seen] == pytest.approx([0.0, 1 / 3, 2 / 3])


class TestTrajectory:
    @pytest.mark.parametrize("method", METHODS)
    def test_trajectory_has_n_plus_one_states_and_matches_the_endpoint(
        self, method: str
    ) -> None:
        x0 = torch.randn(3, 2, generator=torch.Generator().manual_seed(1))
        out, trajectory = solve_ode(
            linear_decay_field, x0, n_steps=8, method=method, return_trajectory=True
        )
        assert trajectory.shape == (9, 3, 2)
        assert torch.equal(trajectory[0], x0)
        assert torch.equal(trajectory[-1], out)

    def test_trajectory_is_optional(self) -> None:
        out = solve_ode(linear_decay_field, torch.ones(1, 1), n_steps=4)
        assert isinstance(out, torch.Tensor)


class TestDifferentiability:
    def test_gradients_flow_through_every_step(self) -> None:
        """Rolled-out training needs this; no no_grad inside the solver."""
        weight = torch.tensor([[0.7]], requires_grad=True)

        def field(x, t):
            return x @ weight

        out = solve_ode(field, torch.ones(1, 1), n_steps=4, method="euler")
        out.sum().backward()
        assert weight.grad is not None
        assert float(weight.grad.abs()) > 0

    def test_gradient_matches_finite_differences(self) -> None:
        eps = 1e-4

        def run(w: float) -> torch.Tensor:
            weight = torch.tensor([[w]])
            return solve_ode(
                lambda x, t: x @ weight, torch.ones(1, 1), n_steps=4, method="euler"
            )

        w0 = 0.7
        weight = torch.tensor([[w0]], requires_grad=True)
        out = solve_ode(
            lambda x, t: x @ weight, torch.ones(1, 1), n_steps=4, method="euler"
        )
        out.sum().backward()
        numeric = float((run(w0 + eps) - run(w0 - eps)) / (2 * eps))
        assert float(weight.grad) == pytest.approx(numeric, rel=1e-3)


class TestDeterminism:
    @pytest.mark.parametrize("method", METHODS)
    def test_same_input_twice_is_bit_identical(self, method: str) -> None:
        x0 = torch.randn(4, 3, generator=torch.Generator().manual_seed(2))
        first = solve_ode(linear_decay_field, x0, n_steps=8, method=method)
        second = solve_ode(linear_decay_field, x0, n_steps=8, method=method)
        assert torch.equal(first, second)


class TestValidation:
    @pytest.mark.parametrize("n_steps", [0, -1])
    def test_non_positive_step_count_raises(self, n_steps: int) -> None:
        with pytest.raises(ValueError, match="n_steps"):
            solve_ode(linear_decay_field, torch.ones(1, 1), n_steps=n_steps)

    def test_unknown_method_raises_listing_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            solve_ode(linear_decay_field, torch.ones(1, 1), n_steps=4, method="rk4")

    def test_non_finite_state_aborts_naming_the_step(self) -> None:
        def exploding(x, t):
            return torch.full_like(x, float("nan"))

        with pytest.raises(ValueError, match="step 0"):
            solve_ode(exploding, torch.ones(1, 1), n_steps=4)

    def test_output_shape_equals_input_shape(self) -> None:
        x0 = torch.randn(7, 5)
        assert solve_ode(linear_decay_field, x0, n_steps=3).shape == x0.shape
