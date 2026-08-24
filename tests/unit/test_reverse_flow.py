"""Reverse-flow tests per PRD_reverse_flow section 5.

Two halves. The grid, the constant-field cycle and the known-contraction
control are deterministic infrastructure and get strict TDD: they check the
integrator against closed forms and measured convergence orders. The basin and
meeting tests run on the 2D toy and assert invariants, shapes, determinism and
closed-form agreement, never an accuracy number.

Nothing here reads a real feature cache or a grid result. Reverse flow is a
diagnostic (ADR-027) and its numbers never reach TABLE.md, so its tests never
touch the evaluation path.
"""

import math

import pytest
import torch

from fm_fewshot.services.flow import reverse
from fm_fewshot.services.flow.reverse import (
    ReverseConfig,
    basin_alignment,
    basin_samples,
    cycle_relative_error,
    integrate_segment,
    log_volume_change,
    median_iqr,
    meet_gaps,
    residual_sigma,
)
from fm_fewshot.services.flow.solver import solve_ode
from fm_fewshot.services.flow.toy import make_toy_problem, train_toy_field
from fm_fewshot.services.flow.training.standard import train_standard_field

METHODS = ("euler", "midpoint")


@pytest.fixture(scope="module")
def one_pair():
    """One pair trained to convergence: no conflicting conditional target
    anywhere, so the learned field should reproduce the interpolant."""
    z = torch.tensor([[1.5, -0.5]])
    p = torch.tensor([[-1.0, 2.0]])
    field, _ = train_standard_field(
        z.repeat(16, 1),
        p.repeat(16, 1),
        hidden_dims=(64, 64),
        n_train_steps=1500,
        batch_size=16,
        lr=1e-2,
        init_seed=0,
    )
    field.eval()
    return field, z, p


@pytest.fixture(scope="module")
def fitted_toy():
    problem = make_toy_problem(seed=0, n_classes=3, per_class=128, anisotropy=2.0)
    field = train_toy_field(problem, seed=0, steps=1200)
    field.eval()
    return problem, field


def constant_field(c: torch.Tensor):
    def v(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return c.expand_as(x)

    return v


def contraction_field(a: float):
    """v(x, t) = -a x. The exact flow is x -> x exp(-a), invertible, so the
    exact forward-then-backward cycle is the identity and every bit of the
    measured cycle error is the solver's."""

    def v(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -a * x

    return v


class Recorder:
    """A zero field that records the times it was asked about."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.times.append(float(t[0]))
        return torch.zeros_like(x)


class TestConstantFieldExactness:
    """With v = c the forward and reverse legs are exact term by term, so a
    round trip returns x0 for any T and either method. This is the test that
    the two grids mirror."""

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_steps", [1, 2, 4, 12, 32])
    def test_round_trip_returns_x0(self, method: str, n_steps: int) -> None:
        x0 = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
        field = constant_field(torch.tensor([1.0, -2.0, 0.5]))
        forward = solve_ode(field, x0, n_steps=n_steps, method=method)
        back = integrate_segment(
            field, forward, t_start=1.0, t_end=0.0, n_steps=n_steps, method=method
        )
        assert torch.allclose(back, x0, atol=1e-6)

    @pytest.mark.parametrize("method", METHODS)
    def test_cycle_error_is_zero(self, method: str) -> None:
        x0 = torch.randn(8, 3, generator=torch.Generator().manual_seed(1))
        field = constant_field(torch.tensor([0.3, -0.7, 2.0]))
        error = cycle_relative_error(
            field,
            x0,
            forward_steps=8,
            reverse_steps=8,
            forward_method=method,
            reverse_method=method,
        )
        assert error.shape == (8,)
        assert float(error.max()) < 1e-6


class TestReverseGrid:
    """Forward evaluates t at 0, 1/T, .., (T-1)/T; reverse at 1, (T-1)/T, ..,
    1/T. Neither grid holds the other's missing endpoint, and the off-by-one
    that swaps them is silent, so a recording stub asserts it directly."""

    @pytest.mark.parametrize("n_steps", [4, 8, 16, 32])
    @pytest.mark.parametrize("method", ["euler"])
    def test_reverse_times_are_exact_on_a_dyadic_grid(
        self, n_steps: int, method: str
    ) -> None:
        recorder = Recorder()
        integrate_segment(
            recorder, torch.zeros(3, 2), t_start=1.0, t_end=0.0, n_steps=n_steps, method=method
        )
        expected = [(n_steps - k) / n_steps for k in range(n_steps)]
        assert len(recorder.times) == n_steps
        assert recorder.times == expected

    def test_reverse_times_at_a_non_dyadic_step_count(self) -> None:
        recorder = Recorder()
        integrate_segment(
            recorder, torch.zeros(3, 2), t_start=1.0, t_end=0.0, n_steps=12, method="euler"
        )
        expected = [(12 - k) / 12 for k in range(12)]
        assert len(recorder.times) == 12
        assert recorder.times == pytest.approx(expected, abs=1e-6)

    def test_reverse_starts_at_one_and_never_reaches_zero(self) -> None:
        recorder = Recorder()
        integrate_segment(
            recorder, torch.zeros(2, 2), t_start=1.0, t_end=0.0, n_steps=4, method="euler"
        )
        assert recorder.times[0] == 1.0
        assert 0.0 not in recorder.times

    def test_forward_starts_at_zero_and_never_reaches_one(self) -> None:
        recorder = Recorder()
        solve_ode(recorder, torch.zeros(2, 2), n_steps=4, method="euler")
        assert recorder.times[0] == 0.0
        assert 1.0 not in recorder.times

    def test_midpoint_reverse_evaluates_half_steps_inside_the_segment(self) -> None:
        recorder = Recorder()
        integrate_segment(
            recorder, torch.zeros(2, 2), t_start=1.0, t_end=0.0, n_steps=4, method="midpoint"
        )
        assert recorder.times == pytest.approx(
            [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125], abs=1e-6
        )


class TestKnownContractionControl:
    """v(x, t) = -a x. The exact cycle is the identity, so the measured cycle
    error is purely numerical and must fall at the method's order. This is what
    licenses reading a T-invariant residual on a real field as contraction
    rather than as solver error."""

    @pytest.mark.parametrize(("method", "expected_order"), [("euler", 1.0), ("midpoint", 2.0)])
    def test_the_return_leg_converges_at_the_method_order(
        self, method: str, expected_order: float
    ) -> None:
        """The reverse leg against the exact backward flow. x(1) = x0 exp(-a)
        is known in closed form, so integrating it back to t = 0 has the
        method's ordinary global error and nothing else."""
        x0 = torch.full((1, 1), 1.5)
        a = 1.0
        field = contraction_field(a)
        x1 = x0 * math.exp(-a)

        errors = []
        for n_steps in (4, 8, 16, 32):
            back = reverse.reverse_transport(field, x1, n_steps=n_steps, method=method)
            errors.append(float((back - x0).abs().max()))

        orders = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        measured = sum(orders) / len(orders)
        assert measured == pytest.approx(expected_order, abs=0.2), orders

    @pytest.mark.parametrize(("method", "least_order"), [("euler", 1.0), ("midpoint", 2.0)])
    def test_the_round_trip_falls_at_least_at_the_method_order(
        self, method: str, least_order: float
    ) -> None:
        """The round trip, not the single leg. Euler comes in at order 1 as
        expected; midpoint comes in at 3, not 2, because on a linear field the
        forward and backward steps are (1 - ah + (ah)^2/2) and
        (1 + ah + (ah)^2/2), whose product is 1 + (ah)^4/4, so the leading
        second-order terms cancel and the composed error is one order better
        than either leg. The assertion is therefore a floor, which is all the
        contraction argument needs: what must be true is that solver error
        falls with T while contraction does not."""
        x0 = torch.full((1, 1), 1.5)
        field = contraction_field(1.0)

        errors = []
        for n_steps in (4, 8, 16, 32):
            error = cycle_relative_error(
                field,
                x0,
                forward_steps=n_steps,
                reverse_steps=n_steps,
                forward_method=method,
                reverse_method=method,
            )
            errors.append(float(error.max()))

        orders = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        measured = sum(orders) / len(orders)
        assert measured > least_order - 0.2, orders

    def test_error_falls_monotonically_with_t(self) -> None:
        x0 = torch.full((1, 1), 1.5)
        field = contraction_field(1.0)
        errors = [
            float(
                cycle_relative_error(
                    field, x0, forward_steps=n, reverse_steps=n, forward_method="euler",
                    reverse_method="euler",
                ).max()
            )
            for n in (4, 8, 16, 32)
        ]
        assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))

    def test_midpoint_return_beats_euler_return_at_equal_steps(self) -> None:
        x0 = torch.full((1, 1), 1.5)
        field = contraction_field(1.0)
        euler = float(
            cycle_relative_error(
                field, x0, forward_steps=8, reverse_steps=8, forward_method="midpoint",
                reverse_method="euler",
            ).max()
        )
        midpoint = float(
            cycle_relative_error(
                field, x0, forward_steps=8, reverse_steps=8, forward_method="midpoint",
                reverse_method="midpoint",
            ).max()
        )
        assert midpoint < euler


class TestMedianIqr:
    """Contraction is heavy-tailed, so the reported statistic is the median and
    the interquartile range, never the mean and std."""

    def test_reports_median_and_iqr(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = median_iqr(values)
        assert stats["median"] == pytest.approx(2.5)
        assert stats["iqr"] == pytest.approx(stats["q3"] - stats["q1"])
        assert stats["n"] == 4

    def test_a_single_outlier_moves_the_mean_but_not_the_median(self) -> None:
        clean = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        tailed = torch.tensor([1.0, 1.0, 1.0, 1.0, 1000.0])
        assert median_iqr(clean)["median"] == median_iqr(tailed)["median"]

    def test_empty_population_reports_no_statistic(self) -> None:
        stats = median_iqr(torch.zeros(0))
        assert stats["n"] == 0
        assert stats["median"] is None


class TestSinglePairMeeting:
    """One pair trained to convergence: the learned field has no conflicting
    conditional target anywhere, so the forward and backward states must both
    sit on the true interpolant."""

    @pytest.mark.parametrize("t_star", [0.25, 0.5, 0.75])
    def test_both_legs_agree_with_the_interpolant(self, one_pair, t_star: float) -> None:
        field, z, p = one_pair
        gaps = meet_gaps(field, z, p, t_star=t_star, n_steps=32)
        assert gaps["gap_forward"]["median"] < 0.05
        assert gaps["gap_backward"]["median"] < 0.05
        assert gaps["gap_between"]["median"] < 0.05

    def test_gaps_are_reported_for_every_meeting_time(self, one_pair) -> None:
        field, z, p = one_pair
        for t_star in (0.25, 0.5, 0.75):
            gaps = meet_gaps(field, z, p, t_star=t_star, n_steps=8)
            assert set(gaps) == {"gap_forward", "gap_backward", "gap_between"}


class TestBasinOnTheToy:
    """The anisotropic 2D toy of PRD_flow_matching_block. The claim is
    geometric, not numerical: the cloud that flows to prototype c sits over
    class c's own points."""

    def test_sigma_is_derived_from_the_training_residual(self, fitted_toy) -> None:
        problem, field = fitted_toy
        with torch.no_grad():
            transported = solve_ode(field, problem.train_x, n_steps=4, method="euler")
        targets = problem.prototypes[problem.train_y]
        sigma = residual_sigma(transported, targets, sigma_scale=1.0)
        expected = float((transported - targets).norm(dim=1).median())
        assert sigma == pytest.approx(expected)
        assert residual_sigma(transported, targets, sigma_scale=0.0) == 0.0

    def test_clouds_have_one_row_per_class_and_sample(self, fitted_toy) -> None:
        _, field = fitted_toy
        prototypes = torch.randn(3, 2, generator=torch.Generator().manual_seed(3))
        clouds = basin_samples(
            field, prototypes, sigma=0.2, n_samples=16, n_steps=8, seed=0
        )
        assert clouds.shape == (3, 16, 2)
        assert torch.isfinite(clouds).all()

    def test_zero_sigma_gives_one_point_per_class(self, fitted_toy) -> None:
        _, field = fitted_toy
        prototypes = torch.randn(3, 2, generator=torch.Generator().manual_seed(4))
        clouds = basin_samples(
            field, prototypes, sigma=0.0, n_samples=8, n_steps=8, seed=0
        )
        # Exactly one point in exact arithmetic. The residual is float32
        # noise: a batched GEMM does not return bit-identical rows even for
        # bit-identical inputs, measured at 4e-7 per layer on this field.
        for c in range(clouds.shape[0]):
            spread = (clouds[c] - clouds[c].mean(dim=0)).norm(dim=1).max()
            assert float(spread) < 1e-4

    def test_alignment_is_diagonally_dominant(self, fitted_toy) -> None:
        problem, field = fitted_toy
        with torch.no_grad():
            transported = solve_ode(field, problem.train_x, n_steps=4, method="euler")
        sigma = residual_sigma(transported, problem.prototypes[problem.train_y], 1.0)
        clouds = basin_samples(
            field, problem.prototypes, sigma=sigma, n_samples=64, n_steps=8, seed=0
        )
        alignment = basin_alignment(clouds, problem.test_x, problem.test_y, problem.n_classes)
        assert alignment.shape == (3, 3)
        for c in range(3):
            assert int(alignment[c].argmax()) == c

    def test_each_cloud_sits_over_its_own_class(self, fitted_toy) -> None:
        problem, field = fitted_toy
        with torch.no_grad():
            transported = solve_ode(field, problem.train_x, n_steps=4, method="euler")
        sigma = residual_sigma(transported, problem.prototypes[problem.train_y], 1.0)
        clouds = basin_samples(
            field, problem.prototypes, sigma=sigma, n_samples=64, n_steps=8, seed=0
        )
        centers = clouds.mean(dim=1)
        for c in range(problem.n_classes):
            points = problem.test_x[problem.test_y == c]
            distances = torch.cdist(points, centers).mean(dim=0)
            assert int(distances.argmin()) == c


class TestDeterminism:
    def test_two_invocations_at_one_seed_are_bit_identical(self) -> None:
        problem = make_toy_problem(seed=0, n_classes=3, per_class=32, anisotropy=2.0)
        field = train_toy_field(problem, seed=0, steps=50)
        field.eval()
        first = basin_samples(field, problem.prototypes, sigma=0.3, n_samples=16, n_steps=8, seed=1)
        second = basin_samples(
            field, problem.prototypes, sigma=0.3, n_samples=16, n_steps=8, seed=1
        )
        assert torch.equal(first, second)

    def test_a_different_seed_gives_a_different_cloud(self) -> None:
        problem = make_toy_problem(seed=0, n_classes=3, per_class=32, anisotropy=2.0)
        field = train_toy_field(problem, seed=0, steps=50)
        field.eval()
        first = basin_samples(field, problem.prototypes, sigma=0.3, n_samples=16, n_steps=8, seed=1)
        second = basin_samples(
            field, problem.prototypes, sigma=0.3, n_samples=16, n_steps=8, seed=2
        )
        assert not torch.equal(first, second)

    def test_class_seeds_do_not_depend_on_iteration_order(self) -> None:
        """The seed derives from (seed, "reverse", class_id), so class c's cloud
        is the same whether or not the other classes were drawn first."""
        problem = make_toy_problem(seed=0, n_classes=3, per_class=32, anisotropy=2.0)
        field = train_toy_field(problem, seed=0, steps=50)
        field.eval()
        full = basin_samples(field, problem.prototypes, sigma=0.3, n_samples=8, n_steps=8, seed=1)
        alone = basin_samples(
            field,
            problem.prototypes[2:],
            sigma=0.3,
            n_samples=8,
            n_steps=8,
            seed=1,
            class_ids=(2,),
        )
        assert torch.equal(full[2], alone[0])


class TestValidation:
    def test_volume_without_hutchinson_vectors_refuses(self) -> None:
        with pytest.raises(ValueError, match="n_hutchinson"):
            ReverseConfig(mode="volume", n_hutchinson=0)

    def test_a_non_volume_mode_ignores_n_hutchinson(self) -> None:
        assert ReverseConfig(mode="cycle", n_hutchinson=0).n_hutchinson == 0

    def test_unknown_mode_refuses_listing_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            ReverseConfig(mode="sideways")

    @pytest.mark.parametrize("steps", [(0,), (4, -1)])
    def test_non_positive_reverse_steps_refuse(self, steps) -> None:
        with pytest.raises(ValueError, match="reverse_steps"):
            ReverseConfig(reverse_steps=steps)

    def test_negative_sigma_scale_refuses(self) -> None:
        with pytest.raises(ValueError, match="sigma_scale"):
            ReverseConfig(sigma_scale=-0.1)

    @pytest.mark.parametrize("times", [(0.0,), (1.0,), (0.5, 1.5)])
    def test_meet_times_outside_the_open_interval_refuse(self, times) -> None:
        with pytest.raises(ValueError, match="meet_times"):
            ReverseConfig(meet_times=times)

    def test_unknown_return_solver_refuses(self) -> None:
        with pytest.raises(ValueError, match="reverse_method"):
            ReverseConfig(reverse_method=("rk4",))

    def test_non_finite_start_refuses(self) -> None:
        field = constant_field(torch.zeros(2))
        start = torch.tensor([[float("nan"), 0.0]])
        with pytest.raises(ValueError, match="finite"):
            integrate_segment(field, start, t_start=1.0, t_end=0.0, n_steps=4)

    def test_negative_sigma_refuses_at_sampling_time(self) -> None:
        field = constant_field(torch.zeros(2))
        with pytest.raises(ValueError, match="sigma"):
            basin_samples(field, torch.zeros(2, 2), sigma=-1.0, n_samples=4, n_steps=4, seed=0)

    def test_defaults_match_the_prd_table(self) -> None:
        cfg = ReverseConfig()
        assert cfg.mode == "cycle"
        assert cfg.reverse_steps == (4, 12, 32)
        assert cfg.reverse_method == ("euler", "midpoint")
        assert cfg.sigma_scale == 1.0
        assert cfg.n_samples == 128
        assert cfg.meet_times == (0.25, 0.5, 0.75)
        assert cfg.n_hutchinson == 1


class TestVolume:
    """Stretch mode. The closed form is the check: for v = -a x in D
    dimensions, div v = -a D, so the log volume change over [0, 1] is -a D
    exactly, whatever the point."""

    def test_matches_the_closed_form_on_a_linear_field(self) -> None:
        field = contraction_field(0.5)
        x = torch.randn(6, 3, generator=torch.Generator().manual_seed(5))
        change = log_volume_change(field, x, n_steps=64, n_hutchinson=64, seed=0)
        assert change.shape == (6,)
        assert torch.allclose(change, torch.full((6,), -1.5), atol=0.1)

    def test_a_constant_field_preserves_volume(self) -> None:
        field = constant_field(torch.tensor([1.0, -1.0]))
        x = torch.randn(4, 2, generator=torch.Generator().manual_seed(6))
        change = log_volume_change(field, x, n_steps=8, n_hutchinson=4, seed=0)
        assert torch.allclose(change, torch.zeros(4), atol=1e-5)

    def test_is_deterministic_at_a_fixed_seed(self) -> None:
        field = contraction_field(0.5)
        x = torch.randn(4, 2, generator=torch.Generator().manual_seed(7))
        first = log_volume_change(field, x, n_steps=8, n_hutchinson=2, seed=3)
        second = log_volume_change(field, x, n_steps=8, n_hutchinson=2, seed=3)
        assert torch.equal(first, second)


class TestModuleSurface:
    def test_modes_are_the_four_the_prd_names(self) -> None:
        assert reverse.REVERSE_MODES == ("cycle", "basin", "meet", "volume")
