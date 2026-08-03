"""Linear path, CFM objective, and velocity MLP per PRD_flow_matching_block."""

import math

import pytest
import torch

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.path import conditional_target, interpolate
from fm_fewshot.services.flow.solver import solve_ode
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP


class TestLinearPath:
    def test_endpoints(self) -> None:
        x0 = torch.tensor([[0.0, 0.0]])
        x1 = torch.tensor([[2.0, 4.0]])
        assert torch.allclose(interpolate(x0, x1, torch.tensor([0.0])), x0)
        assert torch.allclose(interpolate(x0, x1, torch.tensor([1.0])), x1)

    def test_midpoint_is_the_average(self) -> None:
        x0 = torch.tensor([[0.0, 0.0]])
        x1 = torch.tensor([[2.0, 4.0]])
        got = interpolate(x0, x1, torch.tensor([0.5]))
        assert torch.allclose(got, torch.tensor([[1.0, 2.0]]))

    def test_matches_the_formula_elementwise(self) -> None:
        g = torch.Generator().manual_seed(0)
        x0 = torch.randn(6, 3, generator=g)
        x1 = torch.randn(6, 3, generator=g)
        t = torch.rand(6, generator=g)
        expected = (1 - t).unsqueeze(1) * x0 + t.unsqueeze(1) * x1
        assert torch.allclose(interpolate(x0, x1, t), expected)

    def test_conditional_target_is_x1_minus_x0(self) -> None:
        """dx_t/dt along the linear path, independent of t."""
        g = torch.Generator().manual_seed(1)
        x0, x1 = torch.randn(4, 2, generator=g), torch.randn(4, 2, generator=g)
        assert torch.equal(conditional_target(x0, x1), x1 - x0)

    def test_time_out_of_range_raises(self) -> None:
        x0 = x1 = torch.zeros(1, 2)
        with pytest.raises(ValueError, match="0, 1"):
            interpolate(x0, x1, torch.tensor([1.5]))


class TestCfmLoss:
    def test_matches_a_hand_computation(self) -> None:
        x0 = torch.tensor([[0.0, 0.0]])
        x1 = torch.tensor([[2.0, 0.0]])
        t = torch.tensor([0.5])
        # Field predicts (1, 0); the target is x1 - x0 = (2, 0).
        field = lambda x, tt: torch.tensor([[1.0, 0.0]])  # noqa: E731
        # MSE over both components: ((1-2)^2 + (0-0)^2) / 2 = 0.5
        assert float(cfm_loss(field, x0, x1, t)) == pytest.approx(0.5)

    def test_is_zero_when_the_field_is_exact(self) -> None:
        g = torch.Generator().manual_seed(2)
        x0, x1 = torch.randn(5, 3, generator=g), torch.randn(5, 3, generator=g)
        t = torch.rand(5, generator=g)
        exact = lambda x, tt: x1 - x0  # noqa: E731
        assert float(cfm_loss(exact, x0, x1, t)) == pytest.approx(0.0, abs=1e-12)

    def test_is_non_negative(self) -> None:
        g = torch.Generator().manual_seed(3)
        x0, x1 = torch.randn(8, 4, generator=g), torch.randn(8, 4, generator=g)
        t = torch.rand(8, generator=g)
        field = lambda x, tt: torch.zeros_like(x)  # noqa: E731
        assert float(cfm_loss(field, x0, x1, t)) >= 0.0

    def test_the_field_is_evaluated_on_the_interpolate(self) -> None:
        seen = {}

        def field(x, t):
            seen["x"] = x.clone()
            return torch.zeros_like(x)

        x0 = torch.zeros(1, 2)
        x1 = torch.tensor([[4.0, 0.0]])
        cfm_loss(field, x0, x1, torch.tensor([0.25]))
        assert torch.allclose(seen["x"], torch.tensor([[1.0, 0.0]]))


class TestVelocityMlp:
    def test_output_shape_equals_input_shape(self) -> None:
        net = VelocityMLP(dim=6, hidden_dims=(16, 16), time_embed_dim=8, seed=0)
        x = torch.randn(5, 6)
        t = torch.rand(5)
        assert net(x, t).shape == (5, 6)

    def test_same_seed_gives_identical_initialization(self) -> None:
        a = VelocityMLP(dim=4, hidden_dims=(8,), time_embed_dim=4, seed=7)
        b = VelocityMLP(dim=4, hidden_dims=(8,), time_embed_dim=4, seed=7)
        x, t = torch.randn(3, 4), torch.rand(3)
        assert torch.equal(a(x, t), b(x, t))

    def test_different_seeds_differ(self) -> None:
        a = VelocityMLP(dim=4, hidden_dims=(8,), time_embed_dim=4, seed=0)
        b = VelocityMLP(dim=4, hidden_dims=(8,), time_embed_dim=4, seed=1)
        x, t = torch.randn(3, 4), torch.rand(3)
        assert not torch.equal(a(x, t), b(x, t))

    def test_time_actually_changes_the_output(self) -> None:
        """A field that ignores t would silently be time-independent."""
        net = VelocityMLP(dim=3, hidden_dims=(16,), time_embed_dim=8, seed=0)
        x = torch.randn(4, 3)
        early = net(x, torch.zeros(4))
        late = net(x, torch.ones(4))
        assert not torch.allclose(early, late)

    def test_odd_time_embed_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="even"):
            VelocityMLP(dim=3, hidden_dims=(8,), time_embed_dim=7, seed=0)


class TestSinglePairConvergence:
    def test_trained_field_recovers_the_conditional_target(self) -> None:
        """One (x0, x1) pair: the optimum is the constant field x1 - x0."""
        torch.manual_seed(0)
        x0 = torch.tensor([[0.0, 0.0]])
        x1 = torch.tensor([[1.0, 2.0]])
        net = VelocityMLP(dim=2, hidden_dims=(64, 64), time_embed_dim=16, seed=0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        for _ in range(600):
            t = torch.rand(32)
            loss = cfm_loss(net, x0.expand(32, 2), x1.expand(32, 2), t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        target = x1 - x0
        for probe in (0.0, 0.25, 0.5, 0.75, 1.0):
            xt = interpolate(x0, x1, torch.tensor([probe]))
            predicted = net(xt, torch.tensor([probe]))
            assert torch.allclose(predicted, target, atol=0.1), (probe, predicted)

    def test_transport_lands_on_the_target(self) -> None:
        torch.manual_seed(0)
        x0 = torch.tensor([[0.0, 0.0]])
        x1 = torch.tensor([[1.0, 2.0]])
        net = VelocityMLP(dim=2, hidden_dims=(64, 64), time_embed_dim=16, seed=0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(600):
            t = torch.rand(32)
            loss = cfm_loss(net, x0.expand(32, 2), x1.expand(32, 2), t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            landed = solve_ode(net, x0, n_steps=8, method="euler")
        assert float((landed - x1).norm()) < 0.1

    def test_loss_decreases(self) -> None:
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        x0 = torch.randn(64, 3, generator=g)
        x1 = torch.randn(64, 3, generator=g)
        net = VelocityMLP(dim=3, hidden_dims=(32, 32), time_embed_dim=8, seed=0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        first = None
        for step in range(300):
            t = torch.rand(64)
            loss = cfm_loss(net, x0, x1, t)
            if step == 0:
                first = float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert float(loss) < first


class TestSinusoidalEmbedding:
    def test_embedding_is_bounded_and_finite(self) -> None:
        from fm_fewshot.services.flow.velocity_mlp import sinusoidal_time_embedding

        t = torch.linspace(0.0, 1.0, 17)
        embedded = sinusoidal_time_embedding(t, 16)
        assert embedded.shape == (17, 16)
        assert torch.isfinite(embedded).all()
        assert float(embedded.abs().max()) <= 1.0 + 1e-6

    def test_distinct_times_get_distinct_embeddings(self) -> None:
        from fm_fewshot.services.flow.velocity_mlp import sinusoidal_time_embedding

        embedded = sinusoidal_time_embedding(torch.tensor([0.0, 0.5, 1.0]), 16)
        assert not torch.allclose(embedded[0], embedded[1])
        assert not torch.allclose(embedded[1], embedded[2])

    def test_t_zero_is_the_canonical_pattern(self) -> None:
        from fm_fewshot.services.flow.velocity_mlp import sinusoidal_time_embedding

        embedded = sinusoidal_time_embedding(torch.zeros(1), 8)
        # sin(0) = 0 for the first half, cos(0) = 1 for the second.
        assert torch.allclose(embedded[0, :4], torch.zeros(4), atol=1e-6)
        assert torch.allclose(embedded[0, 4:], torch.ones(4), atol=1e-6)
        assert math.isclose(float(embedded.sum()), 4.0, abs_tol=1e-5)
