"""Velocity network v_theta(z, t) per PRD_flow_matching_block.

Moved out of test_fm_objective when Stage 2 gave the network a second time
conditioning. The write-up asks for a scalar t concatenated to the feature and
two hidden layers of width 512; our sinusoidal embedding survives as the
ablation (ADR-019), so both paths are tested here side by side.
"""

import math

import pytest
import torch

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP, sinusoidal_time_embedding


class TestDefaults:
    def test_the_write_ups_architecture_is_the_default(self) -> None:
        """His suggestion, not ours, is what runs unless a config says otherwise."""
        net = VelocityMLP(dim=8, seed=0)
        assert net.time_conditioning == "scalar"
        widths = [m.out_features for m in net.net if isinstance(m, torch.nn.Linear)]
        assert widths == [512, 512, 8]

    def test_unknown_time_conditioning_raises(self) -> None:
        with pytest.raises(ValueError, match="time_conditioning"):
            VelocityMLP(dim=4, time_conditioning="learned", seed=0)


class TestScalarConditioning:
    def test_output_shape_equals_input_shape(self) -> None:
        net = VelocityMLP(dim=6, hidden_dims=(16, 16), time_conditioning="scalar", seed=0)
        out = net(torch.randn(5, 6), torch.rand(5))
        assert out.shape == (5, 6)
        assert torch.isfinite(out).all()

    def test_time_is_concatenated_as_one_column(self) -> None:
        net = VelocityMLP(dim=6, hidden_dims=(16,), time_conditioning="scalar", seed=0)
        first = next(m for m in net.net if isinstance(m, torch.nn.Linear))
        assert first.in_features == 7

    @pytest.mark.parametrize("dim", [384, 512])
    def test_parameter_count_matches_the_suggested_net(self, dim: int) -> None:
        net = VelocityMLP(dim=dim, hidden_dims=(512, 512), time_conditioning="scalar", seed=0)
        expected = (
            (dim + 1) * 512 + 512  # input layer, feature plus scalar t
            + 512 * 512 + 512  # second hidden layer
            + 512 * dim + dim  # output layer, back to the feature dimension
        )
        assert sum(p.numel() for p in net.parameters()) == expected

    def test_same_seed_gives_identical_initialization(self) -> None:
        kwargs = {"dim": 4, "hidden_dims": (8,), "time_conditioning": "scalar", "seed": 7}
        x, t = torch.randn(3, 4), torch.rand(3)
        assert torch.equal(VelocityMLP(**kwargs)(x, t), VelocityMLP(**kwargs)(x, t))

    def test_different_seeds_differ(self) -> None:
        x, t = torch.randn(3, 4), torch.rand(3)
        a = VelocityMLP(dim=4, hidden_dims=(8,), time_conditioning="scalar", seed=0)
        b = VelocityMLP(dim=4, hidden_dims=(8,), time_conditioning="scalar", seed=1)
        assert not torch.equal(a(x, t), b(x, t))

    def test_a_trained_scalar_field_is_not_time_invariant(self) -> None:
        """The risk ADR-019 names: one scalar among D dimensions can be ignored.

        A field that ignores t still trains and still transports, so nothing
        else in the suite would catch it. Fit a target that genuinely depends
        on t, then demand the fitted field separate two times on one x.
        """
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        x0 = torch.randn(64, 3, generator=g)
        x1 = torch.randn(64, 3, generator=g)
        net = VelocityMLP(dim=3, hidden_dims=(64, 64), time_conditioning="scalar", seed=0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(300):
            loss = cfm_loss(net, x0, x1, torch.rand(64))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x = torch.randn(16, 3, generator=g)
        with torch.no_grad():
            early = net(x, torch.zeros(16))
            late = net(x, torch.ones(16))
        assert float((early - late).abs().max()) > 1e-3


class TestSinusoidalConditioning:
    """The Phase 7 path, unchanged. It is the ADR-019 ablation now."""

    def test_output_shape_equals_input_shape(self) -> None:
        net = VelocityMLP(
            dim=6, hidden_dims=(16, 16), time_conditioning="sinusoidal", time_embed_dim=8, seed=0
        )
        assert net(torch.randn(5, 6), torch.rand(5)).shape == (5, 6)

    def test_time_embedding_widens_the_input_layer(self) -> None:
        net = VelocityMLP(
            dim=6, hidden_dims=(16,), time_conditioning="sinusoidal", time_embed_dim=8, seed=0
        )
        first = next(m for m in net.net if isinstance(m, torch.nn.Linear))
        assert first.in_features == 14

    def test_same_seed_gives_identical_initialization(self) -> None:
        kwargs = {
            "dim": 4,
            "hidden_dims": (8,),
            "time_conditioning": "sinusoidal",
            "time_embed_dim": 4,
            "seed": 7,
        }
        x, t = torch.randn(3, 4), torch.rand(3)
        assert torch.equal(VelocityMLP(**kwargs)(x, t), VelocityMLP(**kwargs)(x, t))

    def test_time_actually_changes_the_output(self) -> None:
        net = VelocityMLP(
            dim=3, hidden_dims=(16,), time_conditioning="sinusoidal", time_embed_dim=8, seed=0
        )
        x = torch.randn(4, 3)
        assert not torch.allclose(net(x, torch.zeros(4)), net(x, torch.ones(4)))

    def test_odd_time_embed_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="even"):
            VelocityMLP(dim=3, hidden_dims=(8,), time_embed_dim=7, seed=0)


class TestSinusoidalEmbedding:
    def test_embedding_is_bounded_and_finite(self) -> None:
        embedded = sinusoidal_time_embedding(torch.linspace(0.0, 1.0, 17), 16)
        assert embedded.shape == (17, 16)
        assert torch.isfinite(embedded).all()
        assert float(embedded.abs().max()) <= 1.0 + 1e-6

    def test_distinct_times_get_distinct_embeddings(self) -> None:
        embedded = sinusoidal_time_embedding(torch.tensor([0.0, 0.5, 1.0]), 16)
        assert not torch.allclose(embedded[0], embedded[1])
        assert not torch.allclose(embedded[1], embedded[2])

    def test_t_zero_is_the_canonical_pattern(self) -> None:
        embedded = sinusoidal_time_embedding(torch.zeros(1), 8)
        # sin(0) = 0 for the first half, cos(0) = 1 for the second.
        assert torch.allclose(embedded[0, :4], torch.zeros(4), atol=1e-6)
        assert torch.allclose(embedded[0, 4:], torch.ones(4), atol=1e-6)
        assert math.isclose(float(embedded.sum()), 4.0, abs_tol=1e-5)
