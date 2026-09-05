"""Velocity network v_theta(z, t) per PRD_flow_matching_block.

Moved out of test_fm_objective when Stage 2 gave the network a second time
conditioning. The write-up asks for a scalar t concatenated to the feature and
two hidden layers of width 512; our sinusoidal embedding survives as the
ablation (ADR-019), so both paths are tested here side by side.
"""

import hashlib
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


class TestZeroOutputInit:
    """ADR-030: the untrained Stage 3 system must be the linear probe exactly.

    Zeroing the output layer makes `v_theta(z, t) = 0` everywhere, so T Euler
    steps leave every feature where it started and `W zhat + b` is `W z + b`.
    The write-up asks for "close to identity"; this is identity, which a test
    can assert instead of an approximation someone has to eyeball.

    The flag defaults to False and the False path must stay bit-identical to
    every Stage 2 field ever fitted, so the digests below are pinned rather
    than compared against a second construction: a change to the seeded init
    would move both sides of a self-comparison together and pass.
    """

    # sha256 over the sorted state dict, computed before zero_output_init existed.
    SCALAR_DIGEST = "c6ce8e2057e781d12f69c806de9d184f1d79705746bb09c6a65f45016e1fa0ea"
    SINUSOIDAL_DIGEST = "13dec0530c278c4240ace48e31b348a89a256f5523e6198908bcaadaad9ab3b5"

    @staticmethod
    def _digest(net: VelocityMLP) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(net.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def test_the_flag_defaults_to_off(self) -> None:
        assert VelocityMLP(dim=4, hidden_dims=(8,), seed=0).zero_output_init is False

    @pytest.mark.parametrize("dim", [1, 8, 33])
    def test_the_field_is_zero_for_every_input(self, dim: int) -> None:
        net = VelocityMLP(dim=dim, hidden_dims=(16, 16), seed=0, zero_output_init=True)
        g = torch.Generator().manual_seed(11)
        z = torch.randn(7, dim, generator=g) * 100.0
        t = torch.rand(7, generator=g)
        assert torch.equal(net(z, t), torch.zeros(7, dim))

    def test_it_is_zero_under_the_sinusoidal_conditioning_too(self) -> None:
        net = VelocityMLP(
            dim=6,
            hidden_dims=(16,),
            time_conditioning="sinusoidal",
            seed=2,
            zero_output_init=True,
        )
        z, t = torch.randn(4, 6), torch.rand(4)
        assert torch.equal(net(z, t), torch.zeros(4, 6))

    def test_only_the_output_layer_is_zeroed(self) -> None:
        """The hidden layers keep their seeded values, so the first step moves.

        A network zeroed throughout would have no gradient anywhere and would
        never leave the identity. The output layer's gradient is delta (x) h
        with h the penultimate activation, which is nonzero here.
        """
        net = VelocityMLP(dim=5, hidden_dims=(16, 16), seed=0, zero_output_init=True)
        linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
        assert torch.equal(linears[-1].weight, torch.zeros_like(linears[-1].weight))
        assert torch.equal(linears[-1].bias, torch.zeros_like(linears[-1].bias))
        for hidden in linears[:-1]:
            assert hidden.weight.abs().sum() > 0
            assert hidden.bias.abs().sum() > 0

    def test_the_zeroed_field_trains_off_zero_in_one_step(self) -> None:
        net = VelocityMLP(dim=5, hidden_dims=(16, 16), seed=0, zero_output_init=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        z, t = torch.randn(8, 5), torch.rand(8)
        loss = ((net(z, t) - torch.ones(8, 5)) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert net(z, t).abs().sum() > 0

    def test_the_default_path_reproduces_the_stage_2_initialization(self) -> None:
        net = VelocityMLP(dim=8, hidden_dims=(16, 16), time_conditioning="scalar", seed=0)
        assert self._digest(net) == self.SCALAR_DIGEST

    def test_the_default_path_is_unchanged_under_sinusoidal_conditioning(self) -> None:
        net = VelocityMLP(
            dim=8, hidden_dims=(16, 16), time_conditioning="sinusoidal", seed=3
        )
        assert self._digest(net) == self.SINUSOIDAL_DIGEST

    def test_zeroing_consumes_the_generator_identically(self) -> None:
        """The flag changes the output layer, not the random stream.

        Every layer is drawn first and the output layer zeroed afterwards, so
        the hidden weights of a zeroed field are the hidden weights of the
        Stage 2 field at the same seed. Anything else would make ADR-030 a
        silent reinitialization of the whole network.
        """
        plain = VelocityMLP(dim=6, hidden_dims=(16, 16), seed=5)
        zeroed = VelocityMLP(dim=6, hidden_dims=(16, 16), seed=5, zero_output_init=True)
        plain_linears = [m for m in plain.net if isinstance(m, torch.nn.Linear)]
        zeroed_linears = [m for m in zeroed.net if isinstance(m, torch.nn.Linear)]
        for a, b in zip(plain_linears[:-1], zeroed_linears[:-1], strict=True):
            assert torch.equal(a.weight, b.weight)
            assert torch.equal(a.bias, b.bias)
