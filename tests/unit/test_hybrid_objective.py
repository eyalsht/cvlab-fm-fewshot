"""The hybrid objective: Strategy 1's decision plus Strategy 2's path (extra).

    L = CE(W zhat_T + b, y) + mu * L_CFM(v; z -> zhat')

**This is not in the write-up.** He invites alternative schemes under his scope
clause and this is one of ours, so it is an `ablations/` line under ADR-016 and
never a graded row. It exists because his two strategies supervise different
things: Strategy 1 scores only where the rollout ends, Strategy 2 scores the
whole path but never optimizes the decision. The hybrid asks whether they are
complements rather than alternatives.

What the tests hold it to is what a combined objective has to satisfy to be
readable at all: at `mu = 0` it must be the plain cross-entropy **exactly**, so
every existing Strategy 1 row is untouched by the term existing, and at
`mu > 0` the addition must be the flow matching loss on the classifier-guided
coupling and nothing else.
"""

import pytest
import torch
from test_head_contract import make_config

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.training.classifier_guided import (
    GuidedTargetConfig,
    guided_target,
)
from fm_fewshot.services.flow.training.rolled_out_ce import (
    RolledOutCeConfig,
    rolled_out_ce_loss,
    train_rolled_out_ce_field,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import make_head


def problem(n_classes: int = 3, dim: int = 5, per_class: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 4.0
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    return centers[labels] + 0.3 * torch.randn(labels.shape[0], dim, generator=g), labels


def classifier(n_classes: int = 3, dim: int = 5, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_classes, dim, generator=g), torch.randn(n_classes, generator=g)


def field(dim: int = 5, seed: int = 0, zero: bool = False) -> VelocityMLP:
    return VelocityMLP(dim=dim, hidden_dims=(16, 16), seed=seed, zero_output_init=zero)


class TestTheTermIsOffByDefault:
    def test_mu_defaults_to_zero(self) -> None:
        assert RolledOutCeConfig().mu == 0.0

    def test_mu_zero_reproduces_the_plain_cross_entropy_exactly(self) -> None:
        """The guard on every Strategy 1 row: adding the term changes nothing."""
        train_x, train_y = problem()
        weight, bias = classifier()
        net = field()
        plain = rolled_out_ce_loss(net, train_x, train_y, weight, bias, sample_steps=4)
        with_term = rolled_out_ce_loss(
            net, train_x, train_y, weight, bias, sample_steps=4, mu=0.0
        )
        assert torch.equal(plain, with_term)

    def test_mu_zero_consumes_no_randomness(self) -> None:
        """The flow matching term draws t; at mu = 0 it must not draw at all,
        or a Strategy 1 fit would change the moment the term existed."""
        train_x, train_y = problem()
        weight, bias = classifier()
        generator = torch.Generator().manual_seed(0)
        before = generator.get_state()
        rolled_out_ce_loss(
            field(), train_x, train_y, weight, bias, sample_steps=4, mu=0.0,
            generator=generator,
        )
        assert torch.equal(generator.get_state(), before)


class TestTheAddedTermIsTheFlowMatchingLoss:
    def test_the_difference_is_exactly_mu_times_the_cfm_loss(self) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()
        net = field()
        coupling = GuidedTargetConfig()

        plain = rolled_out_ce_loss(net, train_x, train_y, weight, bias, sample_steps=4)
        combined = rolled_out_ce_loss(
            net, train_x, train_y, weight, bias, sample_steps=4, mu=0.5,
            coupling=coupling, generator=torch.Generator().manual_seed(7),
        )
        # Recompute the term the same way, from the same seed.
        target = guided_target(
            net, train_x, train_y, weight, bias, sample_steps=4, config=coupling
        )
        t = torch.rand(train_x.shape[0], generator=torch.Generator().manual_seed(7))
        expected = 0.5 * cfm_loss(net, train_x, target, t)
        assert float((combined - plain - expected).abs()) < 1e-6

    def test_the_term_is_non_negative(self) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()
        net = field()
        plain = rolled_out_ce_loss(net, train_x, train_y, weight, bias, sample_steps=4)
        combined = rolled_out_ce_loss(
            net, train_x, train_y, weight, bias, sample_steps=4, mu=1.0,
            generator=torch.Generator().manual_seed(0),
        )
        assert float(combined.detach()) >= float(plain.detach())

    def test_the_coupling_carries_no_gradient(self) -> None:
        """The target is data for the flow matching term, as it is in Strategy 2."""
        train_x, train_y = problem()
        weight, bias = classifier()
        target = guided_target(
            field(), train_x, train_y, weight, bias, sample_steps=4,
            config=GuidedTargetConfig(),
        )
        assert target.requires_grad is False


class TestTraining:
    def test_the_hybrid_fit_is_deterministic_at_a_fixed_seed(self) -> None:
        def fit():
            train_x, train_y = problem()
            weight, bias = classifier()
            _, history = train_rolled_out_ce_field(
                train_x, train_y, weight, bias, None, sample_steps=4,
                hidden_dims=(16, 16), n_train_steps=15, batch_size=8, lr=1e-2,
                init_seed=0, mu=0.5,
            )
            return history
        assert fit() == fit()

    def test_the_hybrid_loss_falls_on_a_seeded_toy(self) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()
        _, history = train_rolled_out_ce_field(
            train_x, train_y, weight, bias, None, sample_steps=4,
            hidden_dims=(16, 16), n_train_steps=60, batch_size=8, lr=1e-2,
            init_seed=0, mu=0.5,
        )
        assert min(history[-10:]) < history[0]

    def test_mu_zero_trains_the_same_field_as_before_the_term_existed(self) -> None:
        """Bit-identical, not merely close: this is what keeps the graded rows."""
        def fit(**extra):
            train_x, train_y = problem()
            weight, bias = classifier()
            net, _ = train_rolled_out_ce_field(
                train_x, train_y, weight, bias, None, sample_steps=4,
                hidden_dims=(16, 16), n_train_steps=10, batch_size=8, lr=1e-2,
                init_seed=0, **extra,
            )
            return [p.detach().clone() for p in net.parameters()]
        for a, b in zip(fit(), fit(mu=0.0), strict=True):
            assert torch.equal(a, b)


class TestTheStaleCouplingIsRefused:
    def test_a_cached_coupling_is_refused_for_the_hybrid(self) -> None:
        """`target_every > 1` caches per row and the hybrid computes per batch,
        so accepting it would silently ignore the cadence that was asked for."""
        train_x, train_y = problem()
        weight, bias = classifier()
        with pytest.raises(ValueError, match="target_every"):
            rolled_out_ce_loss(
                field(), train_x, train_y, weight, bias, sample_steps=4, mu=0.5,
                coupling=GuidedTargetConfig(target_every=50),
                generator=torch.Generator().manual_seed(0),
            )


class TestTheHead:
    def test_the_head_reads_mu_and_the_coupling(self) -> None:
        cfg = make_config(
            "fm_prelinear_ce", sample_steps=4, n_train_steps=2, hidden_dims=[8],
            eval_every=0, probe_params={"max_epochs": 5},
            mu=0.25, target_step="normalized", rho=0.15,
        )
        head = make_head(cfg, 3)
        assert head.ce_config.mu == 0.25
        assert head.ce_config.coupling.target_step == "normalized"
        assert head.ce_config.coupling.rho == 0.15

    def test_the_graded_default_carries_no_hybrid_term(self) -> None:
        cfg = make_config(
            "fm_prelinear_ce", sample_steps=4, n_train_steps=2, hidden_dims=[8],
            eval_every=0, probe_params={"max_epochs": 5},
        )
        assert make_head(cfg, 3).ce_config.mu == 0.0
