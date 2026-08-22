"""Standard FM head per the Stage 2 write-up, section "Standard FM training".

The head is a classifier and a describable one: it replaces
argmax_c cos(z, p_c) with argmax_c cos(psi(z), p_c), where psi is T Euler steps
of one weight-tied MLP. These tests pin the three things that makes true: the
field is trained simulation-free on (z_i -> p_{y_i}) pairs, inference is exactly
the write-up's Euler grid, and the decision is the Stage 1 prototype rule
unchanged.
"""

from dataclasses import replace

import pytest
import torch
from test_head_contract import make_config, run_head_battery, separable_scenario

from fm_fewshot.services.flow.toy import make_toy_problem
from fm_fewshot.services.flow.training.standard import train_standard_field, transport
from fm_fewshot.services.heads.fm_standard import FmStandardHead
from fm_fewshot.services.heads.prototype import PrototypeHead


def separable_problem(
    n_classes: int = 3, dim: int = 8, per_class: int = 6, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """Well-separated isotropic classes: a problem the prototype rule already solves."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 6.0
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    train_x = centers[labels] + 0.35 * torch.randn(labels.shape[0], dim, generator=g)
    query_labels = torch.arange(n_classes).repeat_interleave(4)
    query_x = centers[query_labels] + 0.35 * torch.randn(query_labels.shape[0], dim, generator=g)
    return train_x, labels, query_x, query_labels


def fast_config(**overrides: object):
    params: dict[str, object] = {
        "sample_steps": 4,
        "n_train_steps": 200,
        "hidden_dims": [32, 32],
        "batch_size": 16,
    }
    params.update(overrides)
    return make_config("fm_standard", **params)


class TestContract:
    def test_battery(self) -> None:
        run_head_battery(fast_config(), 3, *separable_scenario())

    @pytest.mark.parametrize("sample_steps", [4, 12])
    def test_logits_are_finite_at_both_step_counts(self, sample_steps: int) -> None:
        train_x, train_y, query_x, _ = separable_problem()
        head = FmStandardHead.from_context(
            fast_config(sample_steps=sample_steps), 3, context=None
        )
        head.fit(train_x, train_y, train_x, train_y)
        assert torch.isfinite(head.predict(query_x)).all()

    def test_same_seeds_give_bit_identical_logits(self) -> None:
        train_x, train_y, query_x, _ = separable_problem()
        cfg = fast_config()
        first = FmStandardHead.from_context(cfg, 3, context=None)
        second = FmStandardHead.from_context(cfg, 3, context=None)
        first.fit(train_x, train_y, train_x, train_y)
        second.fit(train_x, train_y, train_x, train_y)
        assert torch.equal(first.predict(query_x), second.predict(query_x))

    def test_a_different_init_seed_gives_a_different_field(self) -> None:
        train_x, train_y, query_x, _ = separable_problem()
        cfg = fast_config()
        other = replace(cfg, init_seed=1)
        a = FmStandardHead.from_context(cfg, 3, context=None)
        b = FmStandardHead.from_context(other, 3, context=None)
        a.fit(train_x, train_y, train_x, train_y)
        b.fit(train_x, train_y, train_x, train_y)
        assert not torch.equal(a.predict(query_x), b.predict(query_x))


class TestIdentityReduction:
    def test_a_zero_field_reproduces_the_prototype_head_exactly(self) -> None:
        """psi is the identity when v_theta = 0, so the head must be Stage 1 here.

        Bit-identical, not close: the decision rule is the prototype head's own
        predict, called on the transported features (ADR-020). Any drift means
        a second copy of the rule was written somewhere.
        """
        train_x, train_y, query_x, _ = separable_problem()
        head = FmStandardHead.from_context(fast_config(n_train_steps=0), 3, context=None)
        head.fit(train_x, train_y, train_x, train_y)
        with torch.no_grad():
            for parameter in head.field.parameters():
                parameter.zero_()

        baseline = PrototypeHead(n_classes=3)
        baseline.fit(train_x, train_y, train_x, train_y)
        assert torch.equal(head.predict(query_x), baseline.predict(query_x))

    def test_the_prototypes_are_the_stage_1_prototypes(self) -> None:
        train_x, train_y, _, _ = separable_problem()
        head = FmStandardHead.from_context(fast_config(n_train_steps=0), 3, context=None)
        head.fit(train_x, train_y, train_x, train_y)
        baseline = PrototypeHead(n_classes=3)
        baseline.fit(train_x, train_y, train_x, train_y)
        assert torch.equal(head.prototypes, baseline.prototypes)


class TestEulerGrid:
    @pytest.mark.parametrize("sample_steps", [4, 12])
    def test_the_field_is_called_on_the_write_ups_grid(self, sample_steps: int) -> None:
        """zhat_{k+1} = zhat_k + (1/T) v(zhat_k, k/T), k = 0 .. T-1."""
        seen: list[float] = []

        def recorder(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            seen.append(float(t[0]))
            return torch.zeros_like(x)

        transport(recorder, torch.zeros(2, 3), sample_steps=sample_steps)
        assert len(seen) == sample_steps
        expected = [k / sample_steps for k in range(sample_steps)]
        assert seen == pytest.approx(expected)

    def test_one_step_of_a_constant_field_is_one_over_t(self) -> None:
        constant = lambda x, t: torch.ones_like(x)  # noqa: E731
        landed = transport(constant, torch.zeros(1, 2), sample_steps=4)
        # Four steps of size 1/4 with unit velocity land on 1.0 exactly.
        assert torch.allclose(landed, torch.ones(1, 2))


class TestTraining:
    def test_the_cfm_loss_decreases(self) -> None:
        train_x, train_y, _, _ = separable_problem()
        prototypes = PrototypeHead(n_classes=3)
        prototypes.fit(train_x, train_y, train_x, train_y)
        _, history = train_standard_field(
            train_x,
            prototypes.prototypes[train_y],
            hidden_dims=(32, 32),
            n_train_steps=300,
            batch_size=16,
            init_seed=0,
        )
        assert len(history) == 300
        early = sum(history[:20]) / 20
        late = sum(history[-20:]) / 20
        assert late < early

    def test_the_field_learns_the_single_pair_target(self) -> None:
        """One pair: the conditional optimum is the constant field x1 - x0."""
        x0 = torch.tensor([[0.0, 0.0]])
        x1 = torch.tensor([[1.0, 2.0]])
        field, _ = train_standard_field(
            x0, x1, hidden_dims=(64, 64), n_train_steps=600, batch_size=32, lr=1e-2, init_seed=0
        )
        with torch.no_grad():
            landed = transport(field, x0, sample_steps=12)
        assert float((landed - x1).norm()) < 0.1

    def test_training_ignores_sample_steps(self) -> None:
        """ADR-023 in miniature: standard training never solves the ODE.

        T=4 and T=12 are therefore the same field read at two resolutions, and
        the phase note must not report them as two models.
        """
        train_x, train_y, query_x, _ = separable_problem()
        four = FmStandardHead.from_context(fast_config(sample_steps=4), 3, context=None)
        twelve = FmStandardHead.from_context(fast_config(sample_steps=12), 3, context=None)
        four.fit(train_x, train_y, train_x, train_y)
        twelve.fit(train_x, train_y, train_x, train_y)
        for a, b in zip(four.field.parameters(), twelve.field.parameters(), strict=True):
            assert torch.equal(a, b)
        assert not torch.equal(four.predict(query_x), twelve.predict(query_x))


class TestTransportGeometry:
    def test_transport_lands_training_points_on_their_own_prototype(self) -> None:
        """A fit check on the pairs the loss saw, not a claim about test points.

        Whether transport helps on unseen features is the question Stage 2
        exists to answer, so it is measured in the grid and not asserted here
        (testing policy: invariants, not accuracy numbers).

        The probe is the endpoint distance, not cosine. These features have
        norm 6 and the prototypes are unit norm, so cosine to the right
        prototype starts near 0.99 and has nowhere to go; the distance the
        objective actually minimizes starts at 5 and is the quantity that
        moves. That gap between the two measures is the ADR-018 raw-scale
        concern showing up at toy scale.
        """
        train_x, train_y, _, _ = separable_problem()
        head = FmStandardHead.from_context(
            fast_config(n_train_steps=1500, hidden_dims=[128, 128]), 3, context=None
        )
        head.fit(train_x, train_y, train_x, train_y)

        targets = head.prototypes[train_y]
        before = float((train_x - targets).norm(dim=1).mean())
        after = float((head.transport(train_x) - targets).norm(dim=1).mean())
        assert after < 0.1 * before

    def test_the_trajectory_starts_at_the_query_and_ends_where_transport_lands(self) -> None:
        train_x, train_y, query_x, _ = separable_problem()
        head = FmStandardHead.from_context(fast_config(), 3, context=None)
        head.fit(train_x, train_y, train_x, train_y)
        states = head.trajectory(query_x)
        assert states.shape == (5, query_x.shape[0], query_x.shape[1])
        assert torch.equal(states[0], query_x)
        assert torch.equal(states[-1], head.transport(query_x))


class TestAnisotropicToy:
    """The Phase 7 toy, now driven through the head rather than the bare field.

    The stretched regime is the one that says something: classes that are
    linearly separable but not compact around their means, which is the
    geometry the Stage 1 diagnosis found in real DINOv2 features. The round
    control is there so a passing stretched case cannot be read as transport
    helping everywhere.
    """

    @staticmethod
    def _accuracy(head, x: torch.Tensor, y: torch.Tensor) -> float:
        return float((head.predict(x).argmax(dim=1) == y).float().mean())

    def _both_heads(self, anisotropy: float):
        problem = make_toy_problem(seed=0, anisotropy=anisotropy)
        baseline = PrototypeHead(n_classes=problem.n_classes)
        baseline.fit(problem.train_x, problem.train_y, problem.test_x, problem.test_y)
        head = FmStandardHead(
            problem.n_classes,
            sample_steps=4,
            n_train_steps=800,
            batch_size=128,
            lr=1e-2,
            hidden_dims=(64, 64),
            init_seed=0,
        )
        head.fit(problem.train_x, problem.train_y, problem.test_x, problem.test_y)
        return problem, baseline, head

    def test_transport_beats_the_prototype_rule_on_stretched_classes(self) -> None:
        problem, baseline, head = self._both_heads(anisotropy=3.0)
        before = self._accuracy(baseline, problem.test_x, problem.test_y)
        after = self._accuracy(head, problem.test_x, problem.test_y)
        assert before < 0.99, f"toy is too easy to be informative: {before}"
        assert after > before

    def test_transport_has_nothing_to_recover_on_round_classes(self) -> None:
        problem, baseline, head = self._both_heads(anisotropy=1.0)
        before = self._accuracy(baseline, problem.test_x, problem.test_y)
        after = self._accuracy(head, problem.test_x, problem.test_y)
        assert before > 0.99, f"the control should already be solved: {before}"
        assert after == before
