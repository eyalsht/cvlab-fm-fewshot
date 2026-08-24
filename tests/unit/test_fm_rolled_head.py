"""Rolled-out FM head per the Stage 2 write-up, section "Inference and rolled-out training".

Rolled-out training runs the inference solver on the training features and
scores only where they land:

    zhat_{k+1} = zhat_k + (1 / T) v_theta(zhat_k, k / T),   k = 0 .. T - 1
    L_roll = mean_i || zhat_T - p_{y_i} ||^2

with the gradient carried back through all T velocity predictions. Nothing else
is in the objective. There is no time sampling, no interpolation, no
cross-entropy, no temperature and no CFM term (ADR-021), and training depth is
inference depth, so `sample_steps` carries both meanings and a config that
tries to separate them is refused (ADR-022).

These tests pin that objective, the depth of the graph it builds, and the
memory that depth costs, which is the whole reason the scheme is interesting
enough to compare against standard training.
"""

from dataclasses import replace

import pytest
import torch
from test_head_contract import make_config, run_head_battery, separable_scenario

from fm_fewshot.services.flow.solver import solve_ode
from fm_fewshot.services.flow.training.rolled_out import (
    rolled_out_loss,
    train_rolled_out_field,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import make_head
from fm_fewshot.services.heads.fm_rolled import FmRolledHead
from fm_fewshot.services.heads.prototype import PrototypeHead


def separable_problem(
    n_classes: int = 3, dim: int = 8, per_class: int = 6, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """Well-separated isotropic classes, the same toy the standard head is fitted on."""
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
        "n_train_steps": 150,
        "hidden_dims": [32, 32],
        "batch_size": 16,
    }
    params.update(overrides)
    return make_config("fm_rolled", **params)


def small_field(dim: int = 4, seed: int = 0) -> VelocityMLP:
    return VelocityMLP(dim=dim, hidden_dims=(16, 16), seed=seed)


def retained_bytes(build_loss) -> int:
    """Bytes of activation held for the backward pass while build_loss runs.

    A CPU-usable stand-in for peak training memory, and a more direct one than
    a process-level reading: what rolled-out training pays for depth is exactly
    the tensors the graph saves for backward, and this counts them. Runs on the
    development box and on CI alike, where torch.cuda.max_memory_allocated
    would only work on one of the two.
    """
    total = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        total += tensor.numel() * tensor.element_size()
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        build_loss()
    return total


class VelocityRecorder:
    """Wraps a field and keeps every prediction so its gradient can be read back."""

    def __init__(self, field) -> None:
        self.field = field
        self.times: list[float] = []
        self.velocities: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        velocity = self.field(x, t)
        velocity.retain_grad()
        self.times.append(float(t[0]))
        self.velocities.append(velocity)
        return velocity


class TestContract:
    def test_battery(self) -> None:
        run_head_battery(fast_config(), 3, *separable_scenario())

    @pytest.mark.parametrize("sample_steps", [4, 12])
    def test_logits_are_finite_at_both_step_counts(self, sample_steps: int) -> None:
        train_x, train_y, query_x, _ = separable_problem()
        head = FmRolledHead.from_context(fast_config(sample_steps=sample_steps), 3, context=None)
        head.fit(train_x, train_y, train_x, train_y)
        assert torch.isfinite(head.predict(query_x)).all()

    def test_the_head_exposes_the_standard_heads_surface(self) -> None:
        """Same attributes, so the loop, the figures and the diagnostics need no branch."""
        train_x, train_y, query_x, _ = separable_problem()
        head = FmRolledHead.from_context(fast_config(), 3, context=None)
        head.fit(train_x, train_y, train_x, train_y)

        assert isinstance(head.field, VelocityMLP)
        assert head.prototypes.shape == (3, train_x.shape[1])
        assert len(head.loss_history) == 150
        assert head.transport(query_x).shape == query_x.shape
        states = head.trajectory(query_x)
        assert states.shape == (5, query_x.shape[0], query_x.shape[1])
        assert torch.equal(states[0], query_x)
        assert torch.equal(states[-1], head.transport(query_x))

    def test_the_decision_is_the_stage_1_prototype_rule(self) -> None:
        """Delegated, not reimplemented (ADR-020): equal, not close."""
        train_x, train_y, query_x, _ = separable_problem()
        head = FmRolledHead.from_context(fast_config(), 3, context=None)
        head.fit(train_x, train_y, train_x, train_y)
        baseline = PrototypeHead(n_classes=3)
        baseline.fit(train_x, train_y, train_x, train_y)

        assert torch.equal(head.prototypes, baseline.prototypes)
        assert torch.equal(head.predict(query_x), baseline.predict(head.transport(query_x)))


class TestDeterminism:
    def test_the_same_seed_pair_gives_bit_identical_logits(self) -> None:
        train_x, train_y, query_x, _ = separable_problem()
        cfg = fast_config()
        first = FmRolledHead.from_context(cfg, 3, context=None)
        second = FmRolledHead.from_context(cfg, 3, context=None)
        first.fit(train_x, train_y, train_x, train_y)
        second.fit(train_x, train_y, train_x, train_y)
        assert torch.equal(first.predict(query_x), second.predict(query_x))
        assert first.loss_history == second.loss_history

    def test_a_different_init_seed_gives_a_different_field(self) -> None:
        """The two streams are separate: init_seed moves the fit, subset_seed does not."""
        train_x, train_y, query_x, _ = separable_problem()
        cfg = fast_config()
        a = FmRolledHead.from_context(cfg, 3, context=None)
        b = FmRolledHead.from_context(replace(cfg, init_seed=1), 3, context=None)
        same_subset = FmRolledHead.from_context(replace(cfg, subset_seed=7), 3, context=None)
        for head in (a, b, same_subset):
            head.fit(train_x, train_y, train_x, train_y)

        assert not torch.equal(a.predict(query_x), b.predict(query_x))
        assert torch.equal(a.predict(query_x), same_subset.predict(query_x))


class TestObjective:
    def test_the_loss_is_the_mean_squared_endpoint_distance(self) -> None:
        """Hand batch: a constant unit field, so zhat_T = x0 + 1 for any T."""
        constant = lambda x, t: torch.ones_like(x)  # noqa: E731
        x0 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        # zhat_T = [[1, 1], [2, 2]]; squared distances 0 + 1 = 1 and 4 + 4 = 8.
        expected = (1.0 + 8.0) / 2.0
        for sample_steps in (1, 2, 5):
            loss = rolled_out_loss(constant, x0, targets, sample_steps=sample_steps)
            assert float(loss) == pytest.approx(expected)

    def test_the_norm_is_summed_over_the_feature_dimension(self) -> None:
        """Not a per-element mean: ||.||^2 sums over D, and only the batch is averaged."""
        constant = lambda x, t: torch.zeros_like(x)  # noqa: E731
        x0 = torch.zeros(1, 4)
        targets = torch.ones(1, 4)
        assert float(rolled_out_loss(constant, x0, targets, sample_steps=3)) == pytest.approx(4.0)

    def test_t_equals_one_is_one_euler_step_done_by_hand(self) -> None:
        field = small_field()
        x0 = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
        targets = torch.randn(5, 4, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            landed = x0 + field(x0, torch.zeros(5))
            expected = ((landed - targets) ** 2).sum(dim=1).mean()
        loss = rolled_out_loss(field, x0, targets, sample_steps=1)
        assert torch.allclose(loss, expected)

    def test_the_loss_matches_the_solver_endpoint_at_larger_t(self) -> None:
        field = small_field()
        x0 = torch.randn(5, 4, generator=torch.Generator().manual_seed(3))
        targets = torch.randn(5, 4, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            endpoint = solve_ode(field, x0, n_steps=12, method="euler")
            expected = ((endpoint - targets) ** 2).sum(dim=1).mean()
        loss = rolled_out_loss(field, x0, targets, sample_steps=12)
        assert torch.allclose(loss, expected)

    def test_the_prototypes_carry_no_gradient(self) -> None:
        """The target is a fixed point of the space, never a thing being fitted."""
        field = small_field()
        x0 = torch.randn(5, 4, generator=torch.Generator().manual_seed(5))
        targets = torch.randn(5, 4, generator=torch.Generator().manual_seed(6))
        targets.requires_grad_(True)
        rolled_out_loss(field, x0, targets, sample_steps=4).backward()
        assert targets.grad is None

    def test_a_fitted_head_never_makes_its_prototypes_learnable(self) -> None:
        train_x, train_y, _, _ = separable_problem()
        head = FmRolledHead.from_context(fast_config(n_train_steps=5), 3, context=None)
        head.fit(train_x, train_y, train_x, train_y)
        assert not head.prototypes.requires_grad
        assert head.prototypes.grad is None


class TestGradientReach:
    @pytest.mark.parametrize("sample_steps", [1, 4, 12])
    def test_every_velocity_prediction_receives_gradient(self, sample_steps: int) -> None:
        """The point of the scheme: all T predictions are supervised by one endpoint."""
        field = small_field()
        recorder = VelocityRecorder(field)
        x0 = torch.randn(6, 4, generator=torch.Generator().manual_seed(7))
        targets = torch.randn(6, 4, generator=torch.Generator().manual_seed(8))

        rolled_out_loss(recorder, x0, targets, sample_steps=sample_steps).backward()

        assert len(recorder.velocities) == sample_steps
        assert recorder.times == pytest.approx([k / sample_steps for k in range(sample_steps)])
        for step, velocity in enumerate(recorder.velocities):
            assert velocity.grad is not None, f"step {step} left the graph"
            assert float(velocity.grad.abs().max()) > 0.0, f"step {step} received zero gradient"

    def test_every_parameter_receives_gradient(self) -> None:
        field = small_field()
        x0 = torch.randn(6, 4, generator=torch.Generator().manual_seed(9))
        targets = torch.randn(6, 4, generator=torch.Generator().manual_seed(10))
        rolled_out_loss(field, x0, targets, sample_steps=4).backward()
        for name, parameter in field.named_parameters():
            assert parameter.grad is not None, name
            assert float(parameter.grad.abs().max()) > 0.0, name

    def test_fewer_steps_builds_a_shorter_graph_and_a_different_gradient(self) -> None:
        x0 = torch.randn(6, 4, generator=torch.Generator().manual_seed(11))
        targets = torch.randn(6, 4, generator=torch.Generator().manual_seed(12))

        gradients = {}
        for sample_steps in (4, 12):
            field = small_field()
            recorder = VelocityRecorder(field)
            rolled_out_loss(recorder, x0, targets, sample_steps=sample_steps).backward()
            assert len(recorder.velocities) == sample_steps
            gradients[sample_steps] = [p.grad.clone() for p in field.parameters()]

        assert not any(
            torch.equal(a, b)
            for a, b in zip(gradients[4], gradients[12], strict=True)
        )

    def test_the_gradient_depends_on_the_earliest_step(self) -> None:
        """Truncating the first prediction from the graph changes the answer.

        Stated as detaching after step 0: if only the last step were supervised,
        this would leave the parameter gradient untouched.
        """
        field = small_field()
        x0 = torch.randn(6, 4, generator=torch.Generator().manual_seed(13))
        targets = torch.randn(6, 4, generator=torch.Generator().manual_seed(14))

        full = torch.autograd.grad(
            rolled_out_loss(field, x0, targets, sample_steps=4), list(field.parameters())
        )
        truncated_start = (x0 + 0.25 * field(x0, torch.zeros(6))).detach()
        truncated_loss = rolled_out_loss(
            _shifted(field, 0.25), truncated_start, targets, sample_steps=3
        )
        truncated = torch.autograd.grad(truncated_loss, list(field.parameters()))
        assert not all(torch.allclose(a, b) for a, b in zip(full, truncated, strict=True))


def _shifted(field, offset: float):
    """The same field read on a time grid shifted by offset, for the truncation probe."""
    return lambda x, t: field(x, t * 0.75 + offset)


class TestDepthIsNotSeparable:
    @pytest.mark.parametrize("key", ["n_rollout_steps", "rollout_steps", "train_sample_steps"])
    def test_a_config_separating_training_depth_from_sample_steps_is_refused(
        self, key: str
    ) -> None:
        """ADR-022: one T, used for both. Two values of T are two trained models."""
        cfg = make_config("fm_rolled", sample_steps=4, **{key: 4})
        with pytest.raises(ValueError, match="sample_steps"):
            make_head(cfg, 3)

    def test_the_refusal_holds_even_when_the_two_agree(self) -> None:
        """The field is gone, not merely constrained; accepting it would invite drift."""
        cfg = make_config("fm_rolled", sample_steps=12, n_rollout_steps=12)
        with pytest.raises(ValueError, match="sample_steps"):
            make_head(cfg, 3)

    def test_training_depth_is_the_configured_sample_steps(self) -> None:
        """Unlike standard training, T is read during the fit, so T fixes the model."""
        train_x, train_y, _, _ = separable_problem()
        four = FmRolledHead.from_context(fast_config(sample_steps=4), 3, context=None)
        twelve = FmRolledHead.from_context(fast_config(sample_steps=12), 3, context=None)
        four.fit(train_x, train_y, train_x, train_y)
        twelve.fit(train_x, train_y, train_x, train_y)
        assert not any(
            torch.equal(a, b)
            for a, b in zip(four.field.parameters(), twelve.field.parameters(), strict=True)
        )


class TestMemoryCost:
    def test_retained_activation_grows_with_t(self) -> None:
        """The cost the write-up trades accuracy against; monotone, and worth measuring."""
        field = small_field()
        x0 = torch.randn(8, 4, generator=torch.Generator().manual_seed(15))
        targets = torch.randn(8, 4, generator=torch.Generator().manual_seed(16))
        measured = [
            retained_bytes(
                lambda t=sample_steps: rolled_out_loss(field, x0, targets, sample_steps=t)
            )
            for sample_steps in (1, 4, 12)
        ]
        assert measured[0] < measured[1] < measured[2]

    def test_checkpointing_holds_less_than_the_plain_graph(self) -> None:
        field = small_field()
        x0 = torch.randn(8, 4, generator=torch.Generator().manual_seed(17))
        targets = torch.randn(8, 4, generator=torch.Generator().manual_seed(18))
        plain = retained_bytes(
            lambda: rolled_out_loss(field, x0, targets, sample_steps=12)
        )
        checkpointed = retained_bytes(
            lambda: rolled_out_loss(
                field, x0, targets, sample_steps=12, gradient_checkpointing=True
            )
        )
        assert checkpointed < plain


class TestCheckpointingParity:
    @pytest.mark.parametrize("sample_steps", [1, 4, 12])
    def test_checkpointing_changes_neither_the_loss_nor_the_gradient(
        self, sample_steps: int
    ) -> None:
        """Recomputation, not approximation. Bit-identical, or it is not a free knob."""
        x0 = torch.randn(6, 4, generator=torch.Generator().manual_seed(19))
        targets = torch.randn(6, 4, generator=torch.Generator().manual_seed(20))

        plain_field = small_field()
        plain_loss = rolled_out_loss(plain_field, x0, targets, sample_steps=sample_steps)
        plain_grad = torch.autograd.grad(plain_loss, list(plain_field.parameters()))

        saved_field = small_field()
        saved_loss = rolled_out_loss(
            saved_field, x0, targets, sample_steps=sample_steps, gradient_checkpointing=True
        )
        saved_grad = torch.autograd.grad(saved_loss, list(saved_field.parameters()))

        assert torch.equal(plain_loss, saved_loss)
        assert all(torch.equal(a, b) for a, b in zip(plain_grad, saved_grad, strict=True))

    def test_a_checkpointed_fit_reproduces_the_plain_fit(self) -> None:
        train_x, train_y, query_x, _ = separable_problem()
        plain = FmRolledHead.from_context(fast_config(), 3, context=None)
        saved = FmRolledHead.from_context(
            fast_config(gradient_checkpointing=True), 3, context=None
        )
        plain.fit(train_x, train_y, train_x, train_y)
        saved.fit(train_x, train_y, train_x, train_y)
        assert plain.loss_history == saved.loss_history
        assert torch.equal(plain.predict(query_x), saved.predict(query_x))


class TestTraining:
    def test_the_endpoint_loss_decreases(self) -> None:
        train_x, train_y, _, _ = separable_problem()
        prototypes = PrototypeHead(n_classes=3)
        prototypes.fit(train_x, train_y, train_x, train_y)
        _, history = train_rolled_out_field(
            train_x,
            prototypes.prototypes[train_y],
            sample_steps=4,
            hidden_dims=(32, 32),
            n_train_steps=300,
            batch_size=16,
            init_seed=0,
        )
        assert len(history) == 300
        assert sum(history[-20:]) / 20 < sum(history[:20]) / 20

    def test_transport_lands_training_points_on_their_own_prototype(self) -> None:
        """A fit check on the pairs the loss saw, not a claim about test points.

        Whether rolled-out training beats standard training on unseen features
        is the question the Stage 2 grid exists to answer, so it is measured
        there and not asserted here.
        """
        train_x, train_y, _, _ = separable_problem()
        # Selection off: the claim is about where the fit converges to, and on
        # a toy the prototype rule already solves, validation accuracy
        # saturates immediately and selection keeps an early field (ADR-028).
        head = FmRolledHead.from_context(
            fast_config(n_train_steps=800, hidden_dims=[128, 128], eval_every=0), 3, context=None
        )
        head.fit(train_x, train_y, train_x, train_y)
        targets = head.prototypes[train_y]
        before = float((train_x - targets).norm(dim=1).mean())
        after = float((head.transport(train_x) - targets).norm(dim=1).mean())
        assert after < 0.1 * before

    def test_zero_training_steps_leaves_an_untrained_field_and_an_empty_curve(self) -> None:
        train_x, train_y, _, _ = separable_problem()
        head = FmRolledHead.from_context(fast_config(n_train_steps=0), 3, context=None)
        head.fit(train_x, train_y, train_x, train_y)
        assert head.loss_history == []
        assert isinstance(head.field, VelocityMLP)

    def test_mismatched_endpoint_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="paired endpoints"):
            train_rolled_out_field(torch.zeros(4, 3), torch.zeros(4, 2), sample_steps=4)

    def test_a_step_count_below_one_is_refused(self) -> None:
        """Refused at both doors: there is no depth-zero rolled-out model."""
        with pytest.raises(ValueError, match="sample_steps must be >= 1"):
            FmRolledHead(n_classes=3, sample_steps=0)
        with pytest.raises(ValueError, match="sample_steps must be >= 1"):
            train_rolled_out_field(torch.zeros(4, 3), torch.zeros(4, 3), sample_steps=0)
