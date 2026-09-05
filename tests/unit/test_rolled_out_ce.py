"""Strategy 1: end-to-end rolled-out classification training (FR19, ADR-034).

    zhat_0 = z,  zhat_{k+1} = zhat_k + (1/T) v_theta(zhat_k, k/T)
    L_cls  = CE(W zhat_T + b, y)

backpropagated through all T velocity predictions, with W and b frozen. Same
rollout as Stage 2's `fm_rolled`, different loss: endpoint MSE names a point in
feature space and cross-entropy names a decision, and Stage 3 has no prototype
to be the point.

The two penalties the write-up names are optional and carry zero weight in the
graded grid, so the tests hold them to two things: that they are off by
default, exactly, and that when on they behave like penalties.

`project_velocity` is not his; it is the control from `stage3_alignment.md`
section 3.3 that puts Strategy 1 in Strategy 2's search space. Its test is the
one that would catch a projector that silently does nothing.
"""

import pytest
import torch
from test_head_contract import make_config, run_head_battery, separable_scenario

from fm_fewshot.services.flow.training.rolled_out_ce import (
    RolledOutCeConfig,
    rolled_out_ce_loss,
    row_space_projector,
    train_rolled_out_ce_field,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import make_head

FAST_PROBE: dict[str, object] = {"max_epochs": 20}
PARAMS: dict[str, object] = {
    "sample_steps": 4,
    "n_train_steps": 25,
    "hidden_dims": [16, 16],
    "batch_size": 8,
    "lr": 5e-3,
    "eval_every": 0,
    "probe_params": FAST_PROBE,
}


def problem(n_classes: int = 3, dim: int = 5, per_class: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 4.0
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    train_x = centers[labels] + 0.3 * torch.randn(labels.shape[0], dim, generator=g)
    return train_x, labels


def classifier(n_classes: int = 3, dim: int = 5, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_classes, dim, generator=g), torch.randn(n_classes, generator=g)


def field(dim: int = 5, seed: int = 0, zero: bool = False) -> VelocityMLP:
    return VelocityMLP(dim=dim, hidden_dims=(16, 16), seed=seed, zero_output_init=zero)


class Scaled(torch.nn.Module):
    """The same field with its velocity multiplied, for the monotonicity checks."""

    def __init__(self, inner: VelocityMLP, factor: float) -> None:
        super().__init__()
        self.inner = inner
        self.factor = factor

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.factor * self.inner(x, t)


class TestTheObjective:
    def test_the_classification_loss_falls_on_a_seeded_toy(self) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()
        _, history = train_rolled_out_ce_field(
            train_x,
            train_y,
            weight,
            bias,
            None,
            sample_steps=4,
            hidden_dims=(16, 16),
            n_train_steps=60,
            batch_size=8,
            lr=1e-2,
            init_seed=0,
        )
        assert history[-1] < history[0]

    def test_the_loss_is_a_cross_entropy_on_the_transported_features(self) -> None:
        """Not an endpoint distance: the value is CE of the frozen map's logits."""
        train_x, train_y = problem()
        weight, bias = classifier()
        zero = field(zero=True)
        loss = rolled_out_ce_loss(zero, train_x, train_y, weight, bias, sample_steps=4)
        # A zero field transports nothing, so this is the probe's own CE.
        expected = torch.nn.functional.cross_entropy(train_x @ weight.T + bias, train_y)
        assert torch.equal(loss, expected)

    def test_the_classifier_is_not_updated_by_the_objective(self) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()
        weight.requires_grad_(True)
        bias.requires_grad_(True)
        loss = rolled_out_ce_loss(field(), train_x, train_y, weight, bias, sample_steps=4)
        loss.backward()
        assert weight.grad is None
        assert bias.grad is None

    def test_the_gradient_reaches_every_velocity_prediction(self) -> None:
        """Rolled-out means through all T, not only the last step."""
        train_x, train_y = problem()
        weight, bias = classifier()
        net = field(zero=True)
        rolled_out_ce_loss(net, train_x, train_y, weight, bias, sample_steps=4).backward()
        assert all(p.grad is not None for p in net.parameters())
        assert any(p.grad.abs().sum() > 0 for p in net.parameters())


class TestThePenaltiesAreOffByDefault:
    def test_zero_weights_reproduce_the_unpenalized_loss_exactly(self) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()
        net = field()
        plain = rolled_out_ce_loss(net, train_x, train_y, weight, bias, sample_steps=4)
        weighted = rolled_out_ce_loss(
            net,
            train_x,
            train_y,
            weight,
            bias,
            sample_steps=4,
            lambda_disp=0.0,
            lambda_vel=0.0,
        )
        assert torch.equal(plain, weighted)

    def test_the_defaults_are_zero(self) -> None:
        assert RolledOutCeConfig().lambda_disp == 0.0
        assert RolledOutCeConfig().lambda_vel == 0.0
        assert RolledOutCeConfig().project_velocity is False


class TestThePenaltiesBehaveLikePenalties:
    @staticmethod
    def _penalties(net, factor: float | None = None) -> tuple[float, float]:
        train_x, train_y = problem()
        weight, bias = classifier()
        used = net if factor is None else Scaled(net, factor)
        base = rolled_out_ce_loss(used, train_x, train_y, weight, bias, sample_steps=4)
        with_disp = rolled_out_ce_loss(
            used, train_x, train_y, weight, bias, sample_steps=4, lambda_disp=1.0
        )
        with_vel = rolled_out_ce_loss(
            used, train_x, train_y, weight, bias, sample_steps=4, lambda_vel=1.0
        )
        return float((with_disp - base).detach()), float((with_vel - base).detach())

    def test_both_penalties_are_non_negative(self) -> None:
        displacement, velocity = self._penalties(field())
        assert displacement >= 0.0
        assert velocity >= 0.0

    def test_both_penalties_are_zero_at_the_identity_field(self) -> None:
        displacement, velocity = self._penalties(field(zero=True))
        assert displacement == 0.0
        assert velocity == 0.0

    @pytest.mark.parametrize("index", [0, 1])
    def test_both_penalties_increase_when_the_field_is_scaled_up(self, index: int) -> None:
        net = field()
        small = self._penalties(net, factor=0.5)[index]
        large = self._penalties(net, factor=2.0)[index]
        assert large > small

    def test_a_penalty_moves_the_total_loss(self) -> None:
        """Otherwise "carries zero weight in the graded grid" would say nothing."""
        train_x, train_y = problem()
        weight, bias = classifier()
        net = field()
        plain = rolled_out_ce_loss(net, train_x, train_y, weight, bias, sample_steps=4)
        penalized = rolled_out_ce_loss(
            net, train_x, train_y, weight, bias, sample_steps=4, lambda_disp=1.0
        )
        assert float(penalized.detach()) > float(plain.detach())


class TestRowSpaceProjection:
    """The control for `stage3_alignment.md` 3.3: same objective, Strategy 2's space."""

    @staticmethod
    def _null_component(displacement: torch.Tensor, weight: torch.Tensor) -> float:
        projector = row_space_projector(weight)
        null_part = displacement - displacement @ projector
        return float(null_part.abs().max())

    def test_the_projector_is_an_orthogonal_projection_onto_the_row_space(self) -> None:
        weight, _ = classifier(n_classes=3, dim=7)
        projector = row_space_projector(weight)
        assert projector.shape == (7, 7)
        assert torch.allclose(projector @ projector, projector, atol=1e-5)
        assert torch.allclose(projector, projector.T, atol=1e-6)
        # Every row of W is fixed by it, and the rank is the rank of W.
        assert torch.allclose(weight @ projector, weight, atol=1e-5)
        assert int(torch.linalg.matrix_rank(projector)) == 3

    def test_a_projected_run_lands_with_no_null_space_component(self) -> None:
        train_x, train_y = problem(dim=7)
        weight, bias = classifier(dim=7)
        net, _ = train_rolled_out_ce_field(
            train_x,
            train_y,
            weight,
            bias,
            None,
            sample_steps=4,
            hidden_dims=(16, 16),
            n_train_steps=20,
            batch_size=8,
            lr=1e-2,
            init_seed=0,
            project_velocity=True,
        )
        with torch.no_grad():
            from fm_fewshot.services.flow.training.base import transport

            landed = transport(net, train_x, sample_steps=4)
        assert self._null_component(landed - train_x, weight) < 1e-5

    def test_an_unprojected_run_does_not(self) -> None:
        """The negative control: without the projection the block spends capacity
        in the 4 of 7 directions the classifier cannot see."""
        train_x, train_y = problem(dim=7)
        weight, bias = classifier(dim=7)
        net, _ = train_rolled_out_ce_field(
            train_x,
            train_y,
            weight,
            bias,
            None,
            sample_steps=4,
            hidden_dims=(16, 16),
            n_train_steps=20,
            batch_size=8,
            lr=1e-2,
            init_seed=0,
        )
        with torch.no_grad():
            from fm_fewshot.services.flow.training.base import transport

            landed = transport(net, train_x, sample_steps=4)
        assert self._null_component(landed - train_x, weight) > 1e-4

    def test_projection_does_not_change_the_logits_it_is_scored_on(self) -> None:
        """A projected and an unprojected displacement of the same row-space part
        give the same logits, which is why the null part is capacity spent unseen."""
        weight, bias = classifier(dim=7)
        projector = row_space_projector(weight)
        g = torch.Generator().manual_seed(3)
        z = torch.randn(6, 7, generator=g)
        displacement = torch.randn(6, 7, generator=g)
        projected = displacement @ projector
        assert torch.allclose(
            (z + displacement) @ weight.T + bias,
            (z + projected) @ weight.T + bias,
            atol=1e-5,
        )


class TestTheHead:
    def test_the_head_passes_the_contract_battery(self) -> None:
        # The probe runs its Stage 1 epoch budget here: the battery asserts the
        # head classifies the scenario, and a probe stopped at 20 epochs has not
        # separated it yet, which would be a test of the budget and not of the head.
        cfg = make_config(
            "fm_prelinear_ce", **dict(PARAMS, n_train_steps=5, probe_params={})
        )
        run_head_battery(cfg, 3, *separable_scenario())

    def test_the_head_reads_its_three_stage_3_parameters(self) -> None:
        cfg = make_config(
            "fm_prelinear_ce",
            **dict(PARAMS, lambda_disp=0.25, lambda_vel=0.5, project_velocity=True),
        )
        head = make_head(cfg, 3)
        assert head.ce_config == RolledOutCeConfig(
            lambda_disp=0.25, lambda_vel=0.5, project_velocity=True
        )

    def test_the_graded_defaults_are_the_write_ups_plain_cross_entropy(self) -> None:
        head = make_head(make_config("fm_prelinear_ce", **PARAMS), 3)
        assert head.ce_config == RolledOutCeConfig()

    def test_the_head_trains_the_field_and_leaves_the_probe_alone(self) -> None:
        train_x, train_y = problem()
        head = make_head(make_config("fm_prelinear_ce", **PARAMS), 3)
        head.fit(train_x, train_y, train_x, train_y)
        assert len(head.loss_history) == PARAMS["n_train_steps"]
        assert head.probe.weight.grad is None
        assert not torch.equal(head.transport(train_x), train_x)

    def test_the_fit_is_deterministic_at_a_fixed_seed(self) -> None:
        train_x, train_y = problem()
        first = make_head(make_config("fm_prelinear_ce", **PARAMS), 3)
        second = make_head(make_config("fm_prelinear_ce", **PARAMS), 3)
        first.fit(train_x, train_y, train_x, train_y)
        second.fit(train_x, train_y, train_x, train_y)
        assert torch.equal(first.predict(train_x), second.predict(train_x))
