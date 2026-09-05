"""Strategy 2: classifier-guided targets and standard FM training (FR20, ADR-033).

Per batch, with the field in its current state and under no_grad:

    zhat  = psi_theta(z)                                  # T Euler steps
    g     = W^T (softmax(W zhat + b) - onehot(y))
    zhat <- zhat - eta * step(g)                          # m times
    zhat' = constrain(zhat, z)

then one standard FM update on the coupling z -> zhat'. The update is
`cfm_loss`, unchanged from Stage 2; what is new is that the coupling moves.

Two claims carry the design and both are asserted rather than argued.

The target displacement lies in `row(W)` exactly, because g is a combination of
rows of W. That is the fact `stage3_alignment.md` section 3.3 and the row-space
diagnostic both rest on, and it is what makes Strategy 2's search space
different from Strategy 1's rather than merely differently shaped.

The target compounds. `zhat` is built from the field that is being trained
toward the previous target, so each recompute adds another eta step and nothing
in the write-up bounds the total. The drift test exhibits it by feeding the
coupler a field that has learned the previous target exactly, which is the
training dynamic in its limiting case. The graded row runs the literal
unconstrained version anyway; `normalized` runs beside it in `ablations/` and
its cap is asserted here.
"""

import pytest
import torch
from test_head_contract import make_config, run_head_battery, separable_scenario
from test_scheme_parity import both_heads, field_hash
from test_scheme_parity import problem as parity_problem

from fm_fewshot.services.flow.training import rolled_out as rolled_module
from fm_fewshot.services.flow.training import standard as standard_module
from fm_fewshot.services.flow.training.base import train_field
from fm_fewshot.services.flow.training.classifier_guided import (
    ClassifierGuidedCoupler,
    GuidedTargetConfig,
    classification_gradient,
    guided_target,
    train_classifier_guided_field,
)
from fm_fewshot.services.flow.training.rolled_out_ce import row_space_projector
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import make_head

PARAMS: dict[str, object] = {
    "sample_steps": 4,
    "n_train_steps": 25,
    "hidden_dims": [16, 16],
    "batch_size": 8,
    "lr": 5e-3,
    "eval_every": 0,
    "probe_params": {"max_epochs": 20},
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


def identity_field(dim: int = 5) -> VelocityMLP:
    return VelocityMLP(dim=dim, hidden_dims=(16, 16), seed=0, zero_output_init=True)


class ConstantField(torch.nn.Module):
    """A field that transports every point by a fixed displacement.

    T Euler steps of a constant velocity land exactly on z + displacement, so
    this is "the field has learned its target perfectly" with no fitting and no
    tolerance. The drift test uses it to iterate the coupling the way training
    does.
    """

    def __init__(self, displacement: torch.Tensor) -> None:
        super().__init__()
        self.displacement = displacement

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.displacement


def null_component(displacement: torch.Tensor, weight: torch.Tensor) -> float:
    projector = row_space_projector(weight)
    return float((displacement - displacement @ projector).abs().max())


class TestTheTargetLivesInTheRowSpace:
    """The claim section 1 of the PRD rests on, asserted for random inputs."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("shape", [(3, 5), (4, 9), (7, 8)])
    def test_the_classification_gradient_is_a_combination_of_rows_of_w(
        self, seed: int, shape: tuple[int, int]
    ) -> None:
        n_classes, dim = shape
        g = torch.Generator().manual_seed(seed)
        weight = torch.randn(n_classes, dim, generator=g)
        bias = torch.randn(n_classes, generator=g)
        zhat = torch.randn(11, dim, generator=g)
        y = torch.randint(0, n_classes, (11,), generator=g)
        gradient = classification_gradient(zhat, y, weight, bias)
        assert gradient.shape == (11, dim)
        assert null_component(gradient, weight) < 1e-5

    def test_it_matches_autograd_on_the_same_cross_entropy(self) -> None:
        """The closed form is the gradient of the loss Strategy 1 optimizes."""
        weight, bias = classifier()
        zhat = torch.randn(6, 5, requires_grad=True)
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        torch.nn.functional.cross_entropy(zhat @ weight.T + bias, y, reduction="sum").backward()
        assert torch.allclose(classification_gradient(zhat.detach(), y, weight, bias), zhat.grad)

    def test_the_target_displacement_stays_in_the_row_space(self) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()
        field = identity_field()
        target = guided_target(
            field, train_x, train_y, weight, bias, sample_steps=4, config=GuidedTargetConfig()
        )
        # The field is identity here, so zhat = z and the whole displacement is
        # the guided step.
        assert null_component(target - train_x, weight) < 1e-5

    def test_the_step_reduces_the_classification_loss(self) -> None:
        """It is a descent step on the frozen probe's loss, not an arbitrary move."""
        train_x, train_y = problem()
        weight, bias = classifier()
        target = guided_target(
            identity_field(),
            train_x,
            train_y,
            weight,
            bias,
            sample_steps=4,
            config=GuidedTargetConfig(eta=0.05),
        )
        before = torch.nn.functional.cross_entropy(train_x @ weight.T + bias, train_y)
        after = torch.nn.functional.cross_entropy(target @ weight.T + bias, train_y)
        assert float(after) < float(before)


class TestTheTargetCompounds:
    """ADR-033: the fixed point of the loop is not bounded by the specification."""

    @staticmethod
    def _iterate(config: GuidedTargetConfig, rounds: int = 4) -> list[float]:
        """Distances from z after each recompute, with the field learning each target."""
        train_x, train_y = problem()
        weight, bias = classifier()
        field: torch.nn.Module = identity_field()
        distances = []
        for _ in range(rounds):
            target = guided_target(
                field, train_x, train_y, weight, bias, sample_steps=4, config=config
            )
            distances.append(float((target - train_x).norm(dim=1).mean()))
            field = ConstantField(target - train_x)
        return distances

    def test_the_raw_step_moves_further_from_z_on_every_recompute(self) -> None:
        distances = self._iterate(GuidedTargetConfig(target_step="raw", eta=5.0))
        assert distances == sorted(distances)
        assert distances[-1] > distances[0]

    def test_the_normalized_step_stays_inside_the_anchor_cap(self) -> None:
        distances = self._iterate(
            GuidedTargetConfig(target_step="normalized", eta=5.0, rho=0.5)
        )
        train_x, _ = problem()
        cap = float((0.5 * train_x.norm(dim=1)).max())
        assert max(distances) <= cap + 1e-5

    @pytest.mark.parametrize("eta", [0.5, 5.0, 50.0])
    def test_the_cap_holds_row_by_row_at_any_eta(self, eta: float) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()
        target = guided_target(
            identity_field(),
            train_x,
            train_y,
            weight,
            bias,
            sample_steps=4,
            config=GuidedTargetConfig(target_step="normalized", eta=eta, rho=0.5),
        )
        assert bool(
            ((target - train_x).norm(dim=1) <= 0.5 * train_x.norm(dim=1) + 1e-5).all()
        )

    def test_more_target_steps_move_the_raw_target_further(self) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()

        def distance(m: int) -> float:
            target = guided_target(
                identity_field(),
                train_x,
                train_y,
                weight,
                bias,
                sample_steps=4,
                config=GuidedTargetConfig(eta=5.0, target_steps=m),
            )
            return float((target - train_x).norm(dim=1).mean())

        assert distance(1) < distance(2) < distance(4)

    def test_an_unknown_target_step_is_refused(self) -> None:
        with pytest.raises(ValueError, match="target_step"):
            GuidedTargetConfig(target_step="clipped").validate()


class TestTheRecomputeSchedule:
    def _coupler(self, target_every: int) -> ClassifierGuidedCoupler:
        train_x, train_y = problem()
        weight, bias = classifier()
        return ClassifierGuidedCoupler(
            train_x,
            train_y,
            weight,
            bias,
            sample_steps=4,
            config=GuidedTargetConfig(target_every=target_every),
        )

    @pytest.mark.parametrize(
        ("target_every", "expected"),
        [(1, [1, 2, 3, 4, 5, 6, 7]), (2, [1, 3, 5, 7]), (3, [1, 4, 7]), (7, [1])],
    )
    def test_it_recomputes_on_exactly_the_expected_steps(
        self, target_every: int, expected: list[int]
    ) -> None:
        coupler = self._coupler(target_every)
        field = identity_field()
        rows = torch.arange(4)
        for _ in range(7):
            coupler(field, rows, coupler.x0[rows])
        assert coupler.recompute_steps == expected

    def test_a_stale_target_is_reused_between_recomputes(self) -> None:
        coupler = self._coupler(3)
        rows = torch.arange(4)
        first = coupler(identity_field(), rows, coupler.x0[rows])
        # A different field at the next step, which would move a fresh target.
        moved = ConstantField(torch.full((4, 5), 3.0))
        second = coupler(moved, rows, coupler.x0[rows])
        assert torch.equal(first, second)

    def test_the_cached_rows_are_the_rows_that_were_asked_for(self) -> None:
        coupler = self._coupler(3)
        field = identity_field()
        everything = coupler(field, torch.arange(12), coupler.x0)
        coupler_again = self._coupler(3)
        subset = torch.tensor([7, 2, 11])
        picked = coupler_again(field, subset, coupler.x0[subset])
        assert torch.allclose(picked, everything[subset], atol=1e-6)

    def test_recomputing_every_step_needs_no_cache(self) -> None:
        """The graded path computes the batch only, which is the same target."""
        coupler = self._coupler(1)
        field = identity_field()
        rows = torch.tensor([3, 5])
        batch = coupler(field, rows, coupler.x0[rows])
        full = self._coupler(1)(field, torch.arange(12), coupler.x0)
        assert torch.allclose(batch, full[rows], atol=1e-6)


class TestTraining:
    def test_the_flow_matching_loss_falls_on_a_seeded_toy(self) -> None:
        train_x, train_y = problem()
        weight, bias = classifier()
        _, history = train_classifier_guided_field(
            train_x,
            train_y,
            weight,
            bias,
            None,
            sample_steps=4,
            hidden_dims=(16, 16),
            n_train_steps=80,
            batch_size=8,
            lr=1e-2,
            init_seed=0,
        )
        assert min(history[-10:]) < history[0]

    def test_the_fit_is_deterministic_at_a_fixed_seed(self) -> None:
        def fit() -> list[float]:
            train_x, train_y = problem()
            weight, bias = classifier()
            _, history = train_classifier_guided_field(
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
            return history

        assert fit() == fit()

    def test_the_target_carries_no_gradient_into_the_update(self) -> None:
        """The coupling is data for the FM loss, not something to optimize through."""
        train_x, train_y = problem()
        weight, bias = classifier()
        coupler = ClassifierGuidedCoupler(
            train_x, train_y, weight, bias, sample_steps=4, config=GuidedTargetConfig()
        )
        field = identity_field()
        rows = torch.arange(4)
        target = coupler(field, rows, train_x[rows])
        assert target.requires_grad is False
        assert target.grad_fn is None


class TestTheHead:
    def test_the_head_passes_the_contract_battery(self) -> None:
        cfg = make_config(
            "fm_prelinear_guided", **dict(PARAMS, n_train_steps=5, probe_params={})
        )
        run_head_battery(cfg, 3, *separable_scenario())

    def test_the_head_reads_its_stage_3_parameters(self) -> None:
        cfg = make_config(
            "fm_prelinear_guided",
            **dict(
                PARAMS,
                target_step="normalized",
                eta=0.25,
                target_steps=3,
                target_every=4,
                rho=0.75,
            ),
        )
        head = make_head(cfg, 3)
        assert head.target_config == GuidedTargetConfig(
            target_step="normalized", eta=0.25, target_steps=3, target_every=4, rho=0.75
        )

    def test_the_graded_defaults_are_the_write_ups_literal_step(self) -> None:
        head = make_head(make_config("fm_prelinear_guided", **PARAMS), 3)
        assert head.target_config == GuidedTargetConfig()
        assert head.target_config.target_step == "raw"
        assert head.target_config.target_every == 1

    def test_the_head_trains_the_field_and_leaves_the_probe_alone(self) -> None:
        train_x, train_y = problem()
        head = make_head(make_config("fm_prelinear_guided", **PARAMS), 3)
        head.fit(train_x, train_y, train_x, train_y)
        assert len(head.loss_history) == PARAMS["n_train_steps"]
        assert head.probe.weight.grad is None
        assert not torch.equal(head.transport(train_x), train_x)

    def test_an_unknown_target_step_is_refused_at_construction(self) -> None:
        cfg = make_config("fm_prelinear_guided", **dict(PARAMS, target_step="clipped"))
        with pytest.raises(ValueError, match="target_step"):
            make_head(cfg, 3)

    def test_the_transported_features_move_toward_the_classifier(self) -> None:
        """The point of the strategy, on a toy where it should work."""
        train_x, train_y = problem()
        head = make_head(make_config("fm_prelinear_guided", **dict(PARAMS, lr=1e-2)), 3)
        head.fit(train_x, train_y, train_x, train_y)
        probe_loss = torch.nn.functional.cross_entropy(
            head.probe.predict(train_x), train_y
        )
        head_loss = torch.nn.functional.cross_entropy(head.predict(train_x), train_y)
        assert float(head_loss) < float(probe_loss)


class TestTheSeamIsInvisibleToStageTwo:
    """`couple=None` must leave the Stage 2 fits bit-identical (ADR-033).

    The digests were taken before `train_field` had the hook. A self-comparison
    would move with the code and pass; these do not.
    """

    STANDARD_DIGEST = "5b466e766b53c3ab394d512fe00b5a3bea27acb6dc6af20bdad2bed4d8d0fbf5"
    ROLLED_DIGEST = "7691f9234d02a8bc47bfa1f9edd8e0b36bf5762112c56032ee55a4a2a5f6f6dc"

    def test_both_stage_2_schemes_fit_the_same_field_as_before_the_seam(self) -> None:
        train_x, train_y = parity_problem()
        standard, rolled = both_heads(n_train_steps=25, eval_every=0)
        for head in (standard, rolled):
            head.fit(train_x, train_y, train_x, train_y)
        assert field_hash(standard.field) == self.STANDARD_DIGEST
        assert field_hash(rolled.field) == self.ROLLED_DIGEST

    def test_both_stage_2_schemes_pass_no_coupling(self, monkeypatch) -> None:
        seen: list[object] = []

        def recording(*args, **kwargs):
            seen.append(kwargs.get("couple"))
            return train_field(*args, **kwargs)

        monkeypatch.setattr(standard_module, "train_field", recording)
        monkeypatch.setattr(rolled_module, "train_field", recording)
        train_x, train_y = parity_problem()
        standard, rolled = both_heads(n_train_steps=2, eval_every=0)
        for head in (standard, rolled):
            head.fit(train_x, train_y, train_x, train_y)
        assert seen == [None, None]


class TestTheCoupleHook:
    def test_the_hook_replaces_the_payload_the_loss_receives(self) -> None:
        seen: list[torch.Tensor] = []

        def batch_loss(field, batch_x0, batch_x1, generator):
            seen.append(batch_x1)
            return ((field(batch_x0, torch.zeros(batch_x0.shape[0])) - batch_x1) ** 2).sum(
                dim=1
            ).mean()

        train_field(
            torch.randn(6, 4),
            torch.arange(6),
            batch_loss,
            couple=lambda field, rows, batch_x0: torch.full((rows.shape[0], 4), 2.0),
            hidden_dims=(4,),
            n_train_steps=2,
            batch_size=3,
            init_seed=0,
        )
        assert all(torch.equal(payload, torch.full((3, 4), 2.0)) for payload in seen)

    def test_the_hook_runs_without_gradients(self) -> None:
        recorded: list[bool] = []

        def couple(field, rows, batch_x0):
            recorded.append(torch.is_grad_enabled())
            return batch_x0.clone()

        train_field(
            torch.randn(6, 4),
            torch.arange(6),
            lambda field, x0, x1, g: ((field(x0, torch.zeros(x0.shape[0])) - x1) ** 2).sum(),
            couple=couple,
            hidden_dims=(4,),
            n_train_steps=2,
            batch_size=3,
            init_seed=0,
        )
        assert recorded == [False, False]

    def test_the_hook_sees_the_rows_the_loop_drew(self) -> None:
        x0 = torch.arange(8, dtype=torch.float32).unsqueeze(1)
        seen: list[tuple[list[int], list[int]]] = []

        def couple(field, rows, batch_x0):
            seen.append(([int(r) for r in rows], [int(v) for v in batch_x0.flatten()]))
            return batch_x0.clone()

        train_field(
            x0,
            torch.arange(8),
            lambda field, a, b, g: ((field(a, torch.zeros(a.shape[0])) - b) ** 2).sum(),
            couple=couple,
            hidden_dims=(4,),
            n_train_steps=3,
            batch_size=2,
            init_seed=0,
        )
        assert all(rows == values for rows, values in seen)
