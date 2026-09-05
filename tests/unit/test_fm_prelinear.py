"""The Stage 3 head: an FM block ahead of a frozen Stage 1 linear probe.

ADR-031. The head fits a `LinearProbeHead` on the same tensors and the same
`init_seed` the Stage 1 run used, freezes it, and trains only the velocity
field in front of it. Three claims carry the design and each is asserted here
rather than argued:

- with `n_train_steps = 0` the head's logits are **bit-identical** to that
  probe's. The field is initialized to exact identity (ADR-030), so this one
  equality covers both "initialize close to identity" and "train the classifier
  exactly as in Stage 1" at once, and it is the assertion that would catch a
  head that quietly refitted, rescaled or reordered anything;
- `W` and `b` are unchanged by `fit`, compared element-wise against a probe
  fitted independently from the same config;
- only `field.parameters()` receive gradients. The probe's tensors carry no
  gradient at any point, which is what "kept frozen" means operationally.

The base class is exercised through a local subclass whose `_fit_field` drives
the shared loop with a trivial objective. The two real strategies register their
own keys and run the contract battery in their own files; what is tested here is
the scaffolding all of them sit on.
"""

import pytest
import torch
from test_head_contract import make_config

from fm_fewshot.services.flow.training.base import ValidationSelector, train_field
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import NotFittedError
from fm_fewshot.services.heads.fm_prelinear import FmPreLinearHead
from fm_fewshot.services.heads.linear_probe import LinearProbeHead

PARAMS: dict[str, object] = {
    "sample_steps": 4,
    "n_train_steps": 3,
    "hidden_dims": [16, 16],
    "batch_size": 8,
    "lr": 1e-3,
    "eval_every": 0,
}

# The probe is Stage 1's and its 200 epochs are not what these tests measure.
FAST_PROBE: dict[str, object] = {"max_epochs": 20}


class NullFitHead(FmPreLinearHead):
    """Drives the shared loop with an objective that needs no target.

    Enough to exercise the base end to end without importing either real
    strategy, so a failure here is a failure of the scaffolding.

    The objective pushes the velocity toward a constant rather than toward
    zero, because zero is where the field starts (ADR-030) and an objective
    minimized there would have no gradient, leave the field at identity, and
    make the "something was trained" assertions vacuously false. Both real
    strategies have a nonzero gradient at identity for the same reason the
    PRD gives: the output layer's gradient is delta (x) h with h nonzero.
    """

    def _fit_field(self, train_x, train_y, probe, selector):
        def batch_loss(field, batch_x0, batch_payload, generator):
            times = torch.zeros(batch_x0.shape[0])
            return ((field(batch_x0, times) - 1.0) ** 2).sum(dim=1).mean()

        return train_field(
            train_x,
            train_y,
            batch_loss,
            selector,
            hidden_dims=self._hidden_dims,
            time_conditioning=self._time_conditioning,
            n_train_steps=self._n_train_steps,
            batch_size=self._batch_size,
            lr=self._lr,
            init_seed=self._init_seed,
            zero_output_init=self._zero_output_init,
        )


def problem(n_classes: int = 3, dim: int = 5, per_class: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 4.0
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    train_x = centers[labels] + 0.3 * torch.randn(labels.shape[0], dim, generator=g)
    val_labels = torch.arange(n_classes).repeat_interleave(2)
    val_x = centers[val_labels] + 0.3 * torch.randn(val_labels.shape[0], dim, generator=g)
    return train_x, labels, val_x, val_labels


def head(**overrides: object) -> NullFitHead:
    params = dict(PARAMS)
    params.update(overrides)
    cfg = make_config("fm_prelinear_null", **params)
    return NullFitHead(**NullFitHead.shared_params(cfg, 3))


def fitted_probe(init_seed: int = 0, max_epochs: int = 200) -> LinearProbeHead:
    """The Stage 1 probe at this setting, fitted independently of any FM head."""
    probe = LinearProbeHead(n_classes=3, init_seed=init_seed, max_epochs=max_epochs)
    probe.fit(*problem())
    return probe


class TestTheUntrainedSystemIsTheProbe:
    """The single assertion ADR-030 and ADR-031 both reduce to."""

    def test_zero_training_steps_reproduce_the_probe_bit_for_bit(self) -> None:
        stage3 = head(n_train_steps=0, probe_params=FAST_PROBE)
        stage3.fit(*problem())
        query = problem()[0]
        assert torch.equal(stage3.predict(query), fitted_probe(max_epochs=20).predict(query))

    def test_the_untrained_block_moves_nothing(self) -> None:
        stage3 = head(n_train_steps=0, probe_params=FAST_PROBE)
        stage3.fit(*problem())
        query = problem()[0]
        assert torch.equal(stage3.transport(query), query)

    def test_the_comparison_can_fail(self) -> None:
        """Negative control: a trained block does move the features.

        Otherwise the equality above would hold for a head that never
        transports anything and the test would be vacuous.
        """
        stage3 = head(n_train_steps=5, probe_params=FAST_PROBE, lr=5e-1)
        stage3.fit(*problem())
        query = problem()[0]
        assert not torch.equal(stage3.transport(query), query)


class TestTheClassifierIsFrozen:
    def test_w_and_b_are_unchanged_by_fitting_the_field(self) -> None:
        stage3 = head(probe_params=FAST_PROBE)
        stage3.fit(*problem())
        reference = fitted_probe(max_epochs=20)
        assert torch.equal(stage3.probe.weight, reference.weight)
        assert torch.equal(stage3.probe.bias, reference.bias)

    def test_the_probe_tensors_carry_no_gradient(self) -> None:
        stage3 = head(probe_params=FAST_PROBE)
        stage3.fit(*problem())
        for tensor in (stage3.probe.weight, stage3.probe.bias):
            assert tensor.requires_grad is False
            assert tensor.grad is None

    def test_the_field_is_what_receives_the_gradients(self) -> None:
        """The other half of the claim: something was trained."""
        stage3 = head(probe_params=FAST_PROBE)
        stage3.fit(*problem())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage3.field.parameters())

    def test_the_probe_is_the_stage_1_probe_at_the_same_seed(self) -> None:
        """ADR-031: bit-identical by construction, not by care."""
        stage3 = head(probe_params=FAST_PROBE)
        stage3.fit(*problem())
        reference = fitted_probe(max_epochs=20)
        query = problem()[0]
        assert torch.equal(stage3.probe.predict(query), reference.predict(query))


class TestContractSurface:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            head().predict(problem()[0])

    def test_field_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = head().field

    def test_probe_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = head().probe.weight

    def test_transport_and_trajectory_shapes(self) -> None:
        stage3 = head(probe_params=FAST_PROBE)
        stage3.fit(*problem())
        query = problem()[0]
        assert stage3.transport(query).shape == query.shape
        assert stage3.trajectory(query).shape == (PARAMS["sample_steps"] + 1, *query.shape)

    def test_logits_have_one_column_per_class(self) -> None:
        stage3 = head(probe_params=FAST_PROBE)
        stage3.fit(*problem())
        logits = stage3.predict(problem()[0])
        assert logits.shape == (12, 3)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()

    def test_the_field_is_built_at_identity_by_default(self) -> None:
        """The Stage 3 default is the write-up's requirement, not an opt-in."""
        assert head().zero_output_init is True

    def test_the_fit_is_deterministic_at_a_fixed_seed(self) -> None:
        query = problem()[0]
        first, second = head(probe_params=FAST_PROBE), head(probe_params=FAST_PROBE)
        first.fit(*problem())
        second.fit(*problem())
        assert torch.equal(first.predict(query), second.predict(query))


class TestSelectionIsTheFieldsNotTheProbes:
    """`best_epoch` reports the FM step that was kept, as it does in Stage 2.

    The probe inside the head runs its own Stage 1 epoch selection, so there
    are two selected checkpoints in one run and the summary must carry the one
    Stage 3 varies.
    """

    def test_best_epoch_is_the_selected_training_step(self) -> None:
        stage3 = head(n_train_steps=6, eval_every=2, probe_params=FAST_PROBE)
        stage3.fit(*problem())
        assert stage3.best_epoch in {2, 4, 6}

    def test_best_epoch_is_none_when_selection_is_off(self) -> None:
        stage3 = head(eval_every=0, probe_params=FAST_PROBE)
        stage3.fit(*problem())
        assert stage3.best_epoch is None

    def test_the_validation_history_is_recorded_on_the_grid(self) -> None:
        stage3 = head(n_train_steps=6, eval_every=2, probe_params=FAST_PROBE)
        stage3.fit(*problem())
        assert [step for step, _ in stage3.val_top1_history] == [2, 4, 6]

    def test_the_loss_history_is_the_fields(self) -> None:
        stage3 = head(n_train_steps=3, probe_params=FAST_PROBE)
        stage3.fit(*problem())
        assert len(stage3.loss_history) == 3

    def test_the_selector_scores_with_the_frozen_probe(self) -> None:
        """ADR-020: the number selection maximizes is the number the run reports."""
        captured: list[ValidationSelector] = []
        original = FmPreLinearHead._fit_field

        class Capturing(NullFitHead):
            def _fit_field(self, train_x, train_y, probe, selector):
                captured.append(selector)
                return super()._fit_field(train_x, train_y, probe, selector)

        assert original is not Capturing._fit_field
        cfg = make_config("fm_prelinear_null", **dict(PARAMS, probe_params=FAST_PROBE))
        stage3 = Capturing(**Capturing.shared_params(cfg, 3))
        stage3.fit(*problem())
        assert captured[0].classifier is stage3.probe
        assert captured[0].sample_steps == PARAMS["sample_steps"]


class TestConstruction:
    def test_shared_params_read_the_stage_3_block(self) -> None:
        cfg = make_config(
            "fm_prelinear_null",
            sample_steps=12,
            n_train_steps=7,
            batch_size=4,
            lr=2e-3,
            hidden_dims=[8],
            eval_every=3,
            zero_output_init=False,
        )
        params = FmPreLinearHead.shared_params(cfg, 3)
        assert params["sample_steps"] == 12
        assert params["n_train_steps"] == 7
        assert params["batch_size"] == 4
        assert params["lr"] == pytest.approx(2e-3)
        assert params["hidden_dims"] == (8,)
        assert params["eval_every"] == 3
        assert params["zero_output_init"] is False

    def test_probe_params_are_a_separate_namespace(self) -> None:
        """The FM lr and the probe lr are different numbers with the same name.

        `head_params["lr"]` is the field's, since the head is an FM head. A
        Stage 1 hyperparameter that moved reaches the probe through
        `probe_params` and nowhere else, so raising the field's learning rate
        cannot silently retrain the classifier on different terms.
        """
        cfg = make_config(
            "fm_prelinear_null", lr=0.5, probe_params={"lr": 1e-4, "max_epochs": 3}
        )
        stage3 = NullFitHead(**NullFitHead.shared_params(cfg, 3))
        assert stage3._lr == pytest.approx(0.5)
        assert stage3.probe._lr == pytest.approx(1e-4)
        assert stage3.probe._max_epochs == 3

    def test_the_probe_defaults_are_the_stage_1_defaults(self) -> None:
        cfg = make_config("fm_prelinear_null")
        stage3 = NullFitHead(**NullFitHead.shared_params(cfg, 3))
        reference = LinearProbeHead(n_classes=3)
        assert stage3.probe._lr == reference._lr
        assert stage3.probe._weight_decay == reference._weight_decay
        assert stage3.probe._batch_size == reference._batch_size
        assert stage3.probe._max_epochs == reference._max_epochs

    def test_the_probe_takes_the_runs_init_seed(self) -> None:
        cfg = make_config("fm_prelinear_null")
        stage3 = NullFitHead(**NullFitHead.shared_params(cfg, 3))
        assert stage3.probe._init_seed == cfg.init_seed

    @pytest.mark.parametrize("bad", [0, -1])
    def test_sample_steps_below_one_is_refused(self, bad: int) -> None:
        with pytest.raises(ValueError, match="sample_steps"):
            NullFitHead(3, sample_steps=bad)

    def test_negative_eval_every_is_refused(self) -> None:
        with pytest.raises(ValueError, match="eval_every"):
            NullFitHead(3, eval_every=-1)


class TestTheFieldIsTrainedAtIdentity:
    def test_the_field_handed_to_the_trainer_is_zero_at_the_start(self) -> None:
        """Whatever the strategy, step 0 sees a field that transports nothing."""
        stage3 = head(n_train_steps=0)
        stage3.fit(*problem())
        z, t = torch.randn(4, 5), torch.rand(4)
        assert torch.equal(stage3.field(z, t), torch.zeros(4, 5))

    def test_the_flag_reaches_the_velocity_network(self) -> None:
        stage3 = head(n_train_steps=0, zero_output_init=False)
        stage3.fit(*problem())
        assert isinstance(stage3.field, VelocityMLP)
        assert stage3.field.zero_output_init is False


class TestTheLoopTakesAPerRowPayload:
    """The one change Stage 3 needed in the shared loop, and its guard.

    `x1` is no longer required to be a second endpoint, because Stage 3 pairs
    each feature with its label. What is still required is one payload row per
    source point, and the schemes that really do pair endpoints keep the
    stricter check at their own door.
    """

    def test_labels_are_accepted_as_the_payload(self) -> None:
        seen: list[tuple[int, ...]] = []

        def batch_loss(field, batch_x0, batch_payload, generator):
            seen.append(tuple(batch_payload.shape))
            times = torch.zeros(batch_x0.shape[0])
            return ((field(batch_x0, times) - 1.0) ** 2).sum(dim=1).mean()

        train_field(
            torch.randn(8, 4),
            torch.arange(8),
            batch_loss,
            hidden_dims=(4,),
            n_train_steps=2,
            batch_size=3,
            init_seed=0,
        )
        assert seen == [(3,), (3,)]

    def test_a_payload_with_the_wrong_number_of_rows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one row per source point"):
            train_field(torch.randn(8, 4), torch.arange(7), lambda *a: None, n_train_steps=1)

    def test_the_stage_2_schemes_still_refuse_a_mismatched_endpoint(self) -> None:
        """The guard did not disappear, it moved to where the pairing is real."""
        from fm_fewshot.services.flow.training.standard import train_standard_field

        with pytest.raises(ValueError, match="paired endpoints"):
            train_standard_field(torch.zeros(4, 3), torch.zeros(4, 2))
