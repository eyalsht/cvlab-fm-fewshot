"""Checkpoint selection on validation accuracy for both FM heads (ADR-028).

The 108-run Stage 2 grid is the evidence this exists for: rolled-out training
drove its objective to ~0.15 on 235 training points while test top-1 fell to
0.18 against a 0.48 prototype baseline. The fit was not unstable, it was
memorized, and a fixed 2000-step schedule that keeps the final field has no way
to notice. The linear probe has selected on validation accuracy since Stage 1
(ADR-014); these tests give the FM heads the same rule.

What is asserted here is invariant, never accuracy: which step the head keeps,
that the kept field is the field as it stood at that step, that selection does
not perturb the fit it observes, and that both schemes run one implementation
of all of it. Whether selection raises test accuracy is a question for the
grid, not for a unit test.

`eval_every = 0` turns the whole mechanism off and restores the pre-selection
behaviour exactly, which is what makes the decision reversible.
"""

import pytest
import torch
from test_head_contract import make_config
from test_scheme_parity import field_hash

from fm_fewshot.services.flow.training.base import ValidationSelector
from fm_fewshot.services.flow.training.rolled_out import train_rolled_out_field
from fm_fewshot.services.flow.training.standard import train_standard_field
from fm_fewshot.services.heads.base import make_head
from fm_fewshot.services.heads.fm_base import DEFAULT_EVAL_EVERY, FmHead
from fm_fewshot.services.heads.prototype import PrototypeHead

HEADS = ["fm_standard", "fm_rolled"]

BASE_PARAMS: dict[str, object] = {
    "sample_steps": 4,
    "n_train_steps": 60,
    "hidden_dims": [32, 32],
    "batch_size": 16,
}


def separable_problem(n_classes: int = 3, dim: int = 8, per_class: int = 6, seed: int = 0):
    """Well-separated classes, so the prototype rule is already right on the val split.

    That matters for these tests: validation accuracy saturates at its ceiling
    on the first evaluation, so the tie rule decides, and the selected step is
    the first one rather than the last. A selection that silently kept the
    final field would be visible.
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 6.0
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    train_x = centers[labels] + 0.35 * torch.randn(labels.shape[0], dim, generator=g)
    val_labels = torch.arange(n_classes).repeat_interleave(4)
    val_x = centers[val_labels] + 0.35 * torch.randn(val_labels.shape[0], dim, generator=g)
    return train_x, labels, val_x, val_labels


def fitted(head_type: str, **overrides: object):
    params = dict(BASE_PARAMS)
    params.update(overrides)
    train_x, train_y, val_x, val_y = separable_problem()
    head = make_head(make_config(head_type, **params), 3)
    head.fit(train_x, train_y, val_x, val_y)
    return head


def trained_field(head_type: str, n_train_steps: int, sample_steps: int = 4):
    """The field the scheme's own trainer produces, with no head in the way."""
    train_x, train_y, val_x, val_y = separable_problem()
    prototypes = PrototypeHead(n_classes=3)
    prototypes.fit(train_x, train_y, val_x, val_y)
    shared = {
        "hidden_dims": (32, 32),
        "n_train_steps": n_train_steps,
        "batch_size": 16,
        "init_seed": 0,
    }
    targets = prototypes.prototypes[train_y]
    if head_type == "fm_standard":
        field, _ = train_standard_field(train_x, targets, **shared)
    else:
        field, _ = train_rolled_out_field(
            train_x, targets, sample_steps=sample_steps, **shared
        )
    return field


class TestSharedParameter:
    """One parameter, read in one place, so neither scheme can be selected harder."""

    def test_both_schemes_read_eval_every_from_shared_params(self) -> None:
        params = FmHead.shared_params(make_config("fm_standard", eval_every=7), 3)
        assert params["eval_every"] == 7

    def test_the_two_schemes_agree_on_every_shared_parameter(self) -> None:
        standard = FmHead.shared_params(make_config("fm_standard", **BASE_PARAMS), 3)
        rolled = FmHead.shared_params(make_config("fm_rolled", **BASE_PARAMS), 3)
        assert standard == rolled
        assert standard["eval_every"] == DEFAULT_EVAL_EVERY

    def test_selection_is_on_by_default(self) -> None:
        """The grid's evidence says the final field is the wrong default."""
        assert DEFAULT_EVAL_EVERY > 0

    @pytest.mark.parametrize("head_type", HEADS)
    def test_a_negative_eval_every_is_refused(self, head_type: str) -> None:
        """At construction, before the fit spends anything."""
        with pytest.raises(ValueError, match="eval_every"):
            make_head(make_config(head_type, eval_every=-1), 3)

    def test_the_selector_refuses_it_too(self) -> None:
        """The head is not the only guard: the selector is usable on its own."""
        train_x, train_y, val_x, val_y = separable_problem()
        prototypes = PrototypeHead(n_classes=3)
        prototypes.fit(train_x, train_y, val_x, val_y)
        with pytest.raises(ValueError, match="eval_every"):
            ValidationSelector(val_x, val_y, prototypes, sample_steps=4, eval_every=-1)


class TestSelectionOff:
    """eval_every = 0 is the pre-selection behaviour, reachable and unchanged."""

    @pytest.mark.parametrize("head_type", HEADS)
    def test_nothing_is_recorded_and_nothing_is_selected(self, head_type: str) -> None:
        head = fitted(head_type, eval_every=0)
        assert head.val_top1_history == []
        assert head.best_epoch is None

    @pytest.mark.parametrize("head_type", HEADS)
    def test_the_head_keeps_the_final_field(self, head_type: str) -> None:
        head = fitted(head_type, eval_every=0)
        assert field_hash(head.field) == field_hash(trained_field(head_type, 60))


class TestSelectionOn:
    @pytest.mark.parametrize("head_type", HEADS)
    def test_validation_top1_is_recorded_on_the_eval_every_grid(self, head_type: str) -> None:
        head = fitted(head_type, eval_every=20)
        steps = [step for step, _ in head.val_top1_history]
        assert steps == [20, 40, 60]
        assert all(0.0 <= accuracy <= 1.0 for _, accuracy in head.val_top1_history)

    @pytest.mark.parametrize("head_type", HEADS)
    def test_the_selected_step_is_the_earliest_argmax(self, head_type: str) -> None:
        head = fitted(head_type, eval_every=20)
        history = head.val_top1_history
        best = max(accuracy for _, accuracy in history)
        assert head.best_epoch == next(step for step, a in history if a == best)

    @pytest.mark.parametrize("head_type", HEADS)
    def test_a_tie_keeps_the_earlier_step(self, head_type: str) -> None:
        """The linear probe's rule (ADR-014), on a problem that ties by design."""
        head = fitted(head_type, eval_every=20)
        history = head.val_top1_history
        assert history[0][1] == max(accuracy for _, accuracy in history)
        assert head.best_epoch == 20

    @pytest.mark.parametrize("head_type", HEADS)
    def test_the_kept_field_is_the_field_as_it_stood_at_the_selected_step(
        self, head_type: str
    ) -> None:
        """Training is one sequence, so the first N steps of a long run are a
        short run: the selected checkpoint must hash to the field a run of
        best_epoch steps produces."""
        head = fitted(head_type, eval_every=20)
        assert field_hash(head.field) == field_hash(trained_field(head_type, head.best_epoch))

    @pytest.mark.parametrize("head_type", HEADS)
    def test_the_kept_field_is_not_the_final_one(self, head_type: str) -> None:
        """Negative control: otherwise the check above would say nothing."""
        head = fitted(head_type, eval_every=20)
        assert head.best_epoch < 60
        assert field_hash(head.field) != field_hash(trained_field(head_type, 60))

    @pytest.mark.parametrize("head_type", HEADS)
    def test_predictions_come_from_the_kept_field(self, head_type: str) -> None:
        _, _, val_x, _ = separable_problem()
        head = fitted(head_type, eval_every=20)
        short = fitted(head_type, eval_every=0, n_train_steps=head.best_epoch)
        assert torch.equal(head.predict(val_x), short.predict(val_x))

    @pytest.mark.parametrize("head_type", HEADS)
    def test_selection_does_not_disturb_the_fit_it_observes(self, head_type: str) -> None:
        """Evaluating the field mid-training must not touch the RNG stream, the
        weights or the optimizer, so the loss curve is the same curve."""
        assert fitted(head_type, eval_every=20).loss_history == pytest.approx(
            fitted(head_type, eval_every=0).loss_history
        )

    @pytest.mark.parametrize("head_type", HEADS)
    def test_selection_is_deterministic_at_a_fixed_seed(self, head_type: str) -> None:
        _, _, val_x, _ = separable_problem()
        first = fitted(head_type, eval_every=20)
        second = fitted(head_type, eval_every=20)
        assert first.val_top1_history == second.val_top1_history
        assert first.best_epoch == second.best_epoch
        assert torch.equal(first.predict(val_x), second.predict(val_x))

    @pytest.mark.parametrize("head_type", HEADS)
    def test_an_unevaluated_tail_is_not_silently_selected(self, head_type: str) -> None:
        """With 60 steps and eval_every 25 the last 10 steps are never scored,
        so no checkpoint past step 50 can be kept."""
        head = fitted(head_type, eval_every=25)
        assert [step for step, _ in head.val_top1_history] == [25, 50]
        assert head.best_epoch in (25, 50)


class TestUnfitted:
    @pytest.mark.parametrize("head_type", HEADS)
    def test_the_selected_step_is_undefined_before_fit(self, head_type: str) -> None:
        from fm_fewshot.services.heads.base import NotFittedError

        with pytest.raises(NotFittedError):
            _ = make_head(make_config(head_type, **BASE_PARAMS), 3).best_epoch

    @pytest.mark.parametrize("head_type", HEADS)
    def test_no_validation_history_before_fit(self, head_type: str) -> None:
        assert make_head(make_config(head_type, **BASE_PARAMS), 3).val_top1_history == []


class TestZeroTrainingSteps:
    @pytest.mark.parametrize("head_type", HEADS)
    def test_nothing_is_selected_when_nothing_is_trained(self, head_type: str) -> None:
        head = fitted(head_type, eval_every=20, n_train_steps=0)
        assert head.val_top1_history == []
        assert head.best_epoch is None
