"""Stage 3's optional extension: unfreezing the classifier (FR23).

    "After completing the frozen-classifier experiments, you may also unfreeze
    the pretrained linear classifier and jointly optimize the FM transformation
    and classifier. [...] different learning rates for the FM and classifier,
    delayed unfreezing, or additional regularization."

It is an option on Strategy 1's head, not a second head and not a second
training loop. Four claims carry the design and each is asserted here.

- **Frozen is the default.** `joint_classifier` is off, so every graded row is
  the fit it was before this existed. The frozen assertions in
  `test_fm_prelinear.py` and `test_rolled_out_ce.py` still hold untouched;
  what is added here is the same claim stated against the new switch.
- **Unfrozen means the gradient reaches W and b**, and they move.
- **Delayed unfreezing is exact at the boundary.** The classifier is untouched
  for steps 1..N and moves from step N+1, asserted per step inside one fit
  rather than by comparing two finished runs.
- **The two learning rates land in two parameter groups**, the field's and the
  classifier's, so "different learning rates" is a fact about the optimizer.

One consequence that is not about training. A run whose classifier moved must
not publish the Stage 1 probe's digest: `subset_check` compares digests across
runs at one setting to prove every Stage 3 head transported into the same
frozen map, and a joint run claiming that digest would make the guard pass over
exactly the case it exists to catch. The head reports the classifier it
actually predicts with, and declares `classifier_frozen = False` so the guard
can hold it to a different rule. That half is tested in
`test_check_subsets.py`.
"""

import hashlib

import pytest
import torch
from test_head_contract import make_config, run_head_battery, separable_scenario

from fm_fewshot.services.flow.training.base import ValidationSelector
from fm_fewshot.services.flow.training.rolled_out_ce import (
    JointClassifier,
    RolledOutCeConfig,
    rolled_out_ce_loss,
    train_rolled_out_ce_field,
)
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import NotFittedError, make_head
from fm_fewshot.services.heads.linear_probe import LinearProbeHead

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
JOINT: dict[str, object] = dict(PARAMS, joint_classifier=True, classifier_lr=5e-3)


def problem(n_classes: int = 3, dim: int = 5, per_class: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 4.0
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    train_x = centers[labels] + 0.3 * torch.randn(labels.shape[0], dim, generator=g)
    return train_x, labels


def stage1_probe() -> LinearProbeHead:
    """The Stage 1 probe at this setting, fitted with no FM head involved."""
    probe = LinearProbeHead(n_classes=3, init_seed=0, **FAST_PROBE)
    train_x, train_y = problem()
    probe.fit(train_x, train_y, train_x, train_y)
    return probe


def fitted(**overrides: object):
    head = make_head(make_config("fm_prelinear_ce", **dict(PARAMS, **overrides)), 3)
    train_x, train_y = problem()
    head.fit(train_x, train_y, train_x, train_y)
    return head


def field(dim: int = 5, seed: int = 0) -> VelocityMLP:
    return VelocityMLP(dim=dim, hidden_dims=(16, 16), seed=seed)


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class TestFrozenIsTheDefault:
    """Every graded row is the fit it was before the switch existed."""

    def test_the_config_defaults_are_the_frozen_setting(self) -> None:
        config = RolledOutCeConfig()
        assert config.joint_classifier is False
        assert config.classifier_lr is None
        assert config.unfreeze_at == 0

    def test_a_default_head_reads_the_frozen_setting(self) -> None:
        head = make_head(make_config("fm_prelinear_ce", **PARAMS), 3)
        assert head.ce_config == RolledOutCeConfig()

    def test_the_default_fit_leaves_the_stage_1_probe_bit_identical(self) -> None:
        head = fitted()
        reference = stage1_probe()
        assert torch.equal(head.probe.weight, reference.weight)
        assert torch.equal(head.probe.bias, reference.bias)
        assert head.classifier_digest == reference.classifier_digest

    def test_the_default_run_declares_its_classifier_frozen(self) -> None:
        assert fitted().classifier_frozen is True
        assert stage1_probe().classifier_frozen is True

    def test_an_unfitted_probe_is_frozen(self) -> None:
        assert LinearProbeHead(n_classes=3).classifier_frozen is True

    def test_the_loss_detaches_the_classifier_unless_told_otherwise(self) -> None:
        train_x, train_y = problem()
        weight, bias = stage1_probe().unfreeze()
        rolled_out_ce_loss(field(), train_x, train_y, weight, bias, sample_steps=4).backward()
        assert weight.grad is None
        assert bias.grad is None


class TestUnfreezing:
    def test_the_gradient_reaches_w_and_b(self) -> None:
        train_x, train_y = problem()
        weight, bias = stage1_probe().unfreeze()
        rolled_out_ce_loss(
            field(), train_x, train_y, weight, bias, sample_steps=4, freeze_classifier=False
        ).backward()
        for tensor in (weight, bias):
            assert tensor.grad is not None
            assert tensor.grad.abs().sum() > 0

    def test_w_and_b_move(self) -> None:
        head = fitted(**JOINT)
        reference = stage1_probe()
        assert not torch.equal(head.probe.weight, reference.weight)
        assert not torch.equal(head.probe.bias, reference.bias)

    def test_the_field_still_moves_too(self) -> None:
        """Joint means both, not the classifier taking over the whole job."""
        train_x, _ = problem()
        head = fitted(**JOINT)
        assert not torch.equal(head.transport(train_x), train_x)

    def test_the_digest_is_the_classifier_the_run_predicts_with(self) -> None:
        head = fitted(**JOINT)
        assert head.classifier_digest != stage1_probe().classifier_digest
        assert head.classifier_digest == head.probe.classifier_digest

    def test_the_run_declares_its_classifier_unfrozen(self) -> None:
        assert fitted(**JOINT).classifier_frozen is False

    def test_the_probe_is_left_detached_after_the_fit(self) -> None:
        """Training is over, so the map is a fact again and predict is a read."""
        head = fitted(**JOINT)
        for tensor in (head.probe.weight, head.probe.bias):
            assert tensor.requires_grad is False
            assert tensor.grad is None

    def test_a_probe_that_was_unfrozen_stays_labelled_after_refreezing(self) -> None:
        """The label is about provenance, not about the current requires_grad."""
        probe = stage1_probe()
        probe.unfreeze()
        probe.refreeze()
        assert probe.classifier_frozen is False

    def test_unfreeze_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            LinearProbeHead(n_classes=3).unfreeze()

    def test_refreeze_before_fit_is_a_no_op(self) -> None:
        probe = LinearProbeHead(n_classes=3)
        probe.refreeze()
        assert probe.classifier_frozen is True

    def test_the_fit_is_deterministic_at_a_fixed_seed(self) -> None:
        first, second = fitted(**JOINT), fitted(**JOINT)
        train_x, _ = problem()
        assert torch.equal(first.predict(train_x), second.predict(train_x))
        assert first.classifier_digest == second.classifier_digest


UNFREEZE_AT = 6


class TestDelayedUnfreezing:
    """Untouched for steps 1..N, moving from step N+1, asserted inside one fit.

    `on_step` fires after `backward` and before `optimizer.step`, so the call
    for step s sees the gradient of step s and the weights left by step s-1.
    That is the only vantage point from which both halves of the claim are
    visible at once.
    """

    @staticmethod
    def _observed(unfreeze_at: int, n_train_steps: int) -> list[tuple[int, bool, str]]:
        train_x, train_y = problem()
        probe = stage1_probe()
        joint = JointClassifier(*probe.unfreeze(), unfreeze_at=unfreeze_at)
        seen: list[tuple[int, bool, str]] = []
        train_rolled_out_ce_field(
            train_x,
            train_y,
            joint.weight,
            joint.bias,
            None,
            sample_steps=4,
            hidden_dims=(16, 16),
            n_train_steps=n_train_steps,
            batch_size=8,
            lr=5e-3,
            init_seed=0,
            joint=joint,
            on_step=lambda step, _norm: seen.append(
                (step, joint.weight.grad is None, digest(joint.weight))
            ),
        )
        return seen

    def test_no_gradient_reaches_the_classifier_before_the_boundary(self) -> None:
        seen = self._observed(UNFREEZE_AT, UNFREEZE_AT + 3)
        without = [step for step, no_grad, _ in seen if no_grad]
        assert without == list(range(1, UNFREEZE_AT + 1))

    def test_w_is_bit_identical_through_the_boundary_and_moves_after(self) -> None:
        seen = self._observed(UNFREEZE_AT, UNFREEZE_AT + 3)
        start = seen[0][2]
        # The call at step N+1 reads the weights left by step N, so N+1 calls
        # see the untouched map and the next one must not.
        assert [d for step, _, d in seen if step <= UNFREEZE_AT + 1] == [start] * (
            UNFREEZE_AT + 1
        )
        assert seen[UNFREEZE_AT + 1][2] != start

    def test_zero_delay_moves_the_classifier_from_the_first_step(self) -> None:
        seen = self._observed(0, 3)
        assert [no_grad for _, no_grad, _ in seen] == [False, False, False]
        assert seen[1][2] != seen[0][2]

    def test_the_gate_is_exact_at_the_step_number(self) -> None:
        joint = JointClassifier(*stage1_probe().unfreeze(), unfreeze_at=UNFREEZE_AT)
        joint.open_at(UNFREEZE_AT)
        assert joint.weight.requires_grad is False
        joint.open_at(UNFREEZE_AT + 1)
        assert joint.weight.requires_grad is True

    def test_a_negative_delay_is_refused(self) -> None:
        weight, bias = stage1_probe().unfreeze()
        with pytest.raises(ValueError, match="unfreeze_at"):
            JointClassifier(weight, bias, unfreeze_at=-1)

    def test_a_delay_past_the_budget_leaves_the_classifier_untouched(self) -> None:
        """The honest degenerate case: it is a frozen run that says it is not."""
        head = fitted(**dict(JOINT, n_train_steps=4, unfreeze_at=10))
        assert torch.equal(head.probe.weight, stage1_probe().weight)
        assert head.classifier_frozen is False


def _record_adam(monkeypatch: pytest.MonkeyPatch) -> list[list[dict]]:
    """Capture the parameter groups of every Adam built while fitting."""
    real_adam = torch.optim.Adam
    seen: list[list[dict]] = []

    class RecordingAdam(real_adam):  # type: ignore[misc, valid-type]
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            seen.append(self.param_groups)

    monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)
    return seen


class TestSeparateLearningRates:
    def test_the_field_and_the_classifier_are_two_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _record_adam(monkeypatch)
        fitted(**dict(JOINT, lr=5e-3, classifier_lr=1e-4))
        # One Adam over the field and the classifier. The probe's own AdamW is
        # a different optimizer on different tensors and is not recorded here.
        assert len(seen) == 1
        groups = seen[0]
        assert len(groups) == 2
        assert groups[0]["lr"] == pytest.approx(5e-3)
        assert groups[1]["lr"] == pytest.approx(1e-4)
        assert [tuple(p.shape) for p in groups[1]["params"]] == [(3, 5), (3,)]

    def test_the_classifier_group_holds_the_probes_own_tensors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not a copy: the selector scores with the probe, so the map the
        optimizer moves has to be the map the probe predicts with."""
        seen = _record_adam(monkeypatch)
        head = fitted(**JOINT)
        groups = seen[0]
        assert torch.equal(groups[1]["params"][0], head.probe.weight)
        assert torch.equal(groups[1]["params"][1], head.probe.bias)

    def test_an_unset_classifier_lr_follows_the_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _record_adam(monkeypatch)
        fitted(**dict(JOINT, lr=5e-3, classifier_lr=None))
        assert seen[0][1]["lr"] == pytest.approx(5e-3)

    def test_the_frozen_fit_builds_one_group_over_the_field_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _record_adam(monkeypatch)
        fitted()
        assert len(seen) == 1
        assert len(seen[0]) == 1


class _Trackable:
    """The smallest thing a selector can checkpoint alongside the field."""

    def __init__(self) -> None:
        self.value = 0.0

    def snapshot(self) -> float:
        return self.value

    def restore(self, state: float) -> None:
        self.value = state


class _ScriptedClassifier:
    """Scores a fixed accuracy sequence, one entry per call to predict."""

    def __init__(self, val_y: torch.Tensor, accuracies: list[float]) -> None:
        self._val_y = val_y
        self._accuracies = list(accuracies)
        self._calls = 0

    def predict(self, query_x: torch.Tensor) -> torch.Tensor:
        accuracy = self._accuracies[min(self._calls, len(self._accuracies) - 1)]
        self._calls += 1
        n_right = round(accuracy * self._val_y.shape[0])
        logits = torch.zeros(self._val_y.shape[0], 3)
        for row, label in enumerate(self._val_y.tolist()):
            logits[row, label if row < n_right else (label + 1) % 3] = 1.0
        return logits


class TestCheckpointSelectionCoversBoth:
    """The kept checkpoint is the field **and** the classifier of one step.

    Restoring the field of step s next to the classifier of the last step
    would leave the head predicting with a system that was never scored, and
    `best_epoch` would name a step whose accuracy the run does not have.
    """

    @staticmethod
    def _selector(accuracies: list[float]) -> ValidationSelector:
        val_x, val_y = problem()
        return ValidationSelector(
            val_x,
            val_y,
            _ScriptedClassifier(val_y, accuracies),
            sample_steps=1,
            eval_every=1,
        )

    def test_the_selector_restores_what_it_was_told_to_track(self) -> None:
        recorder = _Trackable()
        selector = self._selector([0.0, 1.0, 0.5])
        selector.track(recorder)
        net = VelocityMLP(dim=5, hidden_dims=(4,), seed=0)
        for step in (1, 2, 3):
            recorder.value = float(step)
            selector.observe(step, net)
        recorder.value = 99.0
        selector.restore(net)
        assert selector.best_step == 2
        assert recorder.value == 2.0

    def test_an_untracked_selector_restores_only_the_field(self) -> None:
        selector = self._selector([1.0])
        net = VelocityMLP(dim=5, hidden_dims=(4,), seed=0)
        selector.observe(1, net)
        selector.restore(net)  # no extra state; must not raise
        assert selector.best_step == 1

    def test_the_kept_classifier_is_the_one_from_the_selected_step(self) -> None:
        selected = fitted(**dict(JOINT, n_train_steps=12, eval_every=1))
        stopped_there = fitted(**dict(JOINT, n_train_steps=selected.best_epoch, eval_every=0))
        assert selected.classifier_digest == stopped_there.classifier_digest


class TestTheContractBatteryPassesBothWays:
    @pytest.mark.parametrize("joint", [False, True])
    def test_the_head_passes_the_battery(self, joint: bool) -> None:
        cfg = make_config(
            "fm_prelinear_ce",
            **dict(
                PARAMS,
                n_train_steps=5,
                probe_params={},
                joint_classifier=joint,
                classifier_lr=1e-4,
            ),
        )
        run_head_battery(cfg, 3, *separable_scenario())


class TestTheHeadReadsTheJointParameters:
    def test_the_three_knobs_reach_the_config(self) -> None:
        cfg = make_config(
            "fm_prelinear_ce",
            **dict(PARAMS, joint_classifier=True, classifier_lr=1e-4, unfreeze_at=500),
        )
        head = make_head(cfg, 3)
        assert head.ce_config == RolledOutCeConfig(
            joint_classifier=True, classifier_lr=1e-4, unfreeze_at=500
        )

    def test_joint_training_refuses_the_row_space_projection(self) -> None:
        """The projector is built once from W, so a moving W makes it stale.

        Refused rather than silently wrong: the ablation and the extension ask
        different questions and combining them answers neither.
        """
        with pytest.raises(ValueError, match="project_velocity"):
            RolledOutCeConfig(joint_classifier=True, project_velocity=True)
