"""The fairness guard for the standard-versus-rolled-out comparison (TODO 8.4).

The write-up asks us to keep the network architecture and the main training
choices fixed when comparing the two schemes, because the comparison is only
readable if the loss is the one thing that differs. These tests pin that.

What is shared, and asserted here:

- the initial field. Both heads build a VelocityMLP from the same dim,
  hidden_dims, time_conditioning and init_seed, so before the first optimizer
  step their weights are bit-identical.
- the arguments to the training loop. Both `_fit_field` implementations hand
  `train_field` the same endpoints and the same architecture, step count,
  batch size, learning rate and seed. Only the batch-loss callable differs,
  and rolled-out training's depth is closed over in that callable rather than
  passed to the loop.
- the checkpoint selection. Both hand the loop a ValidationSelector carrying
  the same validation split, the same prototypes, the same schedule and the
  same depth, so neither scheme can be selected on easier terms (ADR-028).
- the optimizer. One Adam, built in `base` and never in a scheme, so its type
  and every default it carries agree.
- the row draw. One `torch.randint` call in the shared loop, from a generator
  seeded with init_seed, so both schemes fit their first batch on the same
  rows.

What is not shared, and is asserted here as divergence rather than glossed:

- the batch sequence past the first draw. Standard training samples
  t ~ U(0, 1) from the same generator and rolled-out training draws no times
  at all (ADR-021), so the stream advances differently and batch 2 onward
  differs by construction. Nothing here claims otherwise; the guard is that
  the divergence comes from the objective and from nothing else.

The last two classes are the other half of the same fairness question, ADR-023
and its amendment ADR-028: within standard FM, T = 4 and T = 12 are one trained
field, because standard training never reads the step count, and that stays
bit-identically true wherever selection is off. Once selection runs, T reaches
the fit through the selection metric and the two runs may stop at two different
points of that one trajectory.
"""

import hashlib
from dataclasses import replace

import pytest
import torch
from test_head_contract import make_config

from fm_fewshot.services.flow.objective import cfm_loss
from fm_fewshot.services.flow.training import rolled_out as rolled_module
from fm_fewshot.services.flow.training import standard as standard_module
from fm_fewshot.services.flow.training.base import ValidationSelector, train_field
from fm_fewshot.services.flow.training.rolled_out import rolled_out_loss
from fm_fewshot.services.flow.velocity_mlp import VelocityMLP
from fm_fewshot.services.heads.base import make_head

SHARED_PARAMS: dict[str, object] = {
    "sample_steps": 4,
    "n_train_steps": 2,
    "hidden_dims": [16, 16],
    "batch_size": 8,
    "lr": 7e-4,
    "time_conditioning": "scalar",
}


def problem(n_classes: int = 3, dim: int = 5, per_class: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, dim) * 4.0
    labels = torch.arange(n_classes).repeat_interleave(per_class)
    train_x = centers[labels] + 0.3 * torch.randn(labels.shape[0], dim, generator=g)
    return train_x, labels


def both_heads(**overrides: object):
    """One head per scheme, built from identical head_params."""
    params = dict(SHARED_PARAMS)
    params.update(overrides)
    return (
        make_head(make_config("fm_standard", **params), 3),
        make_head(make_config("fm_rolled", **params), 3),
    )


class TestIdenticalInitialization:
    def test_both_schemes_start_from_the_same_field(self) -> None:
        train_x, train_y = problem()
        standard, rolled = both_heads(n_train_steps=0)
        for head in (standard, rolled):
            head.fit(train_x, train_y, train_x, train_y)

        left = list(standard.field.parameters())
        right = list(rolled.field.parameters())
        assert len(left) == len(right) != 0
        for a, b in zip(left, right, strict=True):
            assert torch.equal(a, b)

    def test_the_comparison_can_fail(self) -> None:
        """Negative control: the check above is not vacuously true.

        Two fields from different init_seeds differ, so parameter equality is
        a real constraint and not an artifact of comparing one object to
        itself.
        """
        train_x, train_y = problem()
        params = dict(SHARED_PARAMS, n_train_steps=0)
        cfg = make_config("fm_standard", **params)
        head = make_head(cfg, 3)
        other = make_head(replace(cfg, init_seed=1), 3)
        head.fit(train_x, train_y, train_x, train_y)
        other.fit(train_x, train_y, train_x, train_y)
        assert any(
            not torch.equal(a, b)
            for a, b in zip(head.field.parameters(), other.field.parameters(), strict=True)
        )


class TestIdenticalLoopArguments:
    """The two positional arguments past the endpoints are the loss and the selector.

    The loss is the one thing that may differ. The selector may not: it is the
    same class carrying the same validation split, prototypes, schedule and
    depth in both schemes, which is checked field by field here because two
    heads necessarily build two objects (ADR-028).
    """

    def _recorded(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple]:
        recorded: dict[str, tuple] = {}

        def recorder(name: str):
            def fake_train_field(x0, x1, batch_loss, selector=None, **kwargs):
                recorded[name] = (x0, x1, selector, kwargs)
                return VelocityMLP(dim=x0.shape[1], hidden_dims=(4,), seed=0), []

            return fake_train_field

        monkeypatch.setattr(standard_module, "train_field", recorder("standard"))
        monkeypatch.setattr(rolled_module, "train_field", recorder("rolled"))

        train_x, train_y = problem()
        standard, rolled = both_heads()
        standard.fit(train_x, train_y, train_x, train_y)
        rolled.fit(train_x, train_y, train_x, train_y)
        return recorded

    def test_the_two_schemes_hand_the_loop_the_same_everything_but_the_loss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = self._recorded(monkeypatch)
        standard_x0, standard_x1, _, standard_kwargs = recorded["standard"]
        rolled_x0, rolled_x1, _, rolled_kwargs = recorded["rolled"]
        assert torch.equal(standard_x0, rolled_x0)
        assert torch.equal(standard_x1, rolled_x1)
        assert standard_kwargs == rolled_kwargs
        # The loop is told nothing about T: rolled-out training closes over its
        # depth inside the loss (ADR-022), so no keyword argument to the shared
        # loop can carry it and the two argument sets stay comparable. The
        # selector carries T because scoring a checkpoint means transporting at
        # it, and it carries the same T in both schemes.
        assert "sample_steps" not in standard_kwargs

    def test_both_schemes_hand_the_loop_the_same_selection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = self._recorded(monkeypatch)
        standard_selector = recorded["standard"][2]
        rolled_selector = recorded["rolled"][2]
        assert type(standard_selector) is type(rolled_selector) is ValidationSelector
        assert standard_selector.eval_every == rolled_selector.eval_every
        assert standard_selector.sample_steps == rolled_selector.sample_steps
        assert torch.equal(standard_selector.val_x, rolled_selector.val_x)
        assert torch.equal(standard_selector.val_y, rolled_selector.val_y)
        # Same decision rule, not merely the same class of rule: the selector
        # scores with the head's own fitted PrototypeHead (ADR-020).
        assert torch.equal(
            standard_selector.classifier.prototypes, rolled_selector.classifier.prototypes
        )


class TestIdenticalOptimizer:
    def test_both_schemes_build_one_adam_with_the_same_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_adam = torch.optim.Adam
        seen: list[tuple[str, dict, list]] = []

        class RecordingAdam(real_adam):  # type: ignore[misc, valid-type]
            def __init__(self, params, **kwargs):
                params = list(params)
                super().__init__(params, **kwargs)
                seen.append((type(self).__name__, dict(self.defaults),
                             [tuple(p.shape) for p in params]))

        monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)

        train_x, train_y = problem()
        standard, rolled = both_heads()
        standard.fit(train_x, train_y, train_x, train_y)
        rolled.fit(train_x, train_y, train_x, train_y)

        assert len(seen) == 2
        assert seen[0] == seen[1]
        assert seen[0][1]["lr"] == SHARED_PARAMS["lr"]


class TestBatchStream:
    """Row draws, observed at the loop where the single randint lives.

    The batch loss receives x0[rows] directly, so with one distinguishable
    value per row the drawn indices are readable from the batch itself.
    """

    ROWS = 32
    STEPS = 4
    BATCH = 8

    def _rows_seen(self, draws_a_time: bool) -> list[list[int]]:
        x0 = torch.arange(self.ROWS, dtype=torch.float32).unsqueeze(1)
        x1 = torch.zeros(self.ROWS, 1)
        seen: list[list[int]] = []

        def batch_loss(field, batch_x0, batch_x1, generator):
            seen.append([int(v) for v in batch_x0.flatten().tolist()])
            if draws_a_time:  # what standard training does, and rolled does not
                torch.rand(batch_x0.shape[0], generator=generator)
            times = torch.zeros(batch_x0.shape[0])
            return (field(batch_x0, times) ** 2).sum(dim=1).mean()

        train_field(
            x0,
            x1,
            batch_loss,
            hidden_dims=(4,),
            n_train_steps=self.STEPS,
            batch_size=self.BATCH,
            init_seed=0,
        )
        return seen

    def test_the_first_batch_is_the_same_rows_in_both_schemes(self) -> None:
        standard_rows = self._rows_seen(draws_a_time=True)
        rolled_rows = self._rows_seen(draws_a_time=False)
        assert standard_rows[0] == rolled_rows[0]
        assert len(standard_rows[0]) == self.BATCH

    def test_the_streams_diverge_after_that_because_only_standard_draws_t(self) -> None:
        """Stated, not hidden: the guard is that this is the only divergence.

        Standard training consumes the generator for t ~ U(0, 1) after each row
        draw, so from batch 2 the two schemes see different rows. Asserting
        identical sequences would be asserting something false.
        """
        standard_rows = self._rows_seen(draws_a_time=True)
        rolled_rows = self._rows_seen(draws_a_time=False)
        assert standard_rows[1:] != rolled_rows[1:]

    def test_the_row_draw_itself_is_shared_and_deterministic(self) -> None:
        """Two losses that touch the generator the same way see the same rows."""
        assert self._rows_seen(draws_a_time=False) == self._rows_seen(draws_a_time=False)
        assert self._rows_seen(draws_a_time=True) == self._rows_seen(draws_a_time=True)


class TestOneLossScale:
    def test_the_two_objectives_agree_on_a_field_that_predicts_nothing(self) -> None:
        """Both are || . ||_2^2 summed over D and averaged over the batch.

        With v_theta = 0 the rollout leaves every point where it started, so
        L_roll is mean_i || z_i - p_{y_i} ||^2, and L_FM is mean_i || u_i ||^2
        for the same pairs. Those are the same number, and they are only the
        same number if the two reductions agree. This is the guard on the
        loss-curve deliverable: one axis, one learning-rate meaning.
        """
        g = torch.Generator().manual_seed(0)
        z = torch.randn(16, 7, generator=g)
        prototypes = torch.randn(16, 7, generator=g)
        zero_field = lambda x, t: torch.zeros_like(x)  # noqa: E731

        standard = float(cfm_loss(zero_field, z, prototypes, torch.rand(16, generator=g)))
        rolled = float(rolled_out_loss(zero_field, z, prototypes, sample_steps=4))
        assert standard == pytest.approx(rolled)
        assert standard == pytest.approx(float(((prototypes - z) ** 2).sum(dim=1).mean()))


def field_hash(field: VelocityMLP) -> str:
    """A digest of the trained weights, the form a finished run can be checked in."""
    digest = hashlib.sha256()
    for name, tensor in sorted(field.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TestStepCountIsNotATrainingAxis:
    """ADR-023: standard training never reads T, so T = 4 and T = 12 are one field.

    Standard FM supervises the velocity at points on the ideal path and never
    solves the ODE, so the step count enters only at what is done with the
    field afterwards. The two rows the grid reports for standard FM come from
    one training trajectory, and the gap between them measures the curvature of
    the field rather than any difference in what was fitted. The phase note
    must not present them as two models.

    Scoped to eval_every = 0, and no further. Bit-identity was ADR-023's claim
    about whole runs; ADR-028 narrows it to the training trajectory, because
    checkpoint selection scores a field by transporting the validation split at
    T, so with selection on the two runs can stop at two different points of
    that one trajectory. The class below is that case.

    Asserted as a hash so the same check can be run over the weights a real
    grid writes, not only over two objects held in one process.
    """

    def _trained(self, head_type: str, sample_steps: int, steps: int = 25) -> VelocityMLP:
        train_x, train_y = problem()
        params = dict(
            SHARED_PARAMS, sample_steps=sample_steps, n_train_steps=steps, eval_every=0
        )
        head = make_head(make_config(head_type, **params), 3)
        head.fit(train_x, train_y, train_x, train_y)
        return head.field

    def test_standard_fm_at_t_4_and_t_12_hashes_to_the_same_field(self) -> None:
        assert field_hash(self._trained("fm_standard", 4)) == field_hash(
            self._trained("fm_standard", 12)
        )

    def test_that_field_is_a_trained_one(self) -> None:
        """Otherwise the equality above would only say both fits did nothing."""
        trained = field_hash(self._trained("fm_standard", 4))
        assert trained != field_hash(self._trained("fm_standard", 4, steps=0))

    def test_rolled_out_training_at_t_4_and_t_12_gives_two_different_fields(self) -> None:
        """The negative control, and ADR-022: there T is the depth being trained."""
        assert field_hash(self._trained("fm_rolled", 4)) != field_hash(
            self._trained("fm_rolled", 12)
        )


class TestStepCountEntersThroughSelection:
    """ADR-028: with selection on, T decides which checkpoint standard FM keeps.

    The training trajectory is still T-independent. What reads T is the
    selection metric, since scoring a checkpoint means transporting the
    validation split at the run's own step count, so the two runs can stop at
    two different points of the same trajectory and the finished fields then
    differ. The claim that a finished standard-FM run is bit-identical across T
    is false wherever eval_every > 0, and this class is what replaces it.

    The problem is deliberately not the separable one used elsewhere in this
    file: classes overlap, so validation accuracy moves during training instead
    of saturating on the first evaluation, which is the regime where the choice
    of checkpoint is a choice at all.
    """

    PARAMS: dict[str, object] = dict(
        SHARED_PARAMS, n_train_steps=400, lr=7e-3, eval_every=50
    )

    @staticmethod
    def _overlapping(seed: int = 2, spread: float = 2.0):
        g = torch.Generator().manual_seed(seed)
        centers = torch.eye(3, 5) * 4.0
        labels = torch.arange(3).repeat_interleave(4)
        train_x = centers[labels] + spread * torch.randn(labels.shape[0], 5, generator=g)
        val_labels = torch.arange(3).repeat_interleave(6)
        val_x = centers[val_labels] + spread * torch.randn(val_labels.shape[0], 5, generator=g)
        return train_x, labels, val_x, val_labels

    def _head(self, sample_steps: int, **overrides: object):
        params = dict(self.PARAMS, sample_steps=sample_steps)
        params.update(overrides)
        head = make_head(make_config("fm_standard", **params), 3)
        head.fit(*self._overlapping())
        return head

    def test_selection_is_deterministic_at_a_fixed_seed(self) -> None:
        first, second = self._head(4), self._head(4)
        assert first.best_epoch == second.best_epoch
        assert field_hash(first.field) == field_hash(second.field)

    def test_the_two_step_counts_can_keep_different_checkpoints(self) -> None:
        four, twelve = self._head(4), self._head(12)
        assert four.best_epoch != twelve.best_epoch
        assert field_hash(four.field) != field_hash(twelve.field)

    def test_both_kept_fields_lie_on_the_one_training_trajectory(self) -> None:
        """What survives of ADR-023: T changes where the run stops, not what it fits.

        A run of n steps is the first n steps of a longer run, so each kept
        field must hash to an unselected fit of exactly best_epoch steps, at
        either step count.
        """
        for head in (self._head(4), self._head(12)):
            unselected = self._head(4, eval_every=0, n_train_steps=head.best_epoch)
            assert field_hash(head.field) == field_hash(unselected.field)
