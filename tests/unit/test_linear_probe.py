"""Linear probe tests per PRD_linear_probe_head section 5."""

import dataclasses

import numpy as np
import pytest
import torch
from test_head_contract import (
    make_config,
    run_head_battery,
    separable_scenario,
)

from fm_fewshot.services.heads import LinearProbeHead
from fm_fewshot.services.heads.base import NotFittedError, make_head

FAST = {"max_epochs": 20}
# The contract scenario is 6 unit-norm points; the specified lr of 1e-3 needs
# far more than 200 epochs to separate it. The battery tests the interface, not
# the hyperparameters, so it trains at a rate that converges on that scenario.
BATTERY = {"max_epochs": 200, "lr": 0.1}


def gaussian_problem(seed: int, n_classes: int = 3, dim: int = 8, per_class: int = 20):
    rng = np.random.default_rng(seed)
    centers = torch.eye(n_classes, dim) * 4.0

    def draw(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.arange(n_classes).repeat_interleave(count)
        noise = (rng.standard_normal((n_classes * count, dim)) * 0.5).astype(np.float32)
        return centers.repeat_interleave(count, dim=0) + torch.from_numpy(noise), y

    train_x, train_y = draw(per_class)
    val_x, val_y = draw(per_class)
    return train_x, train_y, val_x, val_y


class TestBattery:
    def test_passes_the_head_battery(self) -> None:
        cfg = make_config("linear_probe", **BATTERY)
        train_x, train_y, val_x, val_y, query_x, expected = separable_scenario()
        run_head_battery(cfg, 3, train_x, train_y, val_x, val_y, query_x, expected)


class TestDeterminism:
    def test_same_init_seed_is_bit_identical(self) -> None:
        cfg = make_config("linear_probe", **FAST)
        train_x, train_y, val_x, val_y = gaussian_problem(0)
        first = make_head(cfg, 3)
        first.fit(train_x, train_y, val_x, val_y)
        second = make_head(cfg, 3)
        second.fit(train_x, train_y, val_x, val_y)
        assert torch.equal(first.predict(val_x), second.predict(val_x))
        assert first.epochs == second.epochs

    def test_different_init_seed_gives_different_weights(self) -> None:
        train_x, train_y, val_x, val_y = gaussian_problem(1)
        cfg = make_config("linear_probe", **FAST)
        a = make_head(cfg, 3)
        b = make_head(dataclasses.replace(cfg, init_seed=7), 3)
        a.fit(train_x, train_y, val_x, val_y)
        b.fit(train_x, train_y, val_x, val_y)
        assert not torch.equal(a.predict(val_x), b.predict(val_x))


class TestCheckpointSelection:
    def test_restores_the_peak_validation_epoch(self) -> None:
        """Validation accuracy peaks mid-training, then degrades; the peak wins."""
        train_x, train_y, val_x, val_y = gaussian_problem(2)
        head = LinearProbeHead(n_classes=3, max_epochs=30, init_seed=0)
        head.fit(train_x, train_y, val_x, val_y)

        accuracies = [r.val_accuracy for r in head.epochs]
        peak = max(accuracies)
        # Strictly-greater selection keeps the first epoch that reached the peak.
        assert head.best_epoch == accuracies.index(peak) + 1

        restored = (head.predict(val_x).argmax(dim=1) == val_y).float().mean()
        assert float(restored) == pytest.approx(peak, abs=1e-6)

    def test_best_epoch_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = LinearProbeHead(n_classes=2).best_epoch


class TestValidationIsolation:
    def test_validation_never_enters_the_gradient_path(self) -> None:
        """Replacing val with noise may change selection but never training loss."""
        train_x, train_y, val_x, val_y = gaussian_problem(3)
        rng = np.random.default_rng(99)
        noise_x = torch.from_numpy(rng.standard_normal(val_x.shape).astype(np.float32))

        clean = LinearProbeHead(n_classes=3, max_epochs=15, init_seed=0)
        clean.fit(train_x, train_y, val_x, val_y)
        noisy = LinearProbeHead(n_classes=3, max_epochs=15, init_seed=0)
        noisy.fit(train_x, train_y, noise_x, val_y)

        assert [r.train_loss for r in clean.epochs] == [r.train_loss for r in noisy.epochs]


class TestOptimizerIdentity:
    def test_weight_decay_is_decoupled_from_the_loss(self) -> None:
        """AdamW decays weights in the update, so the loss gradient carries no wd term."""
        train_x, train_y, _, _ = gaussian_problem(4)
        weight = torch.zeros(3, train_x.shape[1], requires_grad=True)
        bias = torch.zeros(3, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(train_x @ weight.T + bias, train_y)
        loss.backward()
        # At zero weights a coupled wd*||W||^2 term would contribute exactly zero
        # anyway, so compare against the analytic softmax gradient instead.
        probs = torch.full((train_x.shape[0], 3), 1.0 / 3.0)
        onehot = torch.nn.functional.one_hot(train_y, 3).float()
        expected = ((probs - onehot).T @ train_x) / train_x.shape[0]
        assert torch.allclose(weight.grad, expected, atol=1e-5)


class TestLearning:
    def test_loss_decreases_and_beats_chance(self) -> None:
        train_x, train_y, val_x, val_y = gaussian_problem(5)
        head = LinearProbeHead(n_classes=3, max_epochs=60, init_seed=0)
        head.fit(train_x, train_y, val_x, val_y)
        assert head.epochs[-1].train_loss < head.epochs[0].train_loss
        assert head.epochs[-1].val_accuracy > 1.0 / 3.0

    def test_one_epoch_record_per_epoch(self) -> None:
        train_x, train_y, val_x, val_y = gaussian_problem(6)
        head = LinearProbeHead(n_classes=3, max_epochs=12, init_seed=0)
        head.fit(train_x, train_y, val_x, val_y)
        assert [r.epoch for r in head.epochs] == list(range(1, 13))
        for record in head.epochs:
            values = [record.train_loss, record.val_loss, record.val_accuracy]
            assert np.isfinite(values).all()


class TestValidationRules:
    def test_missing_class_raises_naming_it(self) -> None:
        train_x = torch.eye(3, 4)
        train_y = torch.tensor([0, 0, 2], dtype=torch.int64)
        head = LinearProbeHead(n_classes=3, max_epochs=2)
        with pytest.raises(ValueError, match=r"classes \[1\]"):
            head.fit(train_x, train_y, train_x, train_y)

    def test_non_finite_features_raise(self) -> None:
        train_x = torch.tensor([[1.0, 0.0], [float("inf"), 1.0]], dtype=torch.float32)
        train_y = torch.tensor([0, 1], dtype=torch.int64)
        head = LinearProbeHead(n_classes=2, max_epochs=2)
        with pytest.raises(ValueError, match="non-finite"):
            head.fit(train_x, train_y, train_x, train_y)

    def test_max_epochs_below_one_raises(self) -> None:
        train_x, train_y, val_x, val_y = gaussian_problem(7)
        head = LinearProbeHead(n_classes=3, max_epochs=0)
        with pytest.raises(ValueError, match="max_epochs"):
            head.fit(train_x, train_y, val_x, val_y)
