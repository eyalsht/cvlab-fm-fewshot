"""Prototype head tests per PRD_prototype_head section 5 (Option A)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from test_head_contract import (
    make_config,
    run_head_battery,
    separable_scenario,
)

from fm_fewshot.services.heads import PrototypeHead
from fm_fewshot.services.heads.base import NotFittedError, make_head


class TestBattery:
    def test_passes_the_head_battery(self) -> None:
        cfg = make_config("prototype")
        run_head_battery(cfg, 3, *separable_scenario())


class TestFormulaExactness:
    def test_inner_normalization_is_applied_before_averaging(self) -> None:
        """The write-up normalizes each feature, then averages, then renormalizes.

        Averaging raw features and normalizing once gives a different prototype
        whenever a class's members have unequal norms. This pins the first.
        """
        train_x = torch.tensor(
            [[10.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=torch.float32,
        )
        train_y = torch.tensor([0, 0, 1], dtype=torch.int64)

        head = PrototypeHead(n_classes=2)
        head.fit(train_x, train_y, train_x, train_y)

        expected = F.normalize(
            torch.stack(
                [
                    F.normalize(train_x[:2], dim=1).mean(dim=0),
                    F.normalize(train_x[2:], dim=1).mean(dim=0),
                ]
            ),
            dim=1,
        )
        assert torch.allclose(head.prototypes, expected, atol=1e-6)

        # The normalize-once alternative differs; assert we did not implement it.
        naive = F.normalize(train_x[:2].mean(dim=0), dim=0)
        assert not torch.allclose(head.prototypes[0], naive, atol=1e-3)

    def test_prototypes_are_unit_norm(self) -> None:
        rng = np.random.default_rng(0)
        train_x = torch.from_numpy(rng.standard_normal((30, 8)).astype(np.float32)) * 5.0
        train_y = torch.arange(3).repeat_interleave(10)
        head = PrototypeHead(n_classes=3)
        head.fit(train_x, train_y, train_x, train_y)
        assert torch.allclose(head.prototypes.norm(dim=1), torch.ones(3), atol=1e-6)


class TestInvariants:
    def test_permuting_training_rows_leaves_logits_identical(self) -> None:
        rng = np.random.default_rng(1)
        train_x = torch.from_numpy(rng.standard_normal((12, 5)).astype(np.float32))
        train_y = torch.arange(3).repeat_interleave(4)
        query_x = torch.from_numpy(rng.standard_normal((6, 5)).astype(np.float32))

        head = PrototypeHead(n_classes=3)
        head.fit(train_x, train_y, train_x, train_y)
        first = head.predict(query_x)

        perm = torch.from_numpy(rng.permutation(12))
        shuffled = PrototypeHead(n_classes=3)
        shuffled.fit(train_x[perm], train_y[perm], train_x, train_y)
        assert torch.allclose(shuffled.predict(query_x), first, atol=1e-6)

    def test_two_fits_are_bit_identical(self) -> None:
        cfg = make_config("prototype")
        rng = np.random.default_rng(2)
        train_x = torch.from_numpy(rng.standard_normal((12, 5)).astype(np.float32))
        train_y = torch.arange(3).repeat_interleave(4)
        query_x = torch.from_numpy(rng.standard_normal((4, 5)).astype(np.float32))

        first = make_head(cfg, 3)
        first.fit(train_x, train_y, train_x, train_y)
        second = make_head(cfg, 3)
        second.fit(train_x, train_y, train_x, train_y)
        assert torch.equal(first.predict(query_x), second.predict(query_x))

    def test_recovers_separable_structure(self) -> None:
        rng = np.random.default_rng(3)
        centers = torch.eye(5, 16) * 3.0
        train_x = centers.repeat_interleave(10, dim=0) + torch.from_numpy(
            (rng.standard_normal((50, 16)) * 0.2).astype(np.float32)
        )
        train_y = torch.arange(5).repeat_interleave(10)
        test_x = centers.repeat_interleave(20, dim=0) + torch.from_numpy(
            (rng.standard_normal((100, 16)) * 0.2).astype(np.float32)
        )
        test_y = torch.arange(5).repeat_interleave(20)

        head = PrototypeHead(n_classes=5)
        head.fit(train_x, train_y, train_x, train_y)
        accuracy = (head.predict(test_x).argmax(dim=1) == test_y).float().mean()
        assert accuracy > 0.95


class TestValidation:
    def test_missing_class_raises_naming_it(self) -> None:
        train_x = torch.eye(3, 4)
        train_y = torch.tensor([0, 0, 2], dtype=torch.int64)
        head = PrototypeHead(n_classes=3)
        with pytest.raises(ValueError, match="class 1 has no training example"):
            head.fit(train_x, train_y, train_x, train_y)

    def test_zero_norm_row_raises_instead_of_producing_nan(self) -> None:
        train_x = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        train_y = torch.tensor([0, 1], dtype=torch.int64)
        head = PrototypeHead(n_classes=2)
        with pytest.raises(ValueError, match="row 1 has zero norm"):
            head.fit(train_x, train_y, train_x, train_y)

    def test_non_finite_features_raise(self) -> None:
        train_x = torch.tensor([[1.0, 0.0], [float("nan"), 1.0]], dtype=torch.float32)
        train_y = torch.tensor([0, 1], dtype=torch.int64)
        head = PrototypeHead(n_classes=2)
        with pytest.raises(ValueError, match="non-finite"):
            head.fit(train_x, train_y, train_x, train_y)

    def test_prototypes_property_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = PrototypeHead(n_classes=2).prototypes
