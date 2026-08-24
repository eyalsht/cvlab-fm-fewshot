"""The ADR-018 scale diagnostics, as arithmetic.

Four quantities per run: the distribution of ||z||, ||z_T - p_y|| on train and
on test, cos(z_T, p_y), and the pairwise cosine among transported points. Each
is a one-line formula and each is asserted here against a hand case, so the
module that reads a real run has nothing left to get wrong except which tensor
it passes.

Reported as median and IQR, never mean and std: contraction is heavy-tailed
(PRD_reverse_flow section 2), and these numbers exist to say whether a tail
collapsed.
"""

import math

import pytest
import torch

from fm_fewshot.services.evaluation.scale_diagnostics import (
    feature_norms,
    pairwise_cosines,
    target_cosines,
    target_distances,
)


class TestFeatureNorms:
    def test_one_norm_per_row(self) -> None:
        x = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        assert torch.allclose(feature_norms(x), torch.tensor([5.0, 2.0]))

    def test_shape_is_the_row_count(self) -> None:
        assert feature_norms(torch.randn(7, 5)).shape == (7,)


class TestTargetDistances:
    def test_euclidean_distance_per_row(self) -> None:
        transported = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        targets = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        assert torch.allclose(target_distances(transported, targets), torch.tensor([1.0, 5.0]))

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            target_distances(torch.randn(3, 2), torch.randn(4, 2))


class TestTargetCosines:
    def test_aligned_is_one_and_orthogonal_is_zero(self) -> None:
        transported = torch.tensor([[2.0, 0.0], [0.0, 5.0]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        assert torch.allclose(
            target_cosines(transported, targets), torch.tensor([1.0, 0.0]), atol=1e-6
        )

    def test_is_scale_free_in_the_transported_point(self) -> None:
        transported = torch.tensor([[1.0, 1.0]])
        targets = torch.tensor([[1.0, 0.0]])
        assert torch.allclose(
            target_cosines(transported * 100.0, targets),
            target_cosines(transported, targets),
            atol=1e-6,
        )


class TestPairwiseCosines:
    def test_holds_every_unordered_pair_and_no_self_pair(self) -> None:
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        values = pairwise_cosines(x)
        root_half = 1.0 / math.sqrt(2.0)
        assert values.shape == (3,)
        assert torch.allclose(
            values.sort().values,
            torch.tensor([0.0, root_half, root_half]).sort().values,
            atol=1e-6,
        )

    def test_identical_directions_give_one(self) -> None:
        x = torch.tensor([[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]])
        assert torch.allclose(pairwise_cosines(x), torch.ones(3), atol=1e-6)

    def test_subsamples_deterministically_above_the_cap(self) -> None:
        x = torch.randn(40, 6)
        first = pairwise_cosines(x, max_points=8, seed=3)
        second = pairwise_cosines(x, max_points=8, seed=3)
        assert first.shape == (28,)  # C(8, 2)
        assert torch.equal(first, second)

    def test_a_different_seed_draws_a_different_subsample(self) -> None:
        x = torch.randn(40, 6)
        assert not torch.equal(
            pairwise_cosines(x, max_points=8, seed=0), pairwise_cosines(x, max_points=8, seed=1)
        )

    def test_refuses_fewer_than_two_points(self) -> None:
        with pytest.raises(ValueError, match="two points"):
            pairwise_cosines(torch.randn(1, 4))
