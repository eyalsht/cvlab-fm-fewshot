"""Subset sampler tests per PRD_subset_sampler section 5.

The sampler replaces the episode sampler: the unit is a balanced K-per-class
draw from the official train split, not an N-way episode.
"""

import hashlib
import subprocess
import sys

import pytest
import torch

from fm_fewshot.services.data.subsets import (
    InsufficientClassError,
    balanced_subset,
)

N_CLASSES = 6
PER_CLASS = 30


def labels_fixture(n_classes: int = N_CLASSES, per_class: int = PER_CLASS) -> torch.Tensor:
    """Class-blocked labels, the order a feature cache would hand over."""
    return torch.arange(n_classes, dtype=torch.int64).repeat_interleave(per_class)


def draw(k: int | None, seed: int, labels: torch.Tensor | None = None):
    labels = labels_fixture() if labels is None else labels
    return balanced_subset(labels, k, seed, N_CLASSES, "dtd")


class TestDeterminism:
    def test_two_calls_give_identical_indices(self) -> None:
        assert torch.equal(draw(5, 0).idx, draw(5, 0).idx)

    def test_identical_across_processes(self) -> None:
        """Cross-process determinism, the check the episode sampler used."""
        script = (
            "import hashlib, torch;"
            "from fm_fewshot.services.data.subsets import balanced_subset;"
            f"labels = torch.arange({N_CLASSES}, dtype=torch.int64)"
            f".repeat_interleave({PER_CLASS});"
            f'subset = balanced_subset(labels, 10, 1, {N_CLASSES}, "dtd");'
            "print(hashlib.sha256(subset.idx.numpy().tobytes()).hexdigest())"
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        here = hashlib.sha256(draw(10, 1).idx.numpy().tobytes()).hexdigest()
        assert out.stdout.strip() == here


class TestBalanceAndCoverage:
    @pytest.mark.parametrize("k", [5, 10])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_every_class_appears_exactly_k_times(self, k: int, seed: int) -> None:
        subset = draw(k, seed)
        counts = torch.bincount(subset.labels, minlength=N_CLASSES)
        assert torch.equal(counts, torch.full((N_CLASSES,), k))

    @pytest.mark.parametrize("k", [5, 10])
    def test_all_classes_are_covered(self, k: int) -> None:
        assert sorted(set(draw(k, 0).labels.tolist())) == list(range(N_CLASSES))

    def test_labels_match_the_source_at_those_rows(self) -> None:
        labels = labels_fixture()
        subset = draw(5, 0, labels)
        assert torch.equal(subset.labels, labels[subset.idx])


class TestSeedSeparation:
    def test_three_seeds_give_three_distinct_subsets(self) -> None:
        drawn = [tuple(draw(5, s).idx.tolist()) for s in (0, 1, 2)]
        assert len(set(drawn)) == 3

    def test_k5_is_not_a_prefix_of_k10_at_the_same_seed(self) -> None:
        """Independent draws, deliberately: nesting would correlate the runs."""
        five = set(draw(5, 0).idx.tolist())
        ten = set(draw(10, 0).idx.tolist())
        assert not five.issubset(ten)


class TestFullSetting:
    def test_k_none_returns_the_whole_split(self) -> None:
        labels = labels_fixture()
        subset = draw(None, 0, labels)
        assert torch.equal(subset.idx, torch.arange(labels.shape[0], dtype=torch.int64))
        assert torch.equal(subset.labels, labels)
        assert subset.k is None

    def test_k_none_ignores_the_seed(self) -> None:
        assert torch.equal(draw(None, 0).idx, draw(None, 2).idx)


class TestOrdering:
    @pytest.mark.parametrize("k", [5, 10])
    def test_indices_are_strictly_ascending(self, k: int) -> None:
        idx = draw(k, 0).idx
        assert torch.all(idx[1:] > idx[:-1])

    def test_indices_are_int64(self) -> None:
        subset = draw(5, 0)
        assert subset.idx.dtype == torch.int64
        assert subset.labels.dtype == torch.int64


class TestValidation:
    def test_undersized_class_names_it_with_both_counts(self) -> None:
        labels = torch.tensor([0] * 10 + [1] * 3, dtype=torch.int64)
        with pytest.raises(InsufficientClassError) as excinfo:
            balanced_subset(labels, 5, 0, 2, "dtd")
        message = str(excinfo.value)
        assert "class 1" in message
        assert "3" in message
        assert "5" in message

    def test_missing_class_raises(self) -> None:
        labels = torch.tensor([0] * 10, dtype=torch.int64)
        with pytest.raises(InsufficientClassError):
            balanced_subset(labels, 5, 0, 2, "dtd")

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be"):
            draw(k, 0)

    def test_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            draw(5, -1)

    def test_single_class_raises(self) -> None:
        labels = torch.zeros(10, dtype=torch.int64)
        with pytest.raises(ValueError, match="n_classes"):
            balanced_subset(labels, 5, 0, 1, "dtd")


class TestContract:
    def test_carries_its_provenance(self) -> None:
        subset = balanced_subset(labels_fixture(), 5, 2, N_CLASSES, "fgvc_aircraft")
        assert subset.dataset == "fgvc_aircraft"
        assert subset.k == 5
        assert subset.seed == 2
        assert subset.n_classes == N_CLASSES
