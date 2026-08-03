"""Metrics tests per PRD_evaluation_protocol section 5."""

import math

import pytest
import torch

from fm_fewshot.services.evaluation.metrics import (
    aggregate_cell,
    predict_labels,
    top1_accuracy,
)
from fm_fewshot.shared.contracts import CellSummary


class TestTop1:
    def test_matches_a_hand_computation(self) -> None:
        logits = torch.tensor(
            [[2.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 3.0], [9.0, 0.0, 0.0]]
        )
        labels = torch.tensor([0, 1, 1, 0])  # third row is wrong
        assert top1_accuracy(logits, labels) == pytest.approx(0.75)

    def test_all_correct_and_all_wrong(self) -> None:
        logits = torch.eye(3) * 5.0
        assert top1_accuracy(logits, torch.tensor([0, 1, 2])) == 1.0
        assert top1_accuracy(logits, torch.tensor([1, 2, 0])) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            top1_accuracy(torch.zeros(3, 2), torch.tensor([0, 1]))


class TestTieBreaking:
    def test_all_equal_logits_predict_the_lowest_index(self) -> None:
        """Documented rule; without it a run is not reproducible."""
        logits = torch.zeros(4, 5)
        assert predict_labels(logits).tolist() == [0, 0, 0, 0]

    def test_partial_tie_takes_the_lowest_tied_index(self) -> None:
        logits = torch.tensor([[1.0, 3.0, 3.0, 2.0]])
        assert predict_labels(logits).tolist() == [1]


class TestAggregation:
    def make(self, accuracies: list[float], k: int | None = 5) -> CellSummary:
        return aggregate_cell(
            dataset="dtd",
            encoder="resnet18",
            head="prototype",
            k=k,
            run_ids=tuple(f"run{i}" for i in range(len(accuracies))),
            accuracies=accuracies,
        )

    def test_mean_and_sample_std_match_a_hand_computation(self) -> None:
        cell = self.make([0.40, 0.50, 0.60])
        assert cell.mean == pytest.approx(0.5)
        # Sample std (n-1 denominator): sqrt(((0.1)^2 + 0 + (0.1)^2) / 2) = 0.1
        assert cell.std == pytest.approx(0.1)
        assert cell.n_runs == 3

    def test_uses_the_sample_denominator_not_the_population_one(self) -> None:
        cell = self.make([0.0, 1.0])
        assert cell.std == pytest.approx(math.sqrt(0.5))  # not 0.5

    def test_single_run_reports_zero_std(self) -> None:
        """The full-setting prototype cell is one run; std is not defined."""
        cell = self.make([0.42], k=None)
        assert cell.mean == pytest.approx(0.42)
        assert cell.std == 0.0
        assert cell.n_runs == 1

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            self.make([])

    def test_run_ids_must_match_the_accuracy_count(self) -> None:
        with pytest.raises(ValueError, match="run_ids"):
            aggregate_cell(
                dataset="dtd",
                encoder="resnet18",
                head="prototype",
                k=5,
                run_ids=("a", "b"),
                accuracies=[0.1, 0.2, 0.3],
            )
