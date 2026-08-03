"""Top-1 accuracy and cell aggregation per PRD_evaluation_protocol.

The reported number is top-1 on the complete official test split. Cells
aggregate over the protocol's runs as mean and sample standard deviation. No
confidence intervals: three runs do not support them and none were asked for.
"""

import statistics

import torch
from torch import Tensor

from fm_fewshot.shared.contracts import CellSummary


def predict_labels(logits: Tensor) -> Tensor:
    """Argmax with the lowest tied index winning.

    torch.argmax already returns the first maximal index, but the rule is
    asserted by a test and stated here because a run's reproducibility depends
    on ties resolving the same way every time.
    """
    return logits.argmax(dim=1)


def top1_accuracy(logits: Tensor, labels: Tensor) -> float:
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits has {logits.shape[0]} rows but labels has {labels.shape[0]}"
        )
    return float((predict_labels(logits) == labels).float().mean())


def aggregate_cell(
    *,
    dataset: str,
    encoder: str,
    head: str,
    k: int | None,
    run_ids: tuple[str, ...],
    accuracies: list[float],
) -> CellSummary:
    if not accuracies:
        raise ValueError("a cell needs at least one run to aggregate")
    if len(run_ids) != len(accuracies):
        raise ValueError(
            f"run_ids has {len(run_ids)} entries but {len(accuracies)} accuracies "
            "were given; every accuracy must be traceable to its run"
        )
    # Sample std (n-1). A single run has no spread, and reporting 0.0 is honest
    # where reporting a population std of 0.0 would look like a measurement.
    std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    return CellSummary(
        dataset=dataset,
        encoder=encoder,
        head=head,
        k=k,
        run_ids=tuple(run_ids),
        mean=statistics.fmean(accuracies),
        std=std,
        n_runs=len(accuracies),
    )


def confusion_matrix(predictions: Tensor, labels: Tensor, n_classes: int) -> Tensor:
    """Row-normalized confusion matrix; rows are true classes (PRD_figures F3)."""
    counts = torch.zeros(n_classes, n_classes, dtype=torch.float64)
    for true, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        counts[true, predicted] += 1
    totals = counts.sum(dim=1, keepdim=True)
    return torch.where(totals > 0, counts / totals, counts)
