"""Data contracts per PLAN section 5.

All contracts are frozen: a subset or a result is a fact, never mutated after
creation. Tensor fields compare by identity under dataclass eq, which is
acceptable because equality checks in tests and the report path only ever
compare the tensor-free contracts (ExperimentConfig, EpochRecord, CellSummary).

The unit of measurement is a run, not an episode: one training subset, one
head, one seed pair, scored by top-1 on the complete official test split.
"""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True)
class TrainSubset:
    """A balanced K-per-class draw from the official train split (ADR-012)."""

    dataset: str  # "dtd" | "fgvc_aircraft"
    k: int | None  # images per class; None means the full official train split
    seed: int
    n_classes: int
    idx: Tensor  # int64, ascending, rows into the train feature cache
    labels: Tensor  # int64 [len(idx)], global class ids at those rows


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    dataset: str  # "dtd" | "fgvc_aircraft"
    encoder: str  # "resnet18" | "dinov2_vits14"
    head: str  # registry key
    head_params: dict[str, object] = field(default_factory=dict)
    # Distinguishes two configurations of one head, e.g. the same FM head at
    # T = 4 and at T = 12. Empty when a head has only one configuration in the
    # protocol. It labels the run and the table cell; without it two step
    # counts would aggregate into one cell.
    variant: str = ""
    k: int | None = 5  # None -> the full official train split
    # Two seed streams, kept apart so the full setting's three initialization
    # seeds vary the classifier while the training subset stays fixed.
    subset_seed: int = 0
    init_seed: int = 0
    seed: int = 0  # any other randomness
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    l2_normalize: bool = False  # ADR-004; the prototype head normalizes anyway
    eval_split: str = "test"  # only the evaluation loop reads it


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    config: ExperimentConfig
    git_commit: str
    test_top1: float  # the reported number
    n_test: int
    n_train: int  # size of the training subset actually used
    # The exact rows fitted on. Recorded rather than hashed so a cross-head
    # comparison can be checked by reading two summaries (ADR-012).
    subset_idx: tuple[int, ...]
    best_epoch: int | None  # ADR-014; None for closed-form heads
    epochs: list[EpochRecord]  # empty for closed-form heads
    fit_seconds: float
    predict_seconds: float
    wall_seconds: float
    # Per-step training loss; empty for heads with no stepwise training loop,
    # e.g. prototype (ADR-024). Written to loss_curve.csv, not epochs.csv:
    # FM heads have one scalar loss and nothing to select.
    loss_history: list[float] = field(default_factory=list)
    # sha256 of the fitted linear map, for heads that fit one: the Stage 1
    # probe and both Stage 3 heads, which refit it internally (ADR-031). None
    # for every other head, and for runs written before the field existed.
    classifier_digest: str | None = None
    # Whether that map is the Stage 1 probe at this setting, untouched. False
    # only for a run of Stage 3's optional extension (FR23), which trains the
    # classifier alongside the field, so the digest above is a classifier of
    # the run's own. None for heads that fit no linear map and for runs written
    # before the field existed; both are read as frozen by subset_check, which
    # is what every one of those runs was.
    classifier_frozen: bool | None = None


@dataclass(frozen=True)
class CellSummary:
    """One table cell, aggregated by report over the protocol's runs."""

    dataset: str
    encoder: str
    head: str
    k: int | None
    run_ids: tuple[str, ...]
    mean: float
    std: float  # sample std over runs; 0.0 and flagged when n_runs == 1
    n_runs: int
