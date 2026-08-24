"""The run loop and results writer per PRD_evaluation_protocol.

One run is one head, one dataset, one encoder, one training-set size, one seed
pair. It fits on the drawn training subset, lets the head select a checkpoint
on the full official validation split if it has one to select, and reports
top-1 on the complete official test split.

The test cache is opened after fit returns. That ordering is the point: heads
receive train and validation tensors as arguments and have no route to test
rows, so "the test split is for final evaluation only" is a property of the
code rather than a rule someone has to remember.
"""

import csv
import dataclasses
import json
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fm_fewshot.services.data.subsets import balanced_subset
from fm_fewshot.services.evaluation.metrics import predict_labels, top1_accuracy
from fm_fewshot.services.features.cache import read_features
from fm_fewshot.services.heads import make_head
from fm_fewshot.shared.config import save_config
from fm_fewshot.shared.contracts import ExperimentConfig, RunSummary


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_run_id(run_name: str) -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}_{run_name}"


def run_experiment(
    cfg: ExperimentConfig,
    *,
    data_root: Path = Path("data"),
    results_dir: Path = Path("results"),
) -> RunSummary:
    started = time.perf_counter()

    train_x, train_y, train_meta = read_features(
        cfg.dataset, "train", cfg.encoder, data_root=data_root, l2_normalize=cfg.l2_normalize
    )
    n_classes = len(train_meta["class_names"])
    # Built before any expensive work so an unknown head fails immediately.
    head = make_head(cfg, n_classes)

    val_x, val_y, _ = read_features(
        cfg.dataset, "val", cfg.encoder, data_root=data_root, l2_normalize=cfg.l2_normalize
    )

    subset = balanced_subset(train_y, cfg.k, cfg.subset_seed, n_classes, cfg.dataset)

    fit_started = time.perf_counter()
    head.fit(train_x[subset.idx], subset.labels, val_x, val_y)
    fit_seconds = time.perf_counter() - fit_started

    # Only now. Nothing above this line has seen a test row.
    test_x, test_y, test_meta = read_features(
        cfg.dataset, cfg.eval_split, cfg.encoder, data_root=data_root,
        l2_normalize=cfg.l2_normalize,
    )

    predict_started = time.perf_counter()
    logits = head.predict(test_x)
    predict_seconds = time.perf_counter() - predict_started

    if logits.shape[0] != test_meta["N"]:
        raise ValueError(
            f"scored {logits.shape[0]} rows but the {cfg.eval_split} split holds "
            f"{test_meta['N']}; a partial evaluation must not be reported as a full one"
        )

    accuracy = top1_accuracy(logits, test_y)
    predictions = predict_labels(logits)

    summary = RunSummary(
        run_id=make_run_id(cfg.run_name),
        config=cfg,
        git_commit=git_commit(),
        test_top1=accuracy,
        n_test=int(test_meta["N"]),
        n_train=int(subset.idx.shape[0]),
        subset_idx=tuple(int(i) for i in subset.idx.tolist()),
        best_epoch=getattr(head, "best_epoch", None) if _trains(head) else None,
        epochs=list(getattr(head, "epochs", [])),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        wall_seconds=time.perf_counter() - started,
        loss_history=list(getattr(head, "loss_history", [])),
    )
    # Only the head sees its own intermediate training steps, so a periodic
    # val_top1 in loss_curve.csv can only come from the head choosing to record
    # it there (val_top1_history, not part of the FewShotHead contract because
    # most heads have nothing to record). It is read here and nowhere else:
    # it never reaches RunSummary, so whether a head records it cannot move any
    # number the run reports (ADR-024).
    val_top1_history = list(getattr(head, "val_top1_history", []))
    _write_results(summary, predictions.numpy(), results_dir, val_top1_history=val_top1_history)
    return summary


def _trains(head) -> bool:  # noqa: ANN001 - duck-typed head
    """True for heads that selected a checkpoint, so closed-form heads report None.

    Two shapes of evidence because the two trainers count differently: the
    linear probe selects on epochs and records EpochRecords, the FM heads
    select on steps and record a val_top1 history (ADR-028). Either way the
    number in best_epoch is the checkpoint the head kept, and a head that
    selected nothing reports None.
    """
    return bool(getattr(head, "epochs", [])) or bool(getattr(head, "val_top1_history", []))


def _write_results(
    summary: RunSummary,
    predictions: np.ndarray,
    results_dir: Path,
    *,
    val_top1_history: list[tuple[int, float]] | None = None,
) -> None:
    """Write to a temp directory and rename, so an interrupted run leaves nothing."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".partial_", dir=results_dir))
    try:
        save_config(summary.config, staging / "config.yaml")
        payload = dataclasses.asdict(summary)
        payload.pop("epochs")
        payload.pop("loss_history")
        payload["config"] = dataclasses.asdict(summary.config)
        (staging / "summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        np.save(staging / "preds.npy", predictions)

        if summary.epochs:
            with (staging / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])
                for record in summary.epochs:
                    writer.writerow(
                        [record.epoch, record.train_loss, record.val_loss, record.val_accuracy]
                    )

        if summary.loss_history:
            val_by_step = dict(val_top1_history or [])
            with (staging / "loss_curve.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", "train_loss", "val_top1"])
                for step, loss in enumerate(summary.loss_history, start=1):
                    writer.writerow([step, loss, val_by_step.get(step, "")])
        staging.rename(results_dir / summary.run_id)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
