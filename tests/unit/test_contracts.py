"""Contract tests: frozen dataclasses and config yaml round-trip per PLAN section 5."""

import dataclasses
from pathlib import Path

import pytest
import torch

from fm_fewshot.shared.config import load_config, save_config
from fm_fewshot.shared.contracts import (
    CellSummary,
    EpochRecord,
    ExperimentConfig,
    RunSummary,
    TrainSubset,
)


def make_subset(**overrides: object) -> TrainSubset:
    fields: dict = {
        "dataset": "dtd",
        "k": 5,
        "seed": 0,
        "n_classes": 3,
        "idx": torch.tensor([0, 4, 9], dtype=torch.int64),
        "labels": torch.tensor([0, 1, 2], dtype=torch.int64),
    }
    fields.update(overrides)
    return TrainSubset(**fields)


def make_config(**overrides: object) -> ExperimentConfig:
    fields: dict = {
        "run_name": "smoke",
        "dataset": "dtd",
        "encoder": "resnet18",
        "head": "prototype",
        "head_params": {},
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def make_summary(**overrides: object) -> RunSummary:
    fields: dict = {
        "run_id": "20260803T120000_smoke",
        "config": make_config(),
        "git_commit": "abc1234",
        "test_top1": 0.5,
        "n_test": 1880,
        "n_train": 235,
        "best_epoch": None,
        "epochs": [],
        "fit_seconds": 0.01,
        "predict_seconds": 0.01,
        "wall_seconds": 1.0,
    }
    fields.update(overrides)
    return RunSummary(**fields)


class TestFrozen:
    def test_train_subset_is_frozen(self) -> None:
        subset = make_subset()
        with pytest.raises(dataclasses.FrozenInstanceError):
            subset.dataset = "fgvc_aircraft"  # type: ignore[misc]

    def test_experiment_config_is_frozen(self) -> None:
        cfg = make_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.subset_seed = 1  # type: ignore[misc]

    def test_epoch_record_is_frozen(self) -> None:
        record = EpochRecord(epoch=1, train_loss=2.0, val_loss=2.1, val_accuracy=0.1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.val_accuracy = 1.0  # type: ignore[misc]

    def test_run_summary_is_frozen(self) -> None:
        summary = make_summary()
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.test_top1 = 1.0  # type: ignore[misc]

    def test_cell_summary_is_frozen(self) -> None:
        cell = CellSummary(
            dataset="dtd",
            encoder="resnet18",
            head="prototype",
            k=5,
            run_ids=("a", "b", "c"),
            mean=0.5,
            std=0.01,
            n_runs=3,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cell.mean = 0.0  # type: ignore[misc]


class TestDefaults:
    def test_experiment_config_defaults_match_plan(self) -> None:
        cfg = make_config()
        assert cfg.k == 5
        assert cfg.subset_seed == 0
        assert cfg.init_seed == 0
        assert cfg.seed == 0
        assert cfg.device == "auto"
        assert cfg.l2_normalize is False
        assert cfg.eval_split == "test"

    def test_full_setting_is_k_none(self) -> None:
        cfg = make_config(k=None)
        assert cfg.k is None

    def test_subset_holds_int64_indices(self) -> None:
        subset = make_subset()
        assert subset.idx.dtype == torch.int64
        assert subset.labels.dtype == torch.int64

    def test_closed_form_run_has_no_epochs(self) -> None:
        summary = make_summary()
        assert summary.epochs == []
        assert summary.best_epoch is None


class TestConfigYaml:
    def test_round_trip_preserves_every_field(self, tmp_path: Path) -> None:
        cfg = make_config(k=10, subset_seed=2, init_seed=1, head="linear_probe")
        path = tmp_path / "config.yaml"
        save_config(cfg, path)
        assert load_config(path) == cfg

    def test_round_trip_preserves_the_full_setting(self, tmp_path: Path) -> None:
        cfg = make_config(k=None)
        path = tmp_path / "config.yaml"
        save_config(cfg, path)
        assert load_config(path).k is None

    def test_defaults_fill_omitted_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "run_name: smoke\ndataset: dtd\nencoder: resnet18\n"
            "head: prototype\nhead_params: {}\n",
            encoding="utf-8",
        )
        cfg = load_config(path)
        assert cfg.k == 5
        assert cfg.subset_seed == 0

    def test_unknown_key_raises_with_its_name(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "run_name: smoke\ndataset: dtd\nencoder: resnet18\n"
            "head: prototype\nhead_params: {}\nn_way: 5\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="n_way"):
            load_config(path)
