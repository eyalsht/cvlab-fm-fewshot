"""Run-loop tests per PRD_evaluation_protocol section 5, against a stub head."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from conftest import STUB_DATASET
from fm_fewshot import sdk
from fm_fewshot.services.evaluation import loop as loop_module
from fm_fewshot.services.heads.base import FewShotHead, HeadContext, register
from fm_fewshot.shared.contracts import ExperimentConfig

N_CLASSES = 3


@register("stub_head")
class StubHead(FewShotHead):
    """Predicts by nearest class mean; records what it was fitted on."""

    last_train_rows: int = 0

    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes
        self._means = None

    @classmethod
    def from_context(cls, cfg, n_classes, context: HeadContext):
        return cls(n_classes=n_classes)

    def fit(self, train_x, train_y, val_x, val_y) -> None:
        type(self).last_train_rows = int(train_x.shape[0])
        means = torch.zeros(self._n_classes, train_x.shape[1])
        for j in range(self._n_classes):
            rows = train_x[train_y == j]
            if rows.shape[0]:
                means[j] = rows.mean(dim=0)
        self._means = means

    def predict(self, query_x):
        return (query_x @ self._means.T).float()


@pytest.fixture
def built_caches(tmp_path: Path, stub_dataset: str) -> Path:
    for split in ("train", "val", "test"):
        sdk.build_features(
            STUB_DATASET, split, encoder="stub", data_root=tmp_path, allow_heavy_on_cpu=True
        )
    return tmp_path


def make_config(**overrides) -> ExperimentConfig:
    fields = {
        "run_name": "stub-run",
        "dataset": STUB_DATASET,
        "encoder": "stub",
        "head": "stub_head",
        "head_params": {},
        "k": 2,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


class TestResultsTriple:
    def test_writes_config_and_summary(self, built_caches: Path) -> None:
        summary = sdk.run_experiment(
            make_config(), data_root=built_caches, results_dir=built_caches / "results"
        )
        run_dir = built_caches / "results" / summary.run_id
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "preds.npy").exists()

    def test_summary_records_the_git_commit(self, built_caches: Path) -> None:
        summary = sdk.run_experiment(
            make_config(), data_root=built_caches, results_dir=built_caches / "results"
        )
        assert summary.git_commit
        assert summary.git_commit != "unknown"

    def test_closed_form_head_writes_no_epochs_csv(self, built_caches: Path) -> None:
        summary = sdk.run_experiment(
            make_config(), data_root=built_caches, results_dir=built_caches / "results"
        )
        run_dir = built_caches / "results" / summary.run_id
        assert summary.epochs == []
        assert not (run_dir / "epochs.csv").exists()


class TestTestSplitDiscipline:
    def test_head_is_fitted_only_on_the_subset(self, built_caches: Path) -> None:
        """K=2 over 3 classes is 6 rows, never the 30-row train split."""
        sdk.run_experiment(
            make_config(k=2), data_root=built_caches, results_dir=built_caches / "results"
        )
        assert StubHead.last_train_rows == 2 * N_CLASSES

    def test_full_setting_fits_on_the_whole_train_split(self, built_caches: Path) -> None:
        sdk.run_experiment(
            make_config(k=None), data_root=built_caches, results_dir=built_caches / "results"
        )
        assert StubHead.last_train_rows == 30

    def test_every_test_row_is_scored(self, built_caches: Path) -> None:
        summary = sdk.run_experiment(
            make_config(), data_root=built_caches, results_dir=built_caches / "results"
        )
        assert summary.n_test == 30

    def test_short_evaluation_aborts_naming_both_counts(
        self, built_caches: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A partial evaluation must never be reported as a full one."""
        original = loop_module.read_features

        def truncating(dataset, split, encoder_name, **kwargs):
            features, labels, meta = original(dataset, split, encoder_name, **kwargs)
            if split == "test":
                return features[:5], labels[:5], meta
            return features, labels, meta

        monkeypatch.setattr(loop_module, "read_features", truncating)
        with pytest.raises(ValueError, match="30"):
            sdk.run_experiment(
                make_config(), data_root=built_caches, results_dir=built_caches / "results"
            )


class TestSubsetSharing:
    def test_two_heads_at_one_seed_see_identical_subsets(self, built_caches: Path) -> None:
        a = sdk.run_experiment(
            make_config(head="stub_head"),
            data_root=built_caches,
            results_dir=built_caches / "results",
        )
        b = sdk.run_experiment(
            make_config(head="prototype"),
            data_root=built_caches,
            results_dir=built_caches / "results",
        )
        assert a.subset_idx == b.subset_idx

    def test_different_subset_seeds_differ(self, built_caches: Path) -> None:
        a = sdk.run_experiment(
            make_config(subset_seed=0),
            data_root=built_caches,
            results_dir=built_caches / "results",
        )
        b = sdk.run_experiment(
            make_config(subset_seed=1),
            data_root=built_caches,
            results_dir=built_caches / "results",
        )
        assert a.subset_idx != b.subset_idx


class TestDeterminism:
    def test_same_config_twice_is_bit_identical(self, built_caches: Path) -> None:
        """NFR2 on one device."""
        results = built_caches / "results"
        first = sdk.run_experiment(make_config(), data_root=built_caches, results_dir=results)
        second = sdk.run_experiment(make_config(), data_root=built_caches, results_dir=results)

        def payload(run_id: str) -> dict:
            raw = json.loads((results / run_id / "summary.json").read_text())
            for volatile in ("run_id", "fit_seconds", "predict_seconds", "wall_seconds"):
                raw.pop(volatile, None)
            return raw

        assert payload(first.run_id) == payload(second.run_id)
        assert first.test_top1 == second.test_top1


class TestAtomicity:
    def test_a_failing_run_leaves_no_directory(
        self, built_caches: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = built_caches / "results"

        def explode(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(loop_module, "top1_accuracy", explode)
        with pytest.raises(RuntimeError, match="boom"):
            sdk.run_experiment(make_config(), data_root=built_caches, results_dir=results)
        assert not results.exists() or list(results.iterdir()) == []


class TestValidation:
    def test_unknown_head_raises_before_any_work(self, built_caches: Path) -> None:
        with pytest.raises(ValueError, match="unknown head"):
            sdk.run_experiment(
                make_config(head="no_such_head"),
                data_root=built_caches,
                results_dir=built_caches / "results",
            )

    def test_missing_cache_says_build_first(self, tmp_path: Path, stub_dataset: str) -> None:
        with pytest.raises(FileNotFoundError, match="build_features"):
            sdk.run_experiment(
                make_config(), data_root=tmp_path, results_dir=tmp_path / "results"
            )


class TestConfigRoundTrip:
    def test_written_config_reloads_to_the_same_object(self, built_caches: Path) -> None:
        from fm_fewshot.shared.config import load_config

        cfg = make_config(k=None, subset_seed=2, init_seed=1)
        summary = sdk.run_experiment(
            cfg, data_root=built_caches, results_dir=built_caches / "results"
        )
        written = load_config(built_caches / "results" / summary.run_id / "config.yaml")
        assert written == dataclasses.replace(cfg)
