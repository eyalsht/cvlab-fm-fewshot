"""Stage 3 through the real run loop, on the stub encoder (TODO 10.6).

The unit tests assert the head's claims against tensors it was handed. These
assert them against what a finished run actually writes, which is what the
gate reads and what a machine that did not produce the run has to trust.

Two claims, end to end:

- a Stage 3 run with `n_train_steps = 0` reports exactly the linear probe's
  accuracy, because the field starts at identity and the probe inside the head
  is the Stage 1 probe refitted at the same seed (ADR-030, ADR-031);
- `subset_check` passes over a store holding the probe run and both Stage 3
  runs, on both of its guards: identical rows, and one frozen classifier.

No dataset is needed. The stub encoder writes an 8-dimensional cache, so the
whole path from `build_features` to `summary.json` runs anywhere.
"""

import json
from pathlib import Path

import pytest

from conftest import STUB_DATASET
from fm_fewshot import sdk
from fm_fewshot.services.evaluation.subset_check import (
    find_classifier_disagreements,
    find_subset_disagreements,
)
from fm_fewshot.services.evaluation.subset_check import main as check_subsets
from fm_fewshot.shared.contracts import ExperimentConfig

STAGE3_PARAMS: dict[str, object] = {
    "sample_steps": 4,
    "n_train_steps": 4,
    "hidden_dims": [8, 8],
    "batch_size": 4,
    "lr": 1e-3,
    "eval_every": 0,
}


@pytest.fixture
def built_caches(tmp_path: Path, stub_dataset: str) -> Path:
    for split in ("train", "val", "test"):
        sdk.build_features(
            STUB_DATASET, split, encoder="stub", data_root=tmp_path, allow_heavy_on_cpu=True
        )
    return tmp_path


def config(head: str, **overrides: object) -> ExperimentConfig:
    fields: dict[str, object] = {
        "run_name": f"stage3-{head}",
        "dataset": STUB_DATASET,
        "encoder": "stub",
        "head": head,
        "head_params": {},
        "k": 2,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def summary_of(results_dir: Path, run_id: str) -> dict:
    return json.loads((results_dir / run_id / "summary.json").read_text(encoding="utf-8"))


class TestTheUntrainedRunIsTheProbeRun:
    def test_zero_training_steps_reports_the_probes_accuracy(
        self, built_caches: Path, tmp_path: Path
    ) -> None:
        results = tmp_path / "results"
        probe = sdk.run_experiment(
            config("linear_probe"), data_root=built_caches, results_dir=results
        )
        stage3 = sdk.run_experiment(
            config("fm_prelinear_ce", head_params=dict(STAGE3_PARAMS, n_train_steps=0)),
            data_root=built_caches,
            results_dir=results,
        )
        assert stage3.test_top1 == probe.test_top1

    def test_both_strategies_report_it(self, built_caches: Path, tmp_path: Path) -> None:
        results = tmp_path / "results"
        probe = sdk.run_experiment(
            config("linear_probe"), data_root=built_caches, results_dir=results
        )
        for head in ("fm_prelinear_ce", "fm_prelinear_guided"):
            stage3 = sdk.run_experiment(
                config(head, head_params=dict(STAGE3_PARAMS, n_train_steps=0)),
                data_root=built_caches,
                results_dir=results,
            )
            assert stage3.test_top1 == probe.test_top1


class TestTheRunStoreCarriesTheClassifier:
    @pytest.fixture
    def store(self, built_caches: Path, tmp_path: Path) -> Path:
        results = tmp_path / "results"
        sdk.run_experiment(config("linear_probe"), data_root=built_caches, results_dir=results)
        for head in ("fm_prelinear_ce", "fm_prelinear_guided"):
            sdk.run_experiment(
                config(head, head_params=STAGE3_PARAMS),
                data_root=built_caches,
                results_dir=results,
            )
        return results

    def test_every_run_records_one_digest(self, store: Path) -> None:
        digests = {
            summary_of(store, run.name)["classifier_digest"] for run in store.iterdir()
        }
        assert len(digests) == 1
        assert None not in digests

    def test_a_head_that_fits_no_classifier_records_none(
        self, built_caches: Path, tmp_path: Path
    ) -> None:
        results = tmp_path / "results"
        summary = sdk.run_experiment(
            config("prototype"), data_root=built_caches, results_dir=results
        )
        assert summary_of(results, summary.run_id)["classifier_digest"] is None

    def test_the_gate_passes_on_both_guards(self, store: Path) -> None:
        assert find_subset_disagreements(store) == []
        assert find_classifier_disagreements(store) == []
        assert check_subsets([str(store)]) == 0

    def test_the_gate_catches_a_run_that_used_another_classifier(
        self, store: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The negative control, forged in the store the way a real fault would show.

        A Stage 3 head that refitted the probe on other terms would write a
        different digest at the same setting, and the gate has to name it.
        """
        run = next(d for d in sorted(store.iterdir()) if "prelinear_ce" in d.name)
        payload = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        payload["classifier_digest"] = "0" * 64
        (run / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

        assert check_subsets([str(store)]) == 1
        assert "classifier mismatch" in capsys.readouterr().out

    def test_the_stage_3_runs_fitted_the_probes_rows(self, store: Path) -> None:
        rows = {
            tuple(summary_of(store, run.name)["subset_idx"]) for run in store.iterdir()
        }
        assert len(rows) == 1
