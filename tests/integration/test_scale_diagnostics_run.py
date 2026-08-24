"""The `diagnose` command: ADR-018's four numbers beside a stored run.

Schema, refusals and determinism only. No value is asserted against a
threshold: like reverse flow these are diagnostics (ADR-027 for the labels they
read, ADR-018 for what they are for), and their values are whatever the fitted
field and the raw features say they are.

The one structural claim worth a test is that nothing here reaches the report:
`diagnostics.json` lives inside the run directory and `TABLE.md` is regenerated
from `summary.json` alone.
"""

import json
from pathlib import Path

import pytest

from conftest import STUB_DATASET
from fm_fewshot import sdk
from fm_fewshot.services.evaluation.reverse_run import MissingRunError, NotAnFmRunError
from fm_fewshot.services.evaluation.scale_diagnostics import run_diagnostics
from fm_fewshot.shared.contracts import ExperimentConfig

HEAD_PARAMS = {
    "sample_steps": 2,
    "n_train_steps": 20,
    "batch_size": 8,
    "hidden_dims": (16, 16),
}


def make_config(**overrides) -> ExperimentConfig:
    fields = {
        "run_name": "stub-fm",
        "dataset": STUB_DATASET,
        "encoder": "stub",
        "head": "fm_standard",
        "head_params": dict(HEAD_PARAMS),
        "k": 2,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


@pytest.fixture
def fitted_run(tmp_path: Path, stub_dataset: str):
    for split in ("train", "val", "test"):
        sdk.build_features(
            STUB_DATASET, split, encoder="stub", data_root=tmp_path, allow_heavy_on_cpu=True
        )
    cfg = make_config()
    summary = sdk.run_experiment(cfg, data_root=tmp_path, results_dir=tmp_path / "results")
    return cfg, summary.run_id, tmp_path


def diagnose(cfg, root: Path, **kwargs) -> dict:
    out = run_diagnostics(cfg, data_root=root, results_dir=root / "results", **kwargs)
    return json.loads(out.read_text(encoding="utf-8"))


class TestSchema:
    def test_writes_diagnostics_json_beside_the_run(self, fitted_run) -> None:
        cfg, run_id, root = fitted_run
        out = run_diagnostics(cfg, data_root=root, results_dir=root / "results")
        assert out == root / "results" / run_id / "diagnostics.json"

    def test_holds_the_four_adr_018_families(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        payload = diagnose(cfg, root)
        assert set(payload) >= {
            "feature_norm",
            "target_distance",
            "target_cosine",
            "pairwise_cosine",
        }

    def test_norms_and_distances_cover_train_and_test(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        payload = diagnose(cfg, root)
        for family in ("feature_norm", "target_distance", "target_cosine"):
            assert set(payload[family]) == {"train", "test"}

    def test_every_number_is_median_and_iqr(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        payload = diagnose(cfg, root)
        assert set(payload["feature_norm"]["test"]) == {"median", "q1", "q3", "iqr", "n"}
        assert set(payload["pairwise_cosine"]["test_transported"]) == {
            "median",
            "q1",
            "q3",
            "iqr",
            "n",
        }

    def test_pairwise_cosine_carries_the_raw_reference(self, fitted_run) -> None:
        """Transported cosine alone cannot say whether the cloud collapsed;
        the raw split measured the same way is what it is read against."""
        cfg, _, root = fitted_run
        payload = diagnose(cfg, root)
        assert set(payload["pairwise_cosine"]) >= {"test_raw", "test_transported"}

    def test_records_the_run_the_config_and_the_commit(self, fitted_run) -> None:
        cfg, run_id, root = fitted_run
        payload = diagnose(cfg, root)
        assert payload["run_id"] == run_id
        assert payload["config"]["head"] == "fm_standard"
        assert payload["git_commit"]
        assert payload["forward_steps"] == 2

    def test_prototype_norm_records_the_scale_the_features_are_read_against(
        self, fitted_run
    ) -> None:
        """ADR-018 is a statement about two scales, so the note needs both:
        the prototypes are unit norm by construction and this records it."""
        cfg, _, root = fitted_run
        payload = diagnose(cfg, root)
        assert payload["prototype_norm"]["median"] == pytest.approx(1.0, abs=1e-5)


class TestDeterminism:
    def test_two_invocations_write_identical_json(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        first = json.dumps(diagnose(cfg, root), sort_keys=True)
        second = json.dumps(diagnose(cfg, root), sort_keys=True)
        assert first == second


class TestRefusals:
    def test_a_non_fm_head_is_refused(self, fitted_run) -> None:
        _, _, root = fitted_run
        prototype = make_config(head="prototype", head_params={}, run_name="stub-proto")
        sdk.run_experiment(prototype, data_root=root, results_dir=root / "results")
        with pytest.raises(NotAnFmRunError, match="velocity field"):
            run_diagnostics(prototype, data_root=root, results_dir=root / "results")

    def test_a_config_with_no_stored_run_is_refused(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        with pytest.raises(MissingRunError, match="no stored run"):
            run_diagnostics(
                make_config(k=5), data_root=root, results_dir=root / "results"
            )


class TestStaysOutOfTheTable:
    def test_the_report_ignores_the_diagnostics_file(self, fitted_run) -> None:
        """ADR-018's ablation and these numbers live in the phase note. The
        guard is structural: report reads summary.json and nothing else."""
        from fm_fewshot.services.evaluation.report import load_cells

        cfg, _, root = fitted_run
        before = load_cells(root / "results", require_complete=False)
        diagnose(cfg, root)
        assert load_cells(root / "results", require_complete=False) == before
