"""The `rowspace` command: ADR-036's row(W)/null(W) split beside a stored run.

Schema, refusals and determinism only, the same footing as `diagnose`
(`test_scale_diagnostics_run.py`). No fraction is asserted against a
threshold except at the one point the write-up itself pins down: before any
training, the field is the identity (ADR-030), so the whole displacement is
zero and both fractions must read as zero rather than NaN, which is the
schema-level echo of the closed-form unit test in
`tests/unit/test_rowspace_diagnostics.py`.
"""

import json
from pathlib import Path

import pytest

from conftest import STUB_DATASET
from fm_fewshot import sdk
from fm_fewshot.services.evaluation.rowspace_diagnostics import (
    MissingRunError,
    NotAnFmRunError,
    run_rowspace_diagnostics,
)
from fm_fewshot.shared.contracts import ExperimentConfig

STAGE3_PARAMS: dict[str, object] = {
    "sample_steps": 2,
    "n_train_steps": 4,
    "hidden_dims": [8, 8],
    "batch_size": 4,
    "lr": 1e-3,
    "eval_every": 0,
}


def config(head: str, **overrides: object) -> ExperimentConfig:
    fields: dict[str, object] = {
        "run_name": f"rowspace-{head}",
        "dataset": STUB_DATASET,
        "encoder": "stub",
        "head": head,
        "head_params": dict(STAGE3_PARAMS),
        "k": 2,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


@pytest.fixture
def built_caches(tmp_path: Path, stub_dataset: str) -> Path:
    for split in ("train", "val", "test"):
        sdk.build_features(
            STUB_DATASET, split, encoder="stub", data_root=tmp_path, allow_heavy_on_cpu=True
        )
    return tmp_path


@pytest.fixture
def fitted_run(built_caches: Path, tmp_path: Path):
    cfg = config("fm_prelinear_ce")
    summary = sdk.run_experiment(
        cfg, data_root=built_caches, results_dir=tmp_path / "results"
    )
    return cfg, summary.run_id, built_caches


def rowspace(cfg: ExperimentConfig, root: Path, **kwargs) -> dict:
    out = run_rowspace_diagnostics(cfg, data_root=root, results_dir=root / "results", **kwargs)
    return json.loads(out.read_text(encoding="utf-8"))


class TestSchema:
    def test_writes_rowspace_json_beside_the_run(self, fitted_run) -> None:
        cfg, run_id, root = fitted_run
        out = run_rowspace_diagnostics(cfg, data_root=root, results_dir=root / "results")
        assert out == root / "results" / run_id / "rowspace.json"

    def test_holds_exactly_the_fixed_top_level_schema(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        payload = rowspace(cfg, root)
        assert set(payload) == {
            "run_id",
            "head",
            "sample_steps",
            "dim",
            "rank_w",
            "n_points",
            "max_points",
            "summary",
            "points",
        }

    def test_summary_holds_the_six_named_statistics(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        payload = rowspace(cfg, root)
        assert set(payload["summary"]) == {
            "row_fraction_mean",
            "row_fraction_median",
            "null_fraction_mean",
            "null_fraction_median",
            "total_norm_mean",
            "total_norm_median",
        }

    def test_points_hold_the_five_named_series_at_n_points_length(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        payload = rowspace(cfg, root)
        expected = {"total_norm", "row_norm", "null_norm", "margin_before", "margin_after"}
        assert set(payload["points"]) == expected
        for series in expected:
            assert len(payload["points"][series]) == payload["n_points"]

    def test_records_the_run_the_head_and_t(self, fitted_run) -> None:
        cfg, run_id, root = fitted_run
        payload = rowspace(cfg, root)
        assert payload["run_id"] == run_id
        assert payload["head"] == "fm_prelinear_ce"
        assert payload["sample_steps"] == 2

    def test_dim_is_the_stub_encoders_width_and_rank_w_is_at_most_n_classes(
        self, fitted_run
    ) -> None:
        cfg, _, root = fitted_run
        payload = rowspace(cfg, root)
        assert payload["dim"] == 8  # StubEncoder.dim
        assert 0 < payload["rank_w"] <= 3  # STUB_CLASSES has 3 classes

    def test_max_points_defaults_and_n_points_never_exceeds_it(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        payload = rowspace(cfg, root)
        assert payload["max_points"] == 4096
        assert payload["n_points"] <= payload["max_points"]
        assert payload["n_points"] > 0

    def test_max_points_below_the_split_size_subsamples(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        payload = rowspace(cfg, root, max_points=5)
        assert payload["n_points"] == 5
        assert payload["max_points"] == 5


class TestTheUntrainedFieldIsTheIdentity:
    """ADR-030: zero_output_init leaves zhat == z exactly before any training."""

    def test_zero_training_steps_gives_zero_displacement_everywhere(
        self, built_caches: Path, tmp_path: Path
    ) -> None:
        cfg = config("fm_prelinear_guided", head_params=dict(STAGE3_PARAMS, n_train_steps=0))
        summary = sdk.run_experiment(
            cfg, data_root=built_caches, results_dir=built_caches / "results"
        )
        payload = json.loads(
            run_rowspace_diagnostics(
                cfg, data_root=built_caches, results_dir=built_caches / "results"
            ).read_text(encoding="utf-8")
        )
        assert payload["run_id"] == summary.run_id
        assert all(v == 0.0 for v in payload["points"]["total_norm"])
        assert all(v == 0.0 for v in payload["points"]["row_norm"])
        assert all(v == 0.0 for v in payload["points"]["null_norm"])

    def test_zero_training_steps_gives_zero_fractions_not_nan(
        self, built_caches: Path, tmp_path: Path
    ) -> None:
        cfg = config("fm_prelinear_ce", head_params=dict(STAGE3_PARAMS, n_train_steps=0))
        sdk.run_experiment(cfg, data_root=built_caches, results_dir=built_caches / "results")
        payload = json.loads(
            run_rowspace_diagnostics(
                cfg, data_root=built_caches, results_dir=built_caches / "results"
            ).read_text(encoding="utf-8")
        )
        for key in ("row_fraction_mean", "row_fraction_median", "null_fraction_mean",
                    "null_fraction_median"):
            assert payload["summary"][key] == 0.0

    def test_zero_training_steps_gives_equal_margins_before_and_after(
        self, built_caches: Path, tmp_path: Path
    ) -> None:
        # zhat == z exactly, so the probe reads the same features either way
        # and the two margins must agree row for row.
        cfg = config("fm_prelinear_ce", head_params=dict(STAGE3_PARAMS, n_train_steps=0))
        sdk.run_experiment(cfg, data_root=built_caches, results_dir=built_caches / "results")
        payload = json.loads(
            run_rowspace_diagnostics(
                cfg, data_root=built_caches, results_dir=built_caches / "results"
            ).read_text(encoding="utf-8")
        )
        assert payload["points"]["margin_before"] == payload["points"]["margin_after"]


class TestDeterminism:
    def test_two_invocations_write_identical_json(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        first = json.dumps(rowspace(cfg, root), sort_keys=True)
        second = json.dumps(rowspace(cfg, root), sort_keys=True)
        assert first == second


class TestRefusals:
    def test_a_non_stage3_head_is_refused(self, fitted_run) -> None:
        _, _, root = fitted_run
        probe = ExperimentConfig(
            run_name="rowspace-probe",
            dataset=STUB_DATASET,
            encoder="stub",
            head="linear_probe",
            head_params={},
            k=2,
        )
        sdk.run_experiment(probe, data_root=root, results_dir=root / "results")
        with pytest.raises(NotAnFmRunError, match="frozen linear probe"):
            run_rowspace_diagnostics(probe, data_root=root, results_dir=root / "results")

    def test_a_stage2_fm_head_is_also_refused(self, fitted_run) -> None:
        """The row(W) constraint is Stage 3's; Stage 2 has no frozen probe at all."""
        _, _, root = fitted_run
        fm_standard = ExperimentConfig(
            run_name="rowspace-fm-standard",
            dataset=STUB_DATASET,
            encoder="stub",
            head="fm_standard",
            head_params={"sample_steps": 2, "n_train_steps": 4, "hidden_dims": [8, 8]},
            k=2,
        )
        sdk.run_experiment(fm_standard, data_root=root, results_dir=root / "results")
        with pytest.raises(NotAnFmRunError, match="frozen linear probe"):
            run_rowspace_diagnostics(fm_standard, data_root=root, results_dir=root / "results")

    def test_a_config_with_no_stored_run_is_refused(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        missing = config("fm_prelinear_ce", k=5)
        with pytest.raises(MissingRunError, match="no stored run"):
            run_rowspace_diagnostics(missing, data_root=root, results_dir=root / "results")


class TestStaysOutOfTheTable:
    def test_the_report_ignores_the_rowspace_file(self, fitted_run) -> None:
        from fm_fewshot.services.evaluation.report import load_cells

        cfg, _, root = fitted_run
        before = load_cells(root / "results", require_complete=False)
        rowspace(cfg, root)
        assert load_cells(root / "results", require_complete=False) == before
