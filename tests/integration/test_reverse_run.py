"""The `reverse` command, per PRD_reverse_flow section 2.

Schema, refusals and determinism only. No number produced here is asserted
against a threshold: reverse flow is a diagnostic (ADR-027) and its values are
whatever the fitted field says they are.
"""

import json
from pathlib import Path

import pytest

from conftest import STUB_DATASET
from fm_fewshot import sdk
from fm_fewshot.services.evaluation.reverse_run import (
    MissingRunError,
    NotAnFmRunError,
    find_run,
    run_reverse,
)
from fm_fewshot.services.flow.reverse import ReverseConfig
from fm_fewshot.shared.config import save_config
from fm_fewshot.shared.contracts import ExperimentConfig

HEAD_PARAMS = {
    "sample_steps": 2,
    "n_train_steps": 20,
    "batch_size": 8,
    "hidden_dims": (16, 16),
}
SMALL = ReverseConfig(
    reverse_steps=(2, 4), reverse_method=("euler", "midpoint"), n_samples=4,
    meet_times=(0.5,),
)


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


class TestSchema:
    def test_writes_reverse_json_beside_the_run(self, fitted_run) -> None:
        cfg, run_id, root = fitted_run
        out = run_reverse(
            cfg, reverse_cfg=SMALL, data_root=root, results_dir=root / "results"
        )
        assert out == root / "results" / run_id / "reverse.json"
        assert out.exists()

    def test_holds_the_three_metric_families(self, fitted_run) -> None:
        cfg, run_id, root = fitted_run
        out = run_reverse(cfg, reverse_cfg=SMALL, data_root=root, results_dir=root / "results")
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert set(payload["cycle"]) == {"2", "4"}
        assert set(payload["cycle"]["2"]) == {"euler", "midpoint"}
        assert set(payload["basin"]) >= {"alignment", "sigma", "n_samples"}
        assert set(payload["meet"]) == {"0.5"}

    def test_cycle_reports_median_and_iqr_not_mean_and_std(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        out = run_reverse(cfg, reverse_cfg=SMALL, data_root=root, results_dir=root / "results")
        entry = json.loads(out.read_text(encoding="utf-8"))["cycle"]["2"]["euler"]["rel_error"]
        assert set(entry) == {"median", "q1", "q3", "iqr", "n"}

    def test_cycle_splits_by_correct_against_incorrect(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        out = run_reverse(cfg, reverse_cfg=SMALL, data_root=root, results_dir=root / "results")
        entry = json.loads(out.read_text(encoding="utf-8"))["cycle"]["2"]["euler"]
        assert "rel_error_correct" in entry
        assert "rel_error_incorrect" in entry
        total = entry["rel_error_correct"]["n"] + entry["rel_error_incorrect"]["n"]
        assert total == entry["rel_error"]["n"]

    def test_basin_alignment_rows_are_normalized_or_empty(self, fitted_run) -> None:
        """Rows sum to 1, except a row whose cloud is anti-aligned with every
        class, which stays zero rather than being manufactured into a
        distribution. On the stub features that case is reachable."""
        cfg, _, root = fitted_run
        out = run_reverse(cfg, reverse_cfg=SMALL, data_root=root, results_dir=root / "results")
        alignment = json.loads(out.read_text(encoding="utf-8"))["basin"]["alignment"]
        assert len(alignment) == 3
        for row in alignment:
            assert len(row) == 3
            assert all(value >= 0.0 for value in row)
            assert sum(row) == pytest.approx(1.0, abs=1e-5) or sum(row) == 0.0

    def test_meet_holds_the_three_gap_families(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        out = run_reverse(cfg, reverse_cfg=SMALL, data_root=root, results_dir=root / "results")
        gaps = json.loads(out.read_text(encoding="utf-8"))["meet"]["0.5"]
        assert set(gaps) == {"gap_forward", "gap_backward", "gap_between"}

    def test_records_the_config_and_the_commit(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        out = run_reverse(cfg, reverse_cfg=SMALL, data_root=root, results_dir=root / "results")
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["config"]["head"] == "fm_standard"
        assert payload["git_commit"]
        assert payload["forward_steps"] == 2

    def test_volume_mode_is_written_only_when_asked(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        out = run_reverse(
            cfg,
            reverse_cfg=ReverseConfig(mode="volume", n_hutchinson=2),
            modes=("volume",),
            data_root=root,
            results_dir=root / "results",
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert set(payload["volume"]) == {"log_volume_change", "n_hutchinson"}
        assert "cycle" not in payload


class TestDeterminism:
    def test_two_invocations_write_identical_json(self, fitted_run) -> None:
        """The field is refitted from the stored config each time (ADR-025), so
        equality here covers the refit as well as the diagnostics."""
        cfg, _, root = fitted_run
        first = run_reverse(
            cfg, reverse_cfg=SMALL, data_root=root, results_dir=root / "results"
        ).read_text(encoding="utf-8")
        second = run_reverse(
            cfg, reverse_cfg=SMALL, data_root=root, results_dir=root / "results"
        ).read_text(encoding="utf-8")
        assert first == second


class TestRefusals:
    def test_a_config_with_no_stored_run_is_refused_by_name(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        with pytest.raises(MissingRunError, match="no stored run"):
            run_reverse(
                cfg.__class__(**{**cfg.__dict__, "k": 5}),
                reverse_cfg=SMALL,
                data_root=root,
                results_dir=root / "results",
            )

    def test_a_non_fm_head_is_refused(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        prototype = make_config(head="prototype", head_params={}, run_name="stub-proto")
        sdk.run_experiment(prototype, data_root=root, results_dir=root / "results")
        with pytest.raises(NotAnFmRunError, match="velocity field"):
            run_reverse(
                prototype, reverse_cfg=SMALL, data_root=root, results_dir=root / "results"
            )

    def test_an_unknown_mode_is_refused(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        with pytest.raises(ValueError, match="unknown reverse modes"):
            run_reverse(
                cfg, modes=("sideways",), data_root=root, results_dir=root / "results"
            )

    def test_a_missing_run_id_is_refused(self, fitted_run) -> None:
        cfg, _, root = fitted_run
        with pytest.raises(MissingRunError, match="no run directory"):
            run_reverse(
                cfg,
                reverse_cfg=SMALL,
                data_root=root,
                results_dir=root / "results",
                run_id="not-a-run",
            )


class TestFindRun:
    def test_matches_on_the_identifying_fields_not_the_name(self, fitted_run) -> None:
        cfg, run_id, root = fitted_run
        renamed = make_config(run_name="a-different-name")
        assert find_run(root / "results", renamed) == run_id

    def test_ignores_directories_without_a_config(self, fitted_run) -> None:
        cfg, run_id, root = fitted_run
        (root / "results" / "stray").mkdir()
        assert find_run(root / "results", cfg) == run_id

    def test_takes_the_latest_of_several_matches(self, fitted_run) -> None:
        cfg, run_id, root = fitted_run
        later = root / "results" / "zzz-later"
        later.mkdir()
        save_config(cfg, later / "config.yaml")
        assert find_run(root / "results", cfg) == "zzz-later"


class TestCli:
    def test_reverse_subcommand_writes_the_file(self, fitted_run, capsys) -> None:
        from fm_fewshot.__main__ import main

        cfg, run_id, root = fitted_run
        config_path = root / "cfg.yaml"
        save_config(cfg, config_path)
        main(
            [
                "reverse",
                "--config", str(config_path),
                "--data-root", str(root),
                "--results-dir", str(root / "results"),
                "--mode", "cycle",
                "--reverse-steps", "2",
                "--reverse-method", "euler",
            ]
        )
        printed = capsys.readouterr().out.strip()
        assert printed.endswith("reverse.json")
        assert Path(printed).exists()

    def test_a_bad_reverse_request_exits_with_the_reason(self, fitted_run) -> None:
        from fm_fewshot.__main__ import main

        cfg, _, root = fitted_run
        config_path = root / "cfg.yaml"
        save_config(cfg, config_path)
        with pytest.raises(SystemExit, match="n_hutchinson"):
            main(
                [
                    "reverse",
                    "--config", str(config_path),
                    "--data-root", str(root),
                    "--results-dir", str(root / "results"),
                    "--mode", "volume",
                    "--n-hutchinson", "0",
                ]
            )
