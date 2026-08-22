"""Report tests per PRD_evaluation_protocol section 5."""

import dataclasses
import json
from pathlib import Path

import pytest

from fm_fewshot.services.evaluation.report import (
    IncompleteCellError,
    expected_runs,
    load_cells,
    write_table,
)
from fm_fewshot.shared.contracts import ExperimentConfig, RunSummary


def write_run(results: Path, run_id: str, accuracy: float, **cfg_overrides) -> None:
    fields = {
        "run_name": "r",
        "dataset": "dtd",
        "encoder": "resnet18",
        "head": "prototype",
        "head_params": {},
        "k": 5,
    }
    fields.update(cfg_overrides)
    cfg = ExperimentConfig(**fields)
    summary = RunSummary(
        run_id=run_id,
        config=cfg,
        git_commit="deadbeef",
        test_top1=accuracy,
        n_test=1880,
        n_train=235,
        subset_idx=(0, 1, 2),
        best_epoch=None,
        epochs=[],
        fit_seconds=0.1,
        predict_seconds=0.1,
        wall_seconds=0.2,
    )
    payload = dataclasses.asdict(summary)
    payload.pop("epochs")
    run_dir = results / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestExpectedRuns:
    def test_protocol_run_counts(self) -> None:
        """Straight from the write-up's reporting rules."""
        assert expected_runs("linear_probe", 5) == 3
        assert expected_runs("linear_probe", 10) == 3
        assert expected_runs("linear_probe", None) == 3
        assert expected_runs("prototype", 5) == 3
        assert expected_runs("prototype", 10) == 3
        # Full-data prototype has no subset draw and no initialization, so
        # three seeds would give three identical numbers.
        assert expected_runs("prototype", None) == 1


class TestAggregation:
    def test_groups_runs_into_one_cell(self, tmp_path: Path) -> None:
        for i, acc in enumerate([0.40, 0.50, 0.60]):
            write_run(tmp_path, f"run{i}", acc, subset_seed=i)
        cells = load_cells(tmp_path)
        assert len(cells) == 1
        assert cells[0].mean == pytest.approx(0.5)
        assert cells[0].std == pytest.approx(0.1)
        assert cells[0].n_runs == 3

    def test_separates_cells_by_every_key(self, tmp_path: Path) -> None:
        write_run(tmp_path, "a", 0.1, k=5)
        write_run(tmp_path, "b", 0.2, k=10)
        write_run(tmp_path, "c", 0.3, encoder="dinov2_vits14")
        write_run(tmp_path, "d", 0.4, head="linear_probe")
        write_run(tmp_path, "e", 0.5, dataset="fgvc_aircraft")
        # Each of the five differs in one key, so no two share a cell; the
        # incomplete-cell check is disabled for this structural assertion.
        cells = load_cells(tmp_path, require_complete=False)
        assert len(cells) == 5

    def test_single_run_prototype_full_cell_is_complete(self, tmp_path: Path) -> None:
        write_run(tmp_path, "only", 0.42, k=None, head="prototype")
        cells = load_cells(tmp_path)
        assert cells[0].n_runs == 1
        assert cells[0].std == 0.0


class TestIncompleteCells:
    def test_two_runs_where_three_are_required_raises(self, tmp_path: Path) -> None:
        write_run(tmp_path, "a", 0.4, subset_seed=0)
        write_run(tmp_path, "b", 0.5, subset_seed=1)
        with pytest.raises(IncompleteCellError, match="2 of 3"):
            load_cells(tmp_path)

    def test_the_message_names_the_cell(self, tmp_path: Path) -> None:
        write_run(tmp_path, "a", 0.4, subset_seed=0)
        with pytest.raises(IncompleteCellError, match="dtd"):
            load_cells(tmp_path)


class TestSkipping:
    def test_directories_without_a_summary_are_skipped(self, tmp_path: Path) -> None:
        write_run(tmp_path, "only", 0.42, k=None, head="prototype")
        (tmp_path / "half_written").mkdir()
        assert len(load_cells(tmp_path)) == 1

    def test_empty_results_dir_gives_no_cells(self, tmp_path: Path) -> None:
        assert load_cells(tmp_path) == []


class TestTable:
    def test_regeneration_is_idempotent(self, tmp_path: Path) -> None:
        for i, acc in enumerate([0.40, 0.50, 0.60]):
            write_run(tmp_path, f"run{i}", acc, subset_seed=i)
        table = tmp_path / "TABLE.md"
        write_table(tmp_path, table)
        first = table.read_bytes()
        write_table(tmp_path, table)
        assert table.read_bytes() == first

    def test_table_reports_mean_and_std(self, tmp_path: Path) -> None:
        for i, acc in enumerate([0.40, 0.50, 0.60]):
            write_run(tmp_path, f"run{i}", acc, subset_seed=i)
        write_table(tmp_path, tmp_path / "TABLE.md")
        text = (tmp_path / "TABLE.md").read_text()
        assert "0.500" in text
        assert "0.100" in text
        assert "dtd" in text and "resnet18" in text and "prototype" in text

    def test_single_run_cell_is_flagged_rather_than_showing_a_std(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path, "only", 0.42, k=None, head="prototype")
        write_table(tmp_path, tmp_path / "TABLE.md")
        text = (tmp_path / "TABLE.md").read_text()
        assert "0.420" in text
        assert "single run" in text


def write_cell(results: Path, head: str, accuracies: list[float], **cfg_overrides) -> None:
    """Three runs at three subset seeds, the protocol's usual cell (FR16)."""
    for i, accuracy in enumerate(accuracies):
        write_run(results, f"{head}-{i}", accuracy, head=head, subset_seed=i, **cfg_overrides)


class TestDeltaAccuracy:
    """dAcc = Acc_FM - Acc_baseline against the prototype row of the same
    (dataset, encoder, K) cell (FR16, write-up deliverables)."""

    def test_dacc_is_computed_against_the_prototype_baseline(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_cell(tmp_path, "fm_standard", [0.53, 0.55, 0.57])
        write_table(tmp_path, tmp_path / "TABLE.md")
        text = (tmp_path / "TABLE.md").read_text()
        assert "+0.150" in text

    def test_dacc_is_blank_on_the_baseline_row_itself(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_cell(tmp_path, "fm_standard", [0.53, 0.55, 0.57])
        write_table(tmp_path, tmp_path / "TABLE.md")
        rows = (tmp_path / "TABLE.md").read_text().splitlines()
        proto_row = next(r for r in rows if "prototype" in r)
        cells = [c.strip() for c in proto_row.strip("|").split("|")]
        assert cells[-1] == ""

    def test_dacc_sign_is_negative_when_fm_underperforms(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "prototype", [0.58, 0.60, 0.62])
        write_cell(tmp_path, "fm_standard", [0.43, 0.45, 0.47])
        write_table(tmp_path, tmp_path / "TABLE.md")
        text = (tmp_path / "TABLE.md").read_text()
        assert "-0.150" in text

    def test_report_stays_idempotent_with_dacc(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_cell(tmp_path, "fm_standard", [0.53, 0.55, 0.57])
        table = tmp_path / "TABLE.md"
        write_table(tmp_path, table)
        first = table.read_bytes()
        write_table(tmp_path, table)
        assert table.read_bytes() == first
