"""Report tests per PRD_evaluation_protocol section 5."""

import dataclasses
import inspect
import json
from pathlib import Path

import pytest

from fm_fewshot.services.evaluation import report as report_module
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
        proto_row = next(r for r in rows if "| prototype |" in r)
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


class TestStage3DeltaAccuracy:
    """ADR-035: a row's baseline is a property of its head, not of the table.

    Stage 2 reports against the prototype row and Stage 3 against the linear
    probe, because the write-up says so for each. `report` must not learn that
    as a branch on which stage a head belongs to; it asks the head, through the
    registry, what it is measured against.
    """

    def test_a_stage_3_row_is_measured_against_the_linear_probe(
        self, tmp_path: Path
    ) -> None:
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_cell(tmp_path, "linear_probe", [0.48, 0.50, 0.52])
        write_cell(tmp_path, "fm_prelinear_ce", [0.53, 0.55, 0.57])
        rows = table_rows(tmp_path)
        assert rows["fm_prelinear_ce"] == "+0.050"

    def test_both_stage_3_strategies_use_it(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_cell(tmp_path, "linear_probe", [0.48, 0.50, 0.52])
        write_cell(tmp_path, "fm_prelinear_guided", [0.43, 0.45, 0.47])
        assert table_rows(tmp_path)["fm_prelinear_guided"] == "-0.050"

    def test_a_stage_2_row_still_uses_the_prototype(self, tmp_path: Path) -> None:
        """The row Stage 3 must not disturb, in the same table as a Stage 3 row."""
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_cell(tmp_path, "linear_probe", [0.48, 0.50, 0.52])
        write_cell(tmp_path, "fm_standard", [0.53, 0.55, 0.57])
        write_cell(tmp_path, "fm_prelinear_ce", [0.53, 0.55, 0.57])
        rows = table_rows(tmp_path)
        assert rows["fm_standard"] == "+0.150"
        assert rows["fm_prelinear_ce"] == "+0.050"

    def test_the_probe_row_keeps_its_own_delta_against_the_prototype(
        self, tmp_path: Path
    ) -> None:
        """Stage 1's table already reports this and the stored TABLE.md carries
        it, so making the probe a baseline must not blank its own row."""
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_cell(tmp_path, "linear_probe", [0.48, 0.50, 0.52])
        assert table_rows(tmp_path)["linear_probe"] == "+0.100"

    def test_a_variant_resolves_through_its_head_key(self, tmp_path: Path) -> None:
        """The table labels a cell `head@variant`; the baseline is the head's."""
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_cell(tmp_path, "linear_probe", [0.48, 0.50, 0.52])
        write_cell(tmp_path, "fm_prelinear_ce", [0.53, 0.55, 0.57], variant="T4")
        assert table_rows(tmp_path)["fm_prelinear_ce@T4"] == "+0.050"

    def test_a_missing_baseline_row_leaves_the_delta_blank(self, tmp_path: Path) -> None:
        """No probe cell reported yet: blank, rather than a number report cannot
        support or a fall back onto whichever baseline happens to exist."""
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_cell(tmp_path, "fm_prelinear_ce", [0.53, 0.55, 0.57])
        assert table_rows(tmp_path)["fm_prelinear_ce"] == ""

    def test_report_holds_no_per_stage_branch(self) -> None:
        """The rule lives on the heads; `report` only resolves it.

        Scoped to the delta path. `expected_runs` still names the prototype
        head, and that stays: it is a protocol fact about run counts, not a
        baseline rule, and the write-up ties it to that head by name.

        Checked as string literals rather than as substrings: the header prose
        names both baselines, which is the table explaining itself to a reader,
        and a branch would need the key itself.
        """
        source = inspect.getsource(report_module._dacc) + inspect.getsource(
            report_module._cell_means
        )
        for head in (
            "prototype",
            "linear_probe",
            "fm_standard",
            "fm_rolled",
            "fm_prelinear_ce",
            "fm_prelinear_guided",
        ):
            assert f'"{head}"' not in source
            assert f"'{head}'" not in source


class TestTheTitleNamesWhatTheTableHolds:
    """The last open Stage 2 caveat, deferred from 10.0 to this commit."""

    def test_the_title_is_not_stage_1_only(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_table(tmp_path, tmp_path / "TABLE.md")
        first_line = (tmp_path / "TABLE.md").read_text().splitlines()[0]
        assert first_line != "# Stage 1 results"
        assert "results" in first_line.lower()

    def test_the_header_explains_both_baselines(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "prototype", [0.38, 0.40, 0.42])
        write_table(tmp_path, tmp_path / "TABLE.md")
        header = (tmp_path / "TABLE.md").read_text().split("|")[0]
        assert "prototype" in header and "linear probe" in header


def table_rows(results: Path) -> dict[str, str]:
    """Every data row's method label mapped to its dAcc cell."""
    write_table(results, results / "TABLE.md")
    rows = {}
    for line in (results / "TABLE.md").read_text().splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Dataset" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[2]] = cells[-1]
    return rows
