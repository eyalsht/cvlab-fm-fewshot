"""The cross-head subset guard behind `scripts/check_subsets` (TODO 8.4).

The scientific claim of the project is that only the head changed. Subsets are
drawn from a seeded sampler shared by every head, so at one
(dataset, encoder, k, subset_seed) every run must have fitted on exactly the
same rows. `RunSummary` records those rows rather than a hash of them so the
claim can be checked by reading two summaries, and this is the reader.

It is a gate, so its two failure modes matter equally: a mismatch it misses is
a silent confound in the table, and a mismatch it invents blocks the sweep for
nothing. The tests below cover both directions on synthetic result trees.
"""

import json
from pathlib import Path

import pytest

from fm_fewshot.services.evaluation.subset_check import (
    find_classifier_disagreements,
    find_subset_disagreements,
    main,
)


def write_run(
    results_dir: Path,
    run_id: str,
    *,
    subset_idx: list[int],
    head: str = "prototype",
    dataset: str = "dtd",
    encoder: str = "resnet18",
    k: int | None = 5,
    subset_seed: int = 0,
    init_seed: int = 0,
    classifier_digest: str | None = None,
) -> Path:
    """One run directory holding the fields the guard reads, and nothing else."""
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "subset_idx": subset_idx,
        "classifier_digest": classifier_digest,
        "config": {
            "dataset": dataset,
            "encoder": encoder,
            "head": head,
            "k": k,
            "subset_seed": subset_seed,
            "init_seed": init_seed,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


class TestEmptyTrees:
    def test_a_missing_results_directory_is_not_a_failure(self, tmp_path: Path) -> None:
        """The worktree of anyone who has not run the grid yet. Nothing to check."""
        assert main([str(tmp_path / "results")]) == 0
        assert find_subset_disagreements(tmp_path / "results") == []

    def test_an_empty_results_directory_is_not_a_failure(self, tmp_path: Path) -> None:
        (tmp_path / "results").mkdir()
        assert main([str(tmp_path / "results")]) == 0

    def test_a_tree_holding_no_summaries_is_not_a_failure(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        (results / ".partial_abc").mkdir(parents=True)
        (results / "TABLE.md").write_text("a table", encoding="utf-8")
        assert main([str(results)]) == 0


class TestAgreement:
    def test_two_heads_on_the_same_rows_pass(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        write_run(results, "a_prototype", subset_idx=[1, 4, 9], head="prototype")
        write_run(results, "b_fm_standard", subset_idx=[1, 4, 9], head="fm_standard")
        write_run(results, "c_fm_rolled", subset_idx=[1, 4, 9], head="fm_rolled")
        assert find_subset_disagreements(results) == []
        assert main([str(results)]) == 0

    def test_different_settings_may_hold_different_rows(self, tmp_path: Path) -> None:
        """Grouping is per (dataset, encoder, k, subset_seed); across those it is
        the sampler's job to differ, not a fault."""
        results = tmp_path / "results"
        write_run(results, "a", subset_idx=[1, 2], subset_seed=0)
        write_run(results, "b", subset_idx=[3, 4], subset_seed=1)
        write_run(results, "c", subset_idx=[5, 6], k=10)
        write_run(results, "d", subset_idx=[7, 8], encoder="dinov2_vits14")
        write_run(results, "e", subset_idx=[9, 10], dataset="fgvc_aircraft")
        assert main([str(results)]) == 0

    def test_a_single_run_in_a_cell_has_nothing_to_disagree_with(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        write_run(results, "only", subset_idx=[2, 3])
        assert main([str(results)]) == 0


class TestDisagreement:
    def test_a_mismatch_exits_non_zero(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        write_run(results, "a_prototype", subset_idx=[1, 4, 9], head="prototype")
        write_run(results, "b_fm_standard", subset_idx=[1, 4, 8], head="fm_standard")
        assert main([str(results)]) != 0

    def test_the_report_names_the_runs_that_disagree(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = tmp_path / "results"
        write_run(results, "a_prototype", subset_idx=[1, 4, 9], head="prototype")
        write_run(results, "b_fm_standard", subset_idx=[1, 4, 8], head="fm_standard")
        write_run(results, "c_fm_rolled", subset_idx=[1, 4, 9], head="fm_rolled")
        main([str(results)])
        out = capsys.readouterr().out
        assert "a_prototype" in out
        assert "b_fm_standard" in out
        assert "dtd" in out and "resnet18" in out

    def test_a_different_length_subset_is_a_mismatch(self, tmp_path: Path) -> None:
        """A head that fitted on more rows than another is the worst case here."""
        results = tmp_path / "results"
        write_run(results, "a", subset_idx=[1, 2, 3], head="prototype")
        write_run(results, "b", subset_idx=[1, 2, 3, 4], head="linear_probe")
        assert main([str(results)]) != 0

    def test_order_counts_because_the_rows_are_recorded_in_order(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        write_run(results, "a", subset_idx=[1, 2, 3], head="prototype")
        write_run(results, "b", subset_idx=[3, 2, 1], head="fm_standard")
        assert main([str(results)]) != 0

    def test_the_full_setting_is_checked_across_initialization_seeds(
        self, tmp_path: Path
    ) -> None:
        """k=None fixes the subset and varies init_seed, so all three runs of a
        full cell must still hold the identical rows."""
        results = tmp_path / "results"
        write_run(results, "a", subset_idx=[0, 1, 2], k=None, init_seed=0)
        write_run(results, "b", subset_idx=[0, 1, 2], k=None, init_seed=1)
        write_run(results, "c", subset_idx=[0, 1, 3], k=None, init_seed=2)
        assert main([str(results)]) != 0

    def test_one_disagreement_is_reported_per_setting(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        write_run(results, "a", subset_idx=[1], head="prototype")
        write_run(results, "b", subset_idx=[2], head="fm_standard")
        write_run(results, "c", subset_idx=[3], head="fm_rolled")
        disagreements = find_subset_disagreements(results)
        assert len(disagreements) == 1
        assert len(disagreements[0].runs) == 3


class TestDefaults:
    def test_the_default_directory_is_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        write_run(Path("results"), "a", subset_idx=[1, 2], head="prototype")
        write_run(Path("results"), "b", subset_idx=[9, 9], head="fm_standard")
        assert main([]) != 0
        assert main(["results"]) != 0


class TestTheScript:
    """The DoD is a script in the gate, not a function someone remembers to call."""

    ROOT = Path(__file__).resolve().parents[2]

    def test_the_script_exists_and_is_executable(self) -> None:
        script = self.ROOT / "scripts" / "check_subsets"
        assert script.is_file()
        assert script.stat().st_mode & 0o111, "check_subsets must be executable"

    def test_the_gate_runs_it(self) -> None:
        gate = (self.ROOT / "scripts" / "check").read_text(encoding="utf-8")
        assert "check_subsets" in gate


class TestTheFrozenClassifierIsTheStage1Probe:
    """TODO 10.6, the second half of the guard.

    Stage 3's claim is that its two strategies transport into the same frozen
    classifier the Stage 1 row reports, refitted at the same init_seed. Every
    head that fits one records its digest, so the claim is checked by reading
    finished summaries rather than trusted because ADR-031 says so.

    The grouping is finer than the subset guard's by one field. The probe
    depends on init_seed, and the full setting varies exactly that, so
    (dataset, encoder, k, subset_seed) would compare three deliberately
    different classifiers and fail on a correct grid.
    """

    def test_a_probe_and_two_stage_3_runs_agreeing_pass(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        for run_id, head in (
            ("a_probe", "linear_probe"),
            ("b_ce", "fm_prelinear_ce"),
            ("c_guided", "fm_prelinear_guided"),
        ):
            write_run(results, run_id, subset_idx=[1, 4, 9], head=head, classifier_digest="abc")
        assert find_classifier_disagreements(results) == []
        assert main([str(results)]) == 0

    def test_a_stage_3_run_that_refitted_a_different_classifier_is_named(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = tmp_path / "results"
        write_run(results, "a_probe", subset_idx=[1, 4], head="linear_probe",
                  classifier_digest="abc")
        write_run(results, "b_ce", subset_idx=[1, 4], head="fm_prelinear_ce",
                  classifier_digest="def")
        disagreements = find_classifier_disagreements(results)
        assert len(disagreements) == 1
        assert main([str(results)]) == 1
        printed = capsys.readouterr().out
        assert "a_probe" in printed and "b_ce" in printed
        assert "classifier" in printed

    def test_the_full_setting_may_hold_three_different_classifiers(
        self, tmp_path: Path
    ) -> None:
        """k=full varies init_seed, so the three probes differ by design."""
        results = tmp_path / "results"
        for seed, digest in enumerate(("aaa", "bbb", "ccc")):
            write_run(
                results,
                f"probe_s{seed}",
                subset_idx=[1, 4],
                head="linear_probe",
                k=None,
                init_seed=seed,
                classifier_digest=digest,
            )
        assert find_classifier_disagreements(results) == []

    def test_the_full_setting_still_pairs_a_probe_with_its_own_stage_3_run(
        self, tmp_path: Path
    ) -> None:
        results = tmp_path / "results"
        write_run(results, "probe_s1", subset_idx=[1, 4], head="linear_probe", k=None,
                  init_seed=1, classifier_digest="bbb")
        write_run(results, "ce_s1", subset_idx=[1, 4], head="fm_prelinear_ce", k=None,
                  init_seed=1, classifier_digest="zzz")
        assert len(find_classifier_disagreements(results)) == 1

    def test_runs_without_a_classifier_are_ignored(self, tmp_path: Path) -> None:
        """A prototype or Stage 2 run fits no linear map and records no digest."""
        results = tmp_path / "results"
        write_run(results, "a_probe", subset_idx=[1, 4], head="linear_probe",
                  classifier_digest="abc")
        write_run(results, "b_prototype", subset_idx=[1, 4], head="prototype")
        write_run(results, "c_fm_rolled", subset_idx=[1, 4], head="fm_rolled")
        assert find_classifier_disagreements(results) == []
        assert main([str(results)]) == 0

    def test_a_run_written_before_the_digest_existed_is_reported_not_failed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The 156 Stage 1 and Stage 2 runs already in the store carry no digest.

        They cannot bind the check and must not break it either; the count is
        printed so it is visible that they do not.
        """
        results = tmp_path / "results"
        write_run(results, "a_probe_old", subset_idx=[1, 4], head="linear_probe")
        write_run(results, "b_ce", subset_idx=[1, 4], head="fm_prelinear_ce",
                  classifier_digest="abc")
        assert main([str(results)]) == 0
        assert "1 head that fits a classifier carries no digest" in capsys.readouterr().out

    def test_one_recorded_classifier_alone_is_not_a_disagreement(
        self, tmp_path: Path
    ) -> None:
        results = tmp_path / "results"
        write_run(results, "a_ce", subset_idx=[1, 4], head="fm_prelinear_ce",
                  classifier_digest="abc")
        assert find_classifier_disagreements(results) == []
