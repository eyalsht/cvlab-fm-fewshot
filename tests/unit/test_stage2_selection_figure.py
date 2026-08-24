"""S7, the figure that says why a run stopped where it did (TODO 8.9, ADR-028).

S2 answers the write-up's stated question for the training curves, which is
whether training was stable. It cannot answer the question a reader asks next:
the run trained 2000 steps, so why is the reported number the field from step
`best_epoch`?

The honest answer is not "the loss was lowest there". Across the whole grid the
training loss is still falling at the step selection keeps, which is precisely
why the rule is validation accuracy and not loss. S7 draws that: the loss on
one row, validation top-1 on another, and one vertical rule through both at the
kept step, so the gap between the kept step and the loss minimum is read off the
figure instead of argued in prose.

What is asserted here is structure, never a pixel: that the marked step is the
one `summary.json` recorded rather than one recomputed in plotting code, that
the loss-minimising step is reported separately so the two cannot be conflated,
that a run with no selection grid draws instead of raising, and that the whole
family regenerates byte-identically.

Two rows rather than two y axes on one plot is a specification, not a style
choice: a loss spanning three decades and an accuracy in [0, 1] share no scale,
and a twin axis invites the reader to read a crossing point that is an artifact
of where the two scales were pinned.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from fm_fewshot.services.evaluation import figures


def _loss(n: int) -> list[float]:
    """Monotonically falling, so the minimum is unambiguously the last step."""
    return [100.0 / step for step in range(1, n + 1)]


def _steps(n: int) -> list[int]:
    return list(range(1, n + 1))


N = 20
# Validation peaks early and decays: the memorization shape the grid showed.
VAL = [(5, 0.51), (10, 0.42), (15, 0.37), (20, 0.35)]
SELECTED = 5


def _panel(**overrides):
    panel = {
        "name": "fm_rolled T=12",
        "steps": _steps(N),
        "losses": _loss(N),
        "validation": list(VAL),
        "selected_step": SELECTED,
    }
    panel.update(overrides)
    return panel


class TestSelectionSummary:
    def test_reports_the_selected_step_and_the_loss_minimum_apart(self) -> None:
        summary = figures.selection_note(
            steps=_steps(N), losses=_loss(N), validation=VAL, selected_step=SELECTED
        )
        assert "5" in summary
        assert "20" in summary, "the loss-minimising step has to appear, not just the kept one"

    def test_says_plainly_when_the_loss_was_still_falling(self) -> None:
        summary = figures.selection_note(
            steps=_steps(N), losses=_loss(N), validation=VAL, selected_step=SELECTED
        )
        assert "still falling" in summary

    def test_scores_the_kept_step_itself_not_the_best_point_on_the_curve(self) -> None:
        """A rule that kept a worse step than it found is a bug the figure must show."""
        summary = figures.selection_note(
            steps=_steps(N), losses=_loss(N),
            validation=[(5, 0.51), (10, 0.90)], selected_step=5,
        )
        assert "0.510" in summary
        assert "0.900" not in summary

    def test_says_so_when_the_kept_step_is_the_loss_minimum(self) -> None:
        """The other case has to be distinguishable, or the label means nothing."""
        summary = figures.selection_note(
            steps=_steps(N), losses=_loss(N), validation=[(20, 0.7)], selected_step=N
        )
        assert "still falling" not in summary

    def test_names_the_absent_selection_grid_rather_than_inventing_a_step(self) -> None:
        summary = figures.selection_note(
            steps=_steps(N), losses=_loss(N), validation=[], selected_step=None
        )
        assert "no selection" in summary.lower()
        assert summary.strip().endswith("left"), "the label has to be a finished sentence"
        assert "step 0" not in summary and "None" not in summary


class TestPlot:
    def test_draws_two_rows_per_scheme_and_never_a_twin_axis(self, tmp_path) -> None:
        out = figures.plot_selection(
            panels=[_panel(), _panel(name="fm_standard T=12")],
            title="dtd / resnet18, K=10",
            out=tmp_path / "sel.png",
        )
        assert out.exists() and out.stat().st_size > 0

    def test_a_run_without_a_selection_grid_still_draws(self, tmp_path) -> None:
        """`eval_every: 0` is the reversal ADR-028 holds open; it must not crash S7."""
        out = figures.plot_selection(
            panels=[_panel(validation=[], selected_step=None)],
            title="no selection",
            out=tmp_path / "none.png",
        )
        assert out.exists() and out.stat().st_size > 0

    def test_regeneration_is_byte_identical(self, tmp_path) -> None:
        first = figures.plot_selection(
            panels=[_panel()], title="t", out=tmp_path / "a.png"
        ).read_bytes()
        second = figures.plot_selection(
            panels=[_panel()], title="t", out=tmp_path / "b.png"
        ).read_bytes()
        assert first == second

    def test_refuses_a_selected_step_outside_the_recorded_run(self, tmp_path) -> None:
        """A marker off the end of the axis would be a silent lie about the run."""
        with pytest.raises(ValueError, match="outside"):
            figures.plot_selection(
                panels=[_panel(selected_step=N + 1)],
                title="t",
                out=tmp_path / "bad.png",
            )
