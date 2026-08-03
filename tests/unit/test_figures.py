"""Figure tests per PRD_figures section 5.

Figures are graded deliverables, so the rules that constrain them are tested:
the projection is fitted jointly to features and prototypes, class colors are
stable across encoders, confusion rows are normalized, error bars come from the
stored std, and a missing run refuses rather than plotting a partial panel.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from fm_fewshot.services.evaluation import figures  # noqa: E402
from fm_fewshot.services.evaluation.metrics import confusion_matrix  # noqa: E402


class RecordingProjector:
    """Stands in for t-SNE or PCA and records what it was fitted on."""

    def __init__(self) -> None:
        self.fit_rows: int | None = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit_rows = x.shape[0]
        return x[:, :2]


class TestJointProjection:
    def test_projection_is_fitted_on_features_and_prototypes_together(self) -> None:
        """The write-up requires a joint fit; transforming prototypes after
        fitting on features alone is a different figure and is wrong here."""
        features = np.random.default_rng(0).standard_normal((40, 6))
        prototypes = np.random.default_rng(1).standard_normal((4, 6))
        projector = RecordingProjector()

        points, protos = figures.project_jointly(features, prototypes, projector)

        assert projector.fit_rows == 44
        assert points.shape == (40, 2)
        assert protos.shape == (4, 2)

    def test_split_is_at_the_right_boundary(self) -> None:
        features = np.arange(30, dtype=np.float64).reshape(10, 3)
        prototypes = np.full((2, 3), 999.0)
        points, protos = figures.project_jointly(features, prototypes, RecordingProjector())
        assert (protos == 999.0).all()
        assert not (points == 999.0).any()


class TestColorStability:
    def test_same_class_gets_the_same_color_across_calls(self) -> None:
        names = ["banded", "bubbly", "cracked", "dotted"]
        first = figures.class_colors(names)
        second = figures.class_colors(list(reversed(names)))
        assert first == second

    def test_colors_are_distinct(self) -> None:
        names = [f"c{i}" for i in range(10)]
        assert len(set(figures.class_colors(names).values())) == 10


class TestConfusion:
    def test_rows_sum_to_one(self) -> None:
        preds = torch.tensor([0, 0, 1, 1, 2, 2])
        labels = torch.tensor([0, 1, 1, 2, 2, 0])
        matrix = confusion_matrix(preds, labels, 3)
        assert torch.allclose(matrix.sum(dim=1), torch.ones(3, dtype=torch.float64))

    def test_matches_sklearn(self) -> None:
        from sklearn.metrics import confusion_matrix as sk_confusion

        rng = np.random.default_rng(3)
        labels = rng.integers(0, 5, 200)
        preds = rng.integers(0, 5, 200)
        ours = confusion_matrix(torch.from_numpy(preds), torch.from_numpy(labels), 5)
        theirs = sk_confusion(labels, preds, normalize="true")
        assert np.allclose(ours.numpy(), theirs)

    def test_absent_class_row_stays_zero_rather_than_nan(self) -> None:
        matrix = confusion_matrix(torch.tensor([0, 0]), torch.tensor([0, 0]), 3)
        assert torch.isfinite(matrix).all()
        assert float(matrix[1].sum()) == 0.0

    def test_top_confusions_are_ranked(self) -> None:
        """With 47 and 100 classes the matrix is unreadable, so the note needs
        the worst pairs in prose."""
        labels = torch.tensor([0, 0, 0, 1, 1, 2])
        preds = torch.tensor([1, 1, 0, 1, 1, 2])
        matrix = confusion_matrix(preds, labels, 3)
        top = figures.top_confusions(matrix, ["a", "b", "c"], limit=1)
        assert top[0][0] == "a" and top[0][1] == "b"
        assert top[0][2] == pytest.approx(2 / 3)


class TestSizeCurve:
    def test_error_bars_equal_the_stored_std(self, tmp_path: Path) -> None:
        cells = [
            _cell("dtd", "resnet18", "prototype", 5, 0.48, 0.013, 3),
            _cell("dtd", "resnet18", "prototype", 10, 0.52, 0.002, 3),
            _cell("dtd", "resnet18", "prototype", None, 0.59, 0.0, 1),
        ]
        series = figures.size_curve_series(cells, "dtd", "resnet18")
        assert series["prototype"]["x"] == ["5", "10", "full"]
        assert series["prototype"]["yerr"] == [0.013, 0.002, None]

    def test_single_run_point_carries_no_error_bar(self, tmp_path: Path) -> None:
        cells = [_cell("dtd", "resnet18", "prototype", None, 0.59, 0.0, 1)]
        series = figures.size_curve_series(cells, "dtd", "resnet18")
        assert series["prototype"]["yerr"] == [None]


class TestMissingRuns:
    def test_absent_cell_refuses_rather_than_plotting_a_partial_panel(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(figures.MissingRunError, match="dtd"):
            figures.require_cells([], "dtd", "resnet18", ["prototype"])

    def test_present_cells_pass(self) -> None:
        cells = [_cell("dtd", "resnet18", "prototype", 5, 0.48, 0.01, 3)]
        figures.require_cells(cells, "dtd", "resnet18", ["prototype"])


class TestVizClasses:
    def test_rejects_a_list_outside_eight_to_ten(self) -> None:
        with pytest.raises(ValueError, match="8 to 10"):
            figures.validate_viz_classes(["a", "b"], ["a", "b", "c"])

    def test_rejects_a_class_the_dataset_does_not_have(self) -> None:
        names = [f"c{i}" for i in range(10)]
        with pytest.raises(ValueError, match="not in the dataset"):
            figures.validate_viz_classes([*names[:9], "nope"], names)

    def test_accepts_a_valid_list(self) -> None:
        names = [f"c{i}" for i in range(20)]
        figures.validate_viz_classes(names[:9], names)


def _cell(dataset, encoder, head, k, mean, std, n_runs):
    from fm_fewshot.shared.contracts import CellSummary

    return CellSummary(
        dataset=dataset,
        encoder=encoder,
        head=head,
        k=k,
        run_ids=tuple(f"r{i}" for i in range(n_runs)),
        mean=mean,
        std=std,
        n_runs=n_runs,
    )


def _write_summary(results: Path, run_id: str, **cfg) -> None:
    run_dir = results / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"config": cfg}), encoding="utf-8")
