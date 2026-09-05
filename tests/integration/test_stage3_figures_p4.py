"""P4 has to accept what the `rowspace` command actually writes.

The synthetic files in `test_stage3_figures` are built to the documented schema;
this runs the real diagnostic on a trained Stage 3 run and feeds its output
through P4's loader and plot, so a tolerance that is right on paper and wrong on
float32 norms would fail here rather than on the graded grid.
"""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from conftest import STUB_DATASET  # noqa: E402
from fm_fewshot import sdk  # noqa: E402
from fm_fewshot.services.evaluation import figures  # noqa: E402
from fm_fewshot.services.evaluation import make_stage3_figures as stage3  # noqa: E402
from fm_fewshot.shared.contracts import ExperimentConfig  # noqa: E402

PARAMS: dict[str, object] = {
    "sample_steps": 2,
    "n_train_steps": 6,
    "hidden_dims": [8, 8],
    "batch_size": 4,
    "lr": 5e-1,
    "eval_every": 0,
}


@pytest.fixture
def run_dir(tmp_path: Path, stub_dataset: str) -> Path:
    for split in ("train", "val", "test"):
        sdk.build_features(
            STUB_DATASET, split, encoder="stub", data_root=tmp_path, allow_heavy_on_cpu=True
        )
    cfg = ExperimentConfig(
        run_name="p4-real",
        dataset=STUB_DATASET,
        encoder="stub",
        head="fm_prelinear_ce",
        head_params=dict(PARAMS),
        variant="T2",
        k=2,
    )
    summary = sdk.run_experiment(cfg, data_root=tmp_path, results_dir=tmp_path / "results")
    written = sdk.run_rowspace_diagnostics(
        cfg, data_root=tmp_path, results_dir=tmp_path / "results"
    )
    assert written.parent.name == summary.run_id
    return written.parent


class TestAgainstTheRealDiagnostic:
    def test_the_written_file_passes_p4_s_orthogonality_check(self, run_dir: Path) -> None:
        group = stage3.load_rowspace(run_dir, name="fm_prelinear_ce T=2")
        assert group["row_fraction"].shape == group["null_fraction"].shape
        assert group["row_norm"].shape[0] >= group["row_fraction"].shape[0]
        assert (group["row_fraction"] >= 0.0).all()
        assert (group["row_fraction"] <= 1.0 + 1e-6).all()

    def test_p4_draws_from_it(self, run_dir: Path, tmp_path: Path) -> None:
        out = figures.plot_displacement(
            groups=[stage3.load_rowspace(run_dir, name="fm_prelinear_ce T=2")],
            title="stub / stub, K=2",
            out=tmp_path / "p4.png",
        )
        assert out.stat().st_size > 0
