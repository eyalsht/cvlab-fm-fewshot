"""End-to-end figure generation against a stub dataset and stub head."""

from pathlib import Path

import matplotlib
import pytest
import yaml

from conftest import STUB_CLASSES, STUB_DATASET

matplotlib.use("Agg")

from fm_fewshot import sdk  # noqa: E402
from fm_fewshot.services.evaluation.figures import MissingRunError  # noqa: E402
from fm_fewshot.shared.contracts import ExperimentConfig  # noqa: E402


@pytest.fixture
def figure_config(tmp_path: Path) -> Path:
    spec = {
        "projection": "pca",
        "viz_seed": 0,
        "max_per_class": 5,
        "dpi": 100,
        "datasets": {
            STUB_DATASET: {
                # The stub has 3 classes; the 8-to-10 rule is a real dataset
                # constraint, so this test drives the path with the check
                # relaxed by monkeypatching rather than by weakening the rule.
                "viz_classes": list(STUB_CLASSES),
                "confusion_setting": {
                    "encoder": "stub",
                    "head": "linear_probe",
                    "k": "full",
                },
            }
        },
    }
    path = tmp_path / "figures.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


@pytest.fixture
def populated(tmp_path: Path, stub_dataset: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    from fm_fewshot.services.evaluation import figures as figures_module

    monkeypatch.setattr(figures_module, "validate_viz_classes", lambda chosen, avail: None)
    for split in ("train", "val", "test"):
        sdk.build_features(
            STUB_DATASET, split, encoder="stub", data_root=tmp_path, allow_heavy_on_cpu=True
        )
    results = tmp_path / "results"
    for head in ("prototype", "linear_probe"):
        for k, seeds in ((5, (0, 1, 2)), (10, (0, 1, 2)), (None, (0, 1, 2))):
            if head == "prototype" and k is None:
                seeds = (0,)
            for seed in seeds:
                sdk.run_experiment(
                    ExperimentConfig(
                        run_name=f"{head}-{k}-{seed}",
                        dataset=STUB_DATASET,
                        encoder="stub",
                        head=head,
                        head_params={"max_epochs": 3},
                        k=k,
                        subset_seed=seed if k is not None else 0,
                        init_seed=seed if k is None else 0,
                    ),
                    data_root=tmp_path,
                    results_dir=results,
                )
    return tmp_path


class TestGeneration:
    def test_writes_all_four_families(self, populated: Path, figure_config: Path) -> None:
        written = sdk.make_figures(
            figure_config,
            results_dir=populated / "results",
            data_root=populated,
            assets_dir=populated / "assets",
        )
        names = {Path(p).name for p in written}
        assert any(n.startswith("size_curve") for n in names)
        assert any(n.startswith("loss_curves") for n in names)
        assert any(n.startswith("features") for n in names)
        assert any(n.startswith("confusion") for n in names)
        assert all(Path(p).exists() for p in written)

    def test_series_csv_accompanies_the_size_curve(
        self, populated: Path, figure_config: Path
    ) -> None:
        sdk.make_figures(
            figure_config,
            results_dir=populated / "results",
            data_root=populated,
            assets_dir=populated / "assets",
        )
        csv_path = populated / "assets" / STUB_DATASET / "size_curve_stub.csv"
        assert csv_path.exists()
        assert "head,k,top1,std" in csv_path.read_text()

    def test_top_confusions_csv_is_written(
        self, populated: Path, figure_config: Path
    ) -> None:
        sdk.make_figures(
            figure_config,
            results_dir=populated / "results",
            data_root=populated,
            assets_dir=populated / "assets",
        )
        path = populated / "assets" / STUB_DATASET / "top_confusions.csv"
        assert path.exists()
        assert path.read_text().startswith("true,predicted,rate")

    def test_regeneration_is_byte_identical(
        self, populated: Path, figure_config: Path
    ) -> None:
        kwargs = {
            "results_dir": populated / "results",
            "data_root": populated,
            "assets_dir": populated / "assets",
        }
        written = sdk.make_figures(figure_config, **kwargs)
        before = {p: Path(p).read_bytes() for p in written}
        sdk.make_figures(figure_config, **kwargs)
        assert all(Path(p).read_bytes() == before[p] for p in written)


class TestRefusal:
    def test_dataset_without_runs_refuses_by_name(
        self, populated: Path, figure_config: Path, tmp_path: Path
    ) -> None:
        empty = tmp_path / "no_runs"
        empty.mkdir()
        with pytest.raises(MissingRunError, match=STUB_DATASET):
            sdk.make_figures(
                figure_config,
                results_dir=empty,
                data_root=populated,
                assets_dir=populated / "assets",
            )

    def test_missing_representative_run_refuses(
        self, populated: Path, figure_config: Path
    ) -> None:
        spec = yaml.safe_load(figure_config.read_text())
        spec["datasets"][STUB_DATASET]["confusion_setting"]["head"] = "no_such_head"
        figure_config.write_text(yaml.safe_dump(spec), encoding="utf-8")
        with pytest.raises(MissingRunError):
            sdk.make_figures(
                figure_config,
                results_dir=populated / "results",
                data_root=populated,
                assets_dir=populated / "assets",
            )
