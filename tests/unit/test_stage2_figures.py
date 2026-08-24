"""Stage 2 figure tests per PRD_stage2_figures section 3.

Same standard as the Stage 1 families: nothing here asserts what a figure
shows, only the rules that make it a defensible deliverable. The projection is
fitted once over all three feature sets and the prototypes, the colors are the
ones the dataset was already committed to in config, S3's panels hold the same
test rows, S4's plotted states are the Euler states of the refitted field, and
a missing run refuses by name instead of leaving a panel half drawn.

The runs are synthetic. There is no Stage 2 grid on disk yet, and results/ and
data/ are gitignored anyway, so the tree and the feature caches are built in
tmp_path exactly as the Stage 1 figure tests build theirs.
"""

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch
import yaml

matplotlib.use("Agg")

from fm_fewshot.services.evaluation import figures  # noqa: E402
from fm_fewshot.services.evaluation import make_stage2_figures as stage2  # noqa: E402
from fm_fewshot.services.features import cache  # noqa: E402
from fm_fewshot.shared.config import save_config  # noqa: E402
from fm_fewshot.shared.contracts import ExperimentConfig  # noqa: E402

DATASET = "figstub"
ENCODER = "stub"
CLASS_NAMES = [f"class_{i}" for i in range(9)]
DIM = 6
# Fixed per split, never hash-derived: PYTHONHASHSEED varies between
# processes and the figures must regenerate byte-identically.
PER_CLASS = {"train": 12, "val": 3, "test": 5}
SPLIT_SEED = {"train": 11, "val": 12, "test": 13}
T_VALUES = (4, 12)
# Small enough that a refit costs milliseconds; the figures do not care how
# well the field fits, only that the fit is reproducible from the config.
FM_PARAMS = {"n_train_steps": 4, "batch_size": 8, "hidden_dims": [8, 8]}


class RecordingProjector:
    """Stands in for PCA and records every fit it was asked to perform.

    Returns the first two columns unchanged, so a projected panel can be
    compared against the rows that produced it.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.calls.append(x.shape[0])
        return np.asarray(x, dtype=np.float64)[:, :2]


# --------------------------------------------------------------------------
# synthetic run tree
# --------------------------------------------------------------------------


def _write_cache(root: Path, split: str) -> None:
    rng = np.random.default_rng(SPLIT_SEED[split])
    per_class = PER_CLASS[split]
    labels = np.repeat(np.arange(len(CLASS_NAMES)), per_class).astype(np.int64)
    centers = np.random.default_rng(7).standard_normal((len(CLASS_NAMES), DIM))
    features = centers[labels] + 0.3 * rng.standard_normal((labels.shape[0], DIM))

    directory = cache.cache_dir(root, DATASET, ENCODER)
    directory.mkdir(parents=True, exist_ok=True)
    npz = directory / f"{split}.npz"
    np.savez(npz, features=features.astype(np.float32), labels=labels)
    (directory / f"{split}_meta.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "split": split,
                "encoder": ENCODER,
                "model_name": ENCODER,
                "weights_tag": ENCODER,
                "N": int(labels.shape[0]),
                "D": DIM,
                "class_names": CLASS_NAMES,
                "normalized": False,
                "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def _write_run(results_dir: Path, cfg: ExperimentConfig, top1: float) -> None:
    run_dir = results_dir / cfg.run_name
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"run_id": cfg.run_name, "config": asdict(cfg), "test_top1": top1},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if cfg.head.startswith("fm_"):
        rows = ["step,train_loss,val_top1"]
        rows += [f"{s},{1.0 / s:.6f}," for s in range(1, FM_PARAMS["n_train_steps"] + 1)]
        (run_dir / "loss_curve.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _k_tag(k: int | None) -> str:
    return "full" if k is None else str(k)


def _write_runs(results_dir: Path) -> None:
    top1 = 0.30
    for k in (5, 10, None):
        seeds = (0, 1, 2)
        proto_seeds = (0,) if k is None else seeds
        for seed in proto_seeds:
            top1 += 0.005
            _write_run(
                results_dir,
                ExperimentConfig(
                    run_name=f"prototype-{_k_tag(k)}-{seed}",
                    dataset=DATASET,
                    encoder=ENCODER,
                    head="prototype",
                    k=k,
                    subset_seed=0 if k is None else seed,
                    init_seed=seed if k is None else 0,
                ),
                top1,
            )
        for head in ("fm_standard", "fm_rolled"):
            for t in T_VALUES:
                for seed in seeds:
                    top1 += 0.005
                    _write_run(
                        results_dir,
                        ExperimentConfig(
                            run_name=f"{head}-T{t}-{_k_tag(k)}-{seed}",
                            dataset=DATASET,
                            encoder=ENCODER,
                            head=head,
                            head_params={"sample_steps": t, **FM_PARAMS},
                            k=k,
                            subset_seed=0 if k is None else seed,
                            init_seed=seed if k is None else 0,
                        ),
                        top1,
                    )
    # A Stage 1 run sharing the results directory: the Stage 2 panels are
    # defined by the write-up's five configurations and must ignore it.
    for seed in (0, 1, 2):
        _write_run(
            results_dir,
            ExperimentConfig(
                run_name=f"linear_probe-10-{seed}",
                dataset=DATASET,
                encoder=ENCODER,
                head="linear_probe",
                k=10,
                subset_seed=seed,
            ),
            0.4,
        )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for split in PER_CLASS:
        _write_cache(tmp_path, split)
    _write_runs(tmp_path / "results")
    return tmp_path


@pytest.fixture
def spec_path(tmp_path: Path) -> Path:
    spec = {
        "projection": "pca",
        "viz_seed": 0,
        "max_per_class": 4,
        "dpi": 100,
        "datasets": {
            DATASET: {
                "viz_classes": list(CLASS_NAMES),
                "confusion_setting": {"encoder": ENCODER, "head": "prototype", "k": "full"},
            }
        },
        "stage2": {
            "representative_k": 10,
            "representative_subset_seed": 0,
            "t_values": list(T_VALUES),
            "feature_t": 4,
            "trajectory_classes": 3,
            "trajectory_per_class": 2,
        },
    }
    path = tmp_path / "figures.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def _generate(tree: Path, spec_path: Path) -> list[Path]:
    return stage2.make_stage2_figures(
        spec_path,
        results_dir=tree / "results",
        data_root=tree,
        assets_dir=tree / "assets",
    )


def _refit(tree: Path, head: str, t: int):  # noqa: ANN202 - Refit is stage2's own type
    run_dir = stage2.find_run(
        tree / "results",
        dataset=DATASET,
        encoder=ENCODER,
        head=head,
        sample_steps=t,
        k=10,
        subset_seed=0,
    )
    return stage2.refit(run_dir, data_root=tree)


def _test_features(tree: Path):  # noqa: ANN202
    return cache.read_features(DATASET, "test", ENCODER, data_root=tree, l2_normalize=False)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


class TestJointFit:
    def test_all_three_feature_sets_and_the_prototypes_are_fitted_in_one_call(
        self, tree: Path
    ) -> None:
        """Fitting per panel produces three unrelated pictures. The rule is
        enforced here, not left to the caller, exactly as F4 enforces it."""
        test_x, test_y, _ = _test_features(tree)
        standard, rolled = _refit(tree, "fm_standard", 4), _refit(tree, "fm_rolled", 4)
        class_ids = list(range(len(CLASS_NAMES)))
        rows, labels = figures.viz_test_rows(test_y, class_ids, 4)
        projector = RecordingProjector()

        panels = stage2.three_way_panels(
            test_x,
            rows,
            labels,
            (("standard FM", standard.head), ("rolled-out FM", rolled.head)),
            standard.head.prototypes[class_ids],
            projector,
        )

        assert projector.calls == [3 * rows.shape[0] + len(class_ids)]
        assert len(panels.panels) == 3

    def test_panels_keep_their_own_rows_after_the_joint_fit(self, tree: Path) -> None:
        test_x, test_y, _ = _test_features(tree)
        standard = _refit(tree, "fm_standard", 4)
        class_ids = list(range(len(CLASS_NAMES)))
        rows, labels = figures.viz_test_rows(test_y, class_ids, 4)

        panels = stage2.three_way_panels(
            test_x,
            rows,
            labels,
            (("standard FM", standard.head),),
            standard.head.prototypes[class_ids],
            RecordingProjector(),
        )

        assert all(points.shape == (rows.shape[0], 2) for _, points in panels.panels)
        assert panels.prototypes.shape == (len(class_ids), 2)

    def test_trajectory_panels_are_fitted_jointly(self, tree: Path) -> None:
        """The T=4 and T=12 panels only contrast if they share coordinates."""
        test_x, test_y, _ = _test_features(tree)
        short, deep = _refit(tree, "fm_standard", 4), _refit(tree, "fm_standard", 12)
        class_ids = [0, 1, 2]
        rows, labels = figures.viz_test_rows(test_y, class_ids, 2)
        projector = RecordingProjector()

        stage2.trajectory_panels(
            test_x,
            rows,
            labels,
            (("T=4", short.head), ("T=12", deep.head)),
            short.head.prototypes[class_ids],
            projector,
        )

        states = (4 + 1) * rows.shape[0] + (12 + 1) * rows.shape[0]
        assert projector.calls == [states + len(class_ids)]


class TestColorStability:
    def test_s3_and_s4_take_their_colors_from_the_config_mapping(
        self, tree: Path, spec_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same mapping Stage 1's F4 driver builds from the same config
        list, so a class keeps its color across stages and figures."""
        seen: list[dict[str, str]] = []

        def record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            seen.append(kwargs["colors"])
            return kwargs["out"]

        monkeypatch.setattr(figures, "plot_feature_panels", record)
        monkeypatch.setattr(figures, "plot_trajectory_panels", record)
        _generate(tree, spec_path)

        expected = figures.class_colors(CLASS_NAMES)
        assert seen, "neither S3 nor S4 was drawn"
        assert all(colors == expected for colors in seen)


class TestSameTestExamples:
    def test_three_panels_hold_the_same_test_rows(self, tree: Path) -> None:
        test_x, test_y, _ = _test_features(tree)
        standard, rolled = _refit(tree, "fm_standard", 4), _refit(tree, "fm_rolled", 4)
        class_ids = list(range(len(CLASS_NAMES)))
        rows, labels = figures.viz_test_rows(test_y, class_ids, 4)

        panels = stage2.three_way_panels(
            test_x,
            rows,
            labels,
            (("standard FM", standard.head), ("rolled-out FM", rolled.head)),
            standard.head.prototypes[class_ids],
            RecordingProjector(),
        )

        assert np.array_equal(panels.indices, rows.numpy())
        assert np.array_equal(panels.labels, labels)
        assert all(points.shape[0] == rows.shape[0] for _, points in panels.panels)

    def test_transported_panels_are_those_rows_after_each_scheme(self, tree: Path) -> None:
        """Not merely the same count: the same rows, transported."""
        test_x, test_y, _ = _test_features(tree)
        standard, rolled = _refit(tree, "fm_standard", 4), _refit(tree, "fm_rolled", 4)
        class_ids = list(range(len(CLASS_NAMES)))
        rows, labels = figures.viz_test_rows(test_y, class_ids, 4)

        panels = stage2.three_way_panels(
            test_x,
            rows,
            labels,
            (("standard FM", standard.head), ("rolled-out FM", rolled.head)),
            standard.head.prototypes[class_ids],
            RecordingProjector(),
        )

        raw = test_x[rows].numpy()
        assert np.allclose(panels.panels[0][1], raw[:, :2])
        for (_, points), head in zip(
            panels.panels[1:], (standard.head, rolled.head), strict=True
        ):
            assert np.allclose(points, head.transport(test_x[rows]).numpy()[:, :2])


class TestTrajectories:
    def test_plotted_states_are_the_euler_states_of_the_refitted_field(
        self, tree: Path
    ) -> None:
        """z_{k+1} = z_k + (1/T) v(z_k, k/T), recomputed from the refitted field."""
        test_x, test_y, _ = _test_features(tree)
        refit = _refit(tree, "fm_standard", 4)
        rows, labels = figures.viz_test_rows(test_y, [0, 1, 2], 2)
        states = refit.head.trajectory(test_x[rows])
        steps = int(refit.config.head_params["sample_steps"])

        for k in range(steps):
            time = torch.full((states.shape[1],), k / steps)
            with torch.no_grad():
                expected = states[k] + (1.0 / steps) * refit.head.field(states[k], time)
            assert torch.allclose(states[k + 1], expected, atol=1e-6)

        panels = stage2.trajectory_panels(
            test_x,
            rows,
            labels,
            (("standard FM", refit.head),),
            refit.head.prototypes[[0, 1, 2]],
            RecordingProjector(),
        )
        plotted = panels.panels[0][1]
        assert plotted.shape == (steps + 1, rows.shape[0], 2)
        assert np.allclose(plotted, states.numpy()[:, :, :2])

    def test_refit_is_deterministic(self, tree: Path) -> None:
        """ADR-025: no field weights are stored, so the figure is only
        reproducible if the fit is."""
        test_x, _, _ = _test_features(tree)
        first, second = _refit(tree, "fm_rolled", 4), _refit(tree, "fm_rolled", 4)
        assert torch.equal(first.head.transport(test_x), second.head.transport(test_x))


class TestMissingRuns:
    def test_absent_scheme_refuses_by_name_and_writes_nothing(
        self, tree: Path, spec_path: Path
    ) -> None:
        for run in (tree / "results").glob("fm_rolled-T12-*"):
            shutil.rmtree(run)

        with pytest.raises(figures.MissingRunError, match="fm_rolled.*12"):
            _generate(tree, spec_path)
        assert not (tree / "assets").exists()

    def test_absent_representative_run_refuses_by_name(
        self, tree: Path, spec_path: Path
    ) -> None:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        spec["stage2"]["representative_subset_seed"] = 7
        spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

        with pytest.raises(figures.MissingRunError, match="subset_seed=7"):
            _generate(tree, spec_path)


class TestGeneration:
    def test_writes_every_family(self, tree: Path, spec_path: Path) -> None:
        written = _generate(tree, spec_path)
        names = {path.name for path in written}
        assert any(name.startswith("stage2_size_curve") for name in names)
        assert any(name.startswith("stage2_loss_curves") for name in names)
        assert any(name.startswith("stage2_features") for name in names)
        assert sum(name.startswith("stage2_trajectories") for name in names) == 3
        assert all(path.exists() for path in written)

    def test_size_curve_csv_carries_every_scheme(self, tree: Path, spec_path: Path) -> None:
        _generate(tree, spec_path)
        text = (tree / "assets" / DATASET / f"stage2_size_curve_{ENCODER}.csv").read_text(
            encoding="utf-8"
        )
        rows = [line.split(",") for line in text.strip().splitlines()[1:]]
        assert {row[0] for row in rows} == {
            "prototype",
            "fm_standard T=4",
            "fm_standard T=12",
            "fm_rolled T=4",
            "fm_rolled T=12",
        }
        single = [row for row in rows if row[0] == "prototype" and row[1] == "full"]
        assert single[0][3] == "", "the single-run prototype cell has no measured spread"

    def test_regeneration_is_byte_identical(self, tree: Path, spec_path: Path) -> None:
        written = _generate(tree, spec_path)
        before = {path: path.read_bytes() for path in written}
        _generate(tree, spec_path)
        assert all(path.read_bytes() == before[path] for path in written)


class TestSchemeLabels:
    def test_label_names_the_head_and_its_step_count(self) -> None:
        assert figures.scheme_label("fm_standard", 4) == "fm_standard T=4"

    def test_cells_separate_the_two_step_counts(self, tree: Path) -> None:
        """T is not part of report's cell key, so a Stage 2 panel that grouped
        by head alone would average T=4 and T=12 into one line."""
        cells = stage2.load_stage2_cells(tree / "results", t_values=T_VALUES)
        heads = {cell.head for cell in cells}
        assert "fm_standard T=4" in heads and "fm_standard T=12" in heads
        assert all(cell.n_runs == 3 or cell.k is None for cell in cells)
        assert "linear_probe" not in heads
