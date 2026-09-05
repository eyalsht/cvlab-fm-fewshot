"""Stage 3 figure tests per PRD_stage3_figures section 3.

Same standard as the Stage 1 and Stage 2 families: nothing here asserts what a
figure shows, only the rules that make it a defensible deliverable. The
projection is fitted once over all three feature sets, the colors are the ones
the dataset was already committed to in config, P3's panels hold the same test
rows, P2 marks the step the run recorded rather than one recomputed here, and a
missing run refuses by name instead of leaving a panel half drawn.

The rule this family adds is the one Stage 1 and Stage 2 cannot state. Their
baseline is the image prototype row; Stage 3's is the linear probe it sits in
front of (ADR-035), and a P1 panel drawn against the prototypes would put the
reader's comparison against a number Stage 3 is not measured on. The baseline is
therefore taken from the registry rather than named in plotting code, and
drawing the prototype row as Stage 3's baseline is refused.

P4 reads `rowspace.json`, written by the `rowspace` command, exactly as S5 reads
`reverse.json`: no figure recomputes a diagnostic the metrics did not see. That
the decomposition is correct against a closed-form W is asserted where it is
computed, in `test_rowspace_diagnostics`. What is asserted here is that the
figure refuses a stored file whose components and total disagree, rather than
plotting fractions that do not mean what the axis label says.

The runs are synthetic. results/ and data/ are gitignored and the Stage 3 grid
does not exist yet, so the tree and the feature caches are built in tmp_path
exactly as the Stage 1 and Stage 2 figure tests build theirs.
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
from fm_fewshot.services.evaluation import make_stage3_figures as stage3  # noqa: E402
from fm_fewshot.services.evaluation.report import IncompleteCellError  # noqa: E402
from fm_fewshot.services.features import cache  # noqa: E402
from fm_fewshot.shared.config import save_config  # noqa: E402
from fm_fewshot.shared.contracts import ExperimentConfig  # noqa: E402

DATASET = "figstub"
ENCODER = "stub"
CLASS_NAMES = [f"class_{i}" for i in range(9)]
DIM = 6
# Fixed per split, never hash-derived: PYTHONHASHSEED varies between processes
# and the figures must regenerate byte-identically.
PER_CLASS = {"train": 12, "val": 3, "test": 5}
SPLIT_SEED = {"train": 11, "val": 12, "test": 13}
T_VALUES = (4, 12)
FEATURE_T = 4
REPRESENTATIVE_K = 10
STAGE3_HEADS = ("fm_prelinear_ce", "fm_prelinear_guided")

# Small enough that a refit costs milliseconds; the figures do not care how well
# the field fits, only that the fit is reproducible from the config.
FM_PARAMS: dict[str, object] = {
    "n_train_steps": 4,
    "batch_size": 8,
    "hidden_dims": [8, 8],
    "eval_every": 2,
    "probe_params": {"max_epochs": 3},
}
EVAL_EVERY = 2
N_TRAIN_STEPS = 4
# The step the run kept, placed so it is neither the loss minimum nor the
# validation argmax of the stored curve. A figure that recomputed either would
# mark a different step and the test would see it.
SELECTED_STEP = 2
VAL_ARGMAX_STEP = 4
LOSS_MIN_STEP = 4


class RecordingProjector:
    """Stands in for PCA and records every fit it was asked to perform.

    Returns the first two columns unchanged, so a projected panel can be
    compared against the rows that produced it. It exposes no `components_`,
    which is how the drawing code knows it has no linear preimage and therefore
    no decision boundary to draw.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.calls.append(x.shape[0])
        return np.asarray(x, dtype=np.float64)[:, :2]


class LinearRecordingProjector(RecordingProjector):
    """The same stub with the two attributes sklearn's PCA exposes.

    `transform(z) = (z - mean_) @ components_.T`, so the plotted plane's
    preimage is `z = mean_ + u @ components_` and the frozen probe restricted to
    it is computable in closed form.
    """

    def __init__(self, components: np.ndarray, mean: np.ndarray) -> None:
        super().__init__()
        self.components_ = np.asarray(components, dtype=np.float64)
        self.mean_ = np.asarray(mean, dtype=np.float64)


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


def _loss_curve_rows() -> list[str]:
    """Loss falling to the last step, validation rising to the last eval step.

    Both extremes therefore sit at a step the run did not keep, so a figure that
    recomputed the marked step from either column would be visible.
    """
    rows = ["step,train_loss,val_top1"]
    for step in range(1, N_TRAIN_STEPS + 1):
        val = f"{0.40 + 0.05 * step:.4f}" if step % EVAL_EVERY == 0 else ""
        rows.append(f"{step},{1.0 / step:.6f},{val}")
    return rows


def _write_run(results_dir: Path, cfg: ExperimentConfig, top1: float) -> Path:
    run_dir = results_dir / cfg.run_name
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "config.yaml")
    payload = {"run_id": cfg.run_name, "config": asdict(cfg), "test_top1": top1}
    if cfg.head.startswith("fm_"):
        payload["best_epoch"] = SELECTED_STEP
    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if cfg.head.startswith("fm_"):
        (run_dir / "loss_curve.csv").write_text(
            "\n".join(_loss_curve_rows()) + "\n", encoding="utf-8"
        )
    return run_dir


def write_rowspace(run_dir: Path, *, head: str, sample_steps: int, seed: int) -> Path:
    """A synthetic `rowspace.json` in the schema the `rowspace` command writes.

    The two components are drawn independently and the total is their
    hypotenuse, so the file satisfies the orthogonality the diagnostic
    guarantees; a test that wants a broken file breaks this one on purpose.
    """
    rng = np.random.default_rng(seed)
    n = 40
    row = np.abs(rng.standard_normal(n)) + 0.05
    null = np.abs(rng.standard_normal(n)) + 0.05
    total = np.hypot(row, null)
    before = rng.standard_normal(n)
    after = before + 0.5 * row
    payload = {
        "run_id": run_dir.name,
        "head": head,
        "sample_steps": sample_steps,
        "dim": DIM,
        "rank_w": len(CLASS_NAMES),
        "n_points": n,
        "max_points": n,
        "summary": {
            "row_fraction_mean": float(np.mean(row / total)),
            "row_fraction_median": float(np.median(row / total)),
            "null_fraction_mean": float(np.mean(null / total)),
            "null_fraction_median": float(np.median(null / total)),
            "total_norm_mean": float(np.mean(total)),
            "total_norm_median": float(np.median(total)),
        },
        "points": {
            "total_norm": total.tolist(),
            "row_norm": row.tolist(),
            "null_norm": null.tolist(),
            "margin_before": before.tolist(),
            "margin_after": after.tolist(),
        },
    }
    out = run_dir / "rowspace.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _k_tag(k: int | None) -> str:
    return "full" if k is None else str(k)


def _write_runs(results_dir: Path) -> None:
    top1 = 0.30
    for k in (5, 10, None):
        seeds = (0, 1, 2)
        # The prototype row exists on disk. It is Stage 1's and Stage 2's
        # baseline and must never be drawn as Stage 3's (ADR-035).
        for seed in (0,) if k is None else seeds:
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
        for seed in seeds:
            top1 += 0.005
            _write_run(
                results_dir,
                ExperimentConfig(
                    run_name=f"linear_probe-{_k_tag(k)}-{seed}",
                    dataset=DATASET,
                    encoder=ENCODER,
                    head="linear_probe",
                    head_params={"max_epochs": 3},
                    k=k,
                    subset_seed=0 if k is None else seed,
                    init_seed=seed if k is None else 0,
                ),
                top1,
            )
        for head in STAGE3_HEADS:
            for t in T_VALUES:
                for seed in seeds:
                    top1 += 0.005
                    run_dir = _write_run(
                        results_dir,
                        ExperimentConfig(
                            run_name=f"{head}-T{t}-{_k_tag(k)}-{seed}",
                            dataset=DATASET,
                            encoder=ENCODER,
                            head=head,
                            head_params={"sample_steps": t, **FM_PARAMS},
                            variant=f"T{t}",
                            k=k,
                            subset_seed=0 if k is None else seed,
                            init_seed=seed if k is None else 0,
                        ),
                        top1,
                    )
                    if (k, t, seed) == (REPRESENTATIVE_K, FEATURE_T, 0):
                        write_rowspace(
                            run_dir,
                            head=head,
                            sample_steps=t,
                            seed=STAGE3_HEADS.index(head) + 1,
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
        "stage3": {
            "representative_k": REPRESENTATIVE_K,
            "representative_subset_seed": 0,
            "t_values": list(T_VALUES),
            "feature_t": FEATURE_T,
            "main_encoder": {DATASET: ENCODER},
        },
    }
    path = tmp_path / "figures.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def _generate(tree: Path, spec_path: Path) -> list[Path]:
    return stage3.make_stage3_figures(
        spec_path,
        results_dir=tree / "results",
        data_root=tree,
        assets_dir=tree / "assets",
    )


def _find(tree: Path, head: str, t: int = FEATURE_T) -> Path:
    return stage3.find_run(
        tree / "results",
        dataset=DATASET,
        encoder=ENCODER,
        head=head,
        sample_steps=t,
        k=REPRESENTATIVE_K,
        subset_seed=0,
    )


def _refit(tree: Path, head: str, t: int = FEATURE_T):  # noqa: ANN202 - stage2's Refit
    return stage3.refit(_find(tree, head, t), data_root=tree)


def _test_features(tree: Path):  # noqa: ANN202
    return cache.read_features(DATASET, "test", ENCODER, data_root=tree, l2_normalize=False)


def _asset(tree: Path, name: str) -> Path:
    return tree / "assets" / DATASET / name


# --------------------------------------------------------------------------
# P1: the baseline is the linear probe, never the prototype row
# --------------------------------------------------------------------------


class TestBaselineIsTheLinearProbe:
    def test_the_baseline_comes_from_the_registry_not_from_plotting_code(self) -> None:
        """ADR-035 puts the answer on the head itself. Naming it here as well
        would let the figure and TABLE.md's dAcc column disagree without either
        being obviously wrong."""
        assert figures.stage3_baseline() == "linear_probe"

    def test_the_panel_lists_the_probe_first_then_both_strategies_at_both_t(self) -> None:
        assert figures.stage3_lines(T_VALUES) == [
            "linear_probe",
            "fm_prelinear_ce@T4",
            "fm_prelinear_ce@T12",
            "fm_prelinear_guided@T4",
            "fm_prelinear_guided@T12",
        ]

    def test_drawing_the_prototype_row_as_the_stage3_baseline_is_refused(
        self, tree: Path
    ) -> None:
        """The whole point of the family. Stage 1's F1 and Stage 2's S1 lead with
        the prototype row; a Stage 3 panel that did would invite the reader to
        compare against the wrong row."""
        cells = stage3.load_stage3_cells(tree / "results")
        with pytest.raises(ValueError, match="prototype"):
            figures.plot_stage3_size_curve(
                cells=cells,
                dataset=DATASET,
                encoder=ENCODER,
                lines=["prototype", *figures.stage3_lines(T_VALUES)[1:]],
                out=tree / "bad.png",
            )

    def test_a_panel_that_does_not_lead_with_the_baseline_is_refused(
        self, tree: Path
    ) -> None:
        cells = stage3.load_stage3_cells(tree / "results")
        with pytest.raises(ValueError, match="linear_probe"):
            figures.plot_stage3_size_curve(
                cells=cells,
                dataset=DATASET,
                encoder=ENCODER,
                lines=figures.stage3_lines(T_VALUES)[1:],
                out=tree / "bad.png",
            )

    def test_the_generated_panel_carries_the_probe_and_not_the_prototypes(
        self, tree: Path, spec_path: Path
    ) -> None:
        """Asserted on the numbers the panel was drawn from, which are written
        beside it, because the alternative is asserting on pixels."""
        _generate(tree, spec_path)
        text = _asset(tree, f"stage3_size_curve_{ENCODER}.csv").read_text(encoding="utf-8")
        rows = [line.split(",") for line in text.strip().splitlines()[1:]]
        heads = [row[0] for row in rows]
        assert heads[0] == "linear_probe"
        assert "prototype" not in set(heads)
        assert set(heads) == set(figures.stage3_lines(T_VALUES))

    def test_the_main_comparison_cell_is_marked_on_the_panel(
        self, tree: Path, spec_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """His required comparison and the extended grid are one picture, so
        nobody has to reconcile two."""
        captured: list[dict] = []

        def record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            captured.append(kwargs)
            return kwargs["out"]

        monkeypatch.setattr(figures, "plot_stage3_size_curve", record)
        _generate(tree, spec_path)

        assert captured, "P1 was not drawn"
        assert captured[0]["marked_k"] == str(REPRESENTATIVE_K)


# --------------------------------------------------------------------------
# P3: one joint fit, the same rows, the config's colors
# --------------------------------------------------------------------------


class TestJointFit:
    def test_all_three_feature_sets_are_fitted_in_one_call(self, tree: Path) -> None:
        """The write-up: compute the embedding jointly over the before/after
        feature sets being compared. Fitting per panel produces three unrelated
        pictures. No prototypes are in the fit: Stage 3 transports toward
        nothing, and a star on the panel would suggest a target that does not
        exist."""
        test_x, test_y, _ = _test_features(tree)
        ce, guided = _refit(tree, "fm_prelinear_ce"), _refit(tree, "fm_prelinear_guided")
        rows, labels = figures.viz_test_rows(test_y, list(range(len(CLASS_NAMES))), 4)
        projector = RecordingProjector()

        panels = stage3.three_way_panels(
            test_x,
            rows,
            labels,
            (("after fm_prelinear_ce", ce.head), ("after fm_prelinear_guided", guided.head)),
            projector,
        )

        assert projector.calls == [3 * rows.shape[0]]
        assert len(panels.panels) == 3
        assert panels.prototypes is None

    def test_the_three_panels_hold_the_same_test_indices(self, tree: Path) -> None:
        test_x, test_y, _ = _test_features(tree)
        ce, guided = _refit(tree, "fm_prelinear_ce"), _refit(tree, "fm_prelinear_guided")
        rows, labels = figures.viz_test_rows(test_y, list(range(len(CLASS_NAMES))), 4)

        panels = stage3.three_way_panels(
            test_x,
            rows,
            labels,
            (("after fm_prelinear_ce", ce.head), ("after fm_prelinear_guided", guided.head)),
            RecordingProjector(),
        )

        assert np.array_equal(panels.indices, rows.numpy())
        assert np.array_equal(panels.labels, labels)
        assert all(points.shape[0] == rows.shape[0] for _, points in panels.panels)

    def test_the_after_panels_are_those_rows_transported(self, tree: Path) -> None:
        """Not merely the same count: the same rows, after each strategy."""
        test_x, test_y, _ = _test_features(tree)
        ce, guided = _refit(tree, "fm_prelinear_ce"), _refit(tree, "fm_prelinear_guided")
        rows, labels = figures.viz_test_rows(test_y, list(range(len(CLASS_NAMES))), 4)

        panels = stage3.three_way_panels(
            test_x,
            rows,
            labels,
            (("after fm_prelinear_ce", ce.head), ("after fm_prelinear_guided", guided.head)),
            RecordingProjector(),
        )

        raw = test_x[rows].numpy()
        assert np.allclose(panels.panels[0][1], raw[:, :2])
        for (_, points), head in zip(panels.panels[1:], (ce.head, guided.head), strict=True):
            assert np.allclose(points, head.transport(test_x[rows]).numpy()[:, :2])

    def test_the_refit_is_deterministic(self, tree: Path) -> None:
        """ADR-025: no field weights are stored, so a figure showing transported
        features is only reproducible if the fit is."""
        test_x, _, _ = _test_features(tree)
        first, second = _refit(tree, "fm_prelinear_ce"), _refit(tree, "fm_prelinear_ce")
        assert torch.equal(first.head.transport(test_x), second.head.transport(test_x))


class TestColorStability:
    def test_p3_takes_its_colors_from_the_config_mapping(
        self, tree: Path, spec_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same mapping F4 and S3 build from the same config list, so a class
        keeps its color across all three stages."""
        seen: list[dict[str, str]] = []

        def record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            seen.append(kwargs["colors"])
            return kwargs["out"]

        monkeypatch.setattr(figures, "plot_feature_panels", record)
        _generate(tree, spec_path)

        assert seen, "P3 was not drawn"
        assert all(colors == figures.class_colors(CLASS_NAMES) for colors in seen)


class TestDecisionBoundary:
    def test_the_probe_restricted_to_the_plotted_plane_is_the_closed_form(self) -> None:
        """s = W z + b with z = mean_ + u components_ is s = u (W C^T) + (W m + b).
        The panel draws that, so it has to be that and not an approximation."""
        rng = np.random.default_rng(0)
        components = rng.standard_normal((2, DIM))
        mean = rng.standard_normal(DIM)
        weight = rng.standard_normal((len(CLASS_NAMES), DIM))
        bias = rng.standard_normal(len(CLASS_NAMES))

        reduced = figures.reduced_probe(
            LinearRecordingProjector(components, mean), weight, bias
        )

        assert reduced is not None
        weight2, bias2 = reduced
        assert np.allclose(weight2, weight @ components.T)
        assert np.allclose(bias2, weight @ mean + bias)

        u = rng.standard_normal((5, 2))
        z = mean + u @ components
        assert np.allclose(u @ weight2.T + bias2, z @ weight.T + bias)

    def test_a_projection_with_no_linear_preimage_draws_no_boundary(self) -> None:
        """A t-SNE panel has no preimage to substitute into the probe, so the
        boundary is omitted rather than drawn where it does not belong."""
        assert (
            figures.reduced_probe(RecordingProjector(), np.zeros((2, DIM)), np.zeros(2))
            is None
        )


# --------------------------------------------------------------------------
# P2: the marked step is the one the run recorded
# --------------------------------------------------------------------------


class TestTrainingCurves:
    def _drawn(self, tree: Path, spec_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
        captured: list[dict] = []

        def record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            captured.append(kwargs)
            return kwargs["out"]

        monkeypatch.setattr(figures, "plot_selection", record)
        _generate(tree, spec_path)
        assert captured, "P2 was not drawn"
        return captured[0]

    def test_marks_the_step_summary_json_recorded(
        self, tree: Path, spec_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not the validation argmax and not the loss minimum, both of which sit
        elsewhere in the stored curve on purpose."""
        assert SELECTED_STEP != VAL_ARGMAX_STEP and SELECTED_STEP != LOSS_MIN_STEP
        drawn = self._drawn(tree, spec_path, monkeypatch)
        assert [panel["selected_step"] for panel in drawn["panels"]] == [
            SELECTED_STEP for _ in STAGE3_HEADS
        ]

    def test_one_column_per_strategy_named_with_its_step_count(
        self, tree: Path, spec_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = self._drawn(tree, spec_path, monkeypatch)
        assert [panel["name"] for panel in drawn["panels"]] == [
            figures.scheme_label(head, FEATURE_T) for head in STAGE3_HEADS
        ]

    def test_reports_the_loss_minimising_step_apart_from_the_kept_one(self) -> None:
        """Conflating them would say the run stopped where the loss bottomed,
        which is the opposite of what selection on validation accuracy does."""
        note = figures.selection_note(
            steps=list(range(1, N_TRAIN_STEPS + 1)),
            losses=[1.0 / step for step in range(1, N_TRAIN_STEPS + 1)],
            validation=[(2, 0.50), (4, 0.60)],
            selected_step=SELECTED_STEP,
        )
        assert f"step {SELECTED_STEP}" in note
        assert str(LOSS_MIN_STEP) in note
        assert "still falling" in note

    def test_the_columns_do_not_share_a_y_axis(
        self, tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cross-entropy and a velocity residual are not comparable in value.
        Sharing the axis would put the two strategies on one scale and invite
        exactly the comparison the numbers do not support."""
        seen: list[dict] = []
        original = figures.plt.subplots

        def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            seen.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(figures.plt, "subplots", spy)
        figures.plot_selection(
            panels=[
                {
                    "name": name,
                    "steps": [1, 2, 3, 4],
                    "losses": [1e3, 1e2, 1e1, 1e0],
                    "validation": [(2, 0.5), (4, 0.6)],
                    "selected_step": 2,
                }
                for name in ("fm_prelinear_ce T=4", "fm_prelinear_guided T=4")
            ],
            title="t",
            out=tree / "sel.png",
        )
        assert seen, "plot_selection drew nothing"
        assert not any(call.get("sharey") for call in seen)

    def test_says_on_the_figure_why_the_two_losses_are_not_one_scale(
        self, tree: Path, spec_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = self._drawn(tree, spec_path, monkeypatch)
        assert "not comparable" in drawn["caption"]

    def test_a_run_without_a_selection_grid_is_labelled_rather_than_raising(
        self, tree: Path, spec_path: Path
    ) -> None:
        """`eval_every: 0` is the reversal ADR-028 holds open. P2 draws the loss
        row and says the grid is absent."""
        for run_dir in (tree / "results").glob("fm_prelinear_*-T4-10-0"):
            rows = ["step,train_loss,val_top1"]
            rows += [f"{s},{1.0 / s:.6f}," for s in range(1, N_TRAIN_STEPS + 1)]
            (run_dir / "loss_curve.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary.pop("best_epoch")
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )

        written = _generate(tree, spec_path)
        curves = _asset(tree, f"stage3_curves_{ENCODER}.png")
        assert curves in written and curves.stat().st_size > 0


# --------------------------------------------------------------------------
# P4: the stored decomposition, never one recomputed here
# --------------------------------------------------------------------------


class TestDisplacement:
    def test_reads_the_stored_diagnostic_rather_than_recomputing_it(
        self, tree: Path
    ) -> None:
        """ADR-036's number in the note and the number on the figure come from
        one file, so they cannot disagree."""
        run_dir = _find(tree, "fm_prelinear_ce")
        stored = json.loads((run_dir / "rowspace.json").read_text(encoding="utf-8"))
        group = stage3.load_rowspace(run_dir, name="fm_prelinear_ce T=4")

        total = np.asarray(stored["points"]["total_norm"])
        assert np.allclose(
            group["row_fraction"], np.asarray(stored["points"]["row_norm"]) / total
        )
        assert np.allclose(
            group["null_fraction"], np.asarray(stored["points"]["null_norm"]) / total
        )
        assert np.allclose(group["row_norm"], stored["points"]["row_norm"])
        assert np.allclose(
            group["margin_change"],
            np.asarray(stored["points"]["margin_after"])
            - np.asarray(stored["points"]["margin_before"]),
        )

    def test_a_file_whose_components_and_total_disagree_is_refused(self, tree: Path) -> None:
        """The two components are orthogonal, so their norms and the total are a
        right triangle. A file where they are not describes some other
        decomposition, and plotting it would mislabel both axes."""
        run_dir = _find(tree, "fm_prelinear_ce")
        payload = json.loads((run_dir / "rowspace.json").read_text(encoding="utf-8"))
        payload["points"]["null_norm"][3] *= 2.0
        (run_dir / "rowspace.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(stage3.RowspaceMismatchError, match="orthogonal"):
            stage3.load_rowspace(run_dir, name="fm_prelinear_ce T=4")

    def test_a_file_whose_point_arrays_disagree_in_length_is_refused(
        self, tree: Path
    ) -> None:
        run_dir = _find(tree, "fm_prelinear_guided")
        payload = json.loads((run_dir / "rowspace.json").read_text(encoding="utf-8"))
        payload["points"]["margin_after"].pop()
        (run_dir / "rowspace.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(stage3.RowspaceMismatchError, match="margin_after"):
            stage3.load_rowspace(run_dir, name="fm_prelinear_guided T=4")

    def test_an_absent_diagnostic_refuses_by_name(self, tree: Path, spec_path: Path) -> None:
        """P4 never recomputes what the metrics did not see, so it cannot fill
        the gap itself; it names the run and the command that writes the file."""
        for path in (tree / "results").glob("*/rowspace.json"):
            path.unlink()

        with pytest.raises(figures.MissingRunError, match="rowspace"):
            _generate(tree, spec_path)
        assert not (tree / "assets").exists()

    def test_the_panels_carry_one_group_per_strategy(
        self, tree: Path, spec_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[dict] = []

        def record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            captured.append(kwargs)
            return kwargs["out"]

        monkeypatch.setattr(figures, "plot_displacement", record)
        _generate(tree, spec_path)

        assert captured, "P4 was not drawn"
        assert [group["name"] for group in captured[0]["groups"]] == [
            figures.scheme_label(head, FEATURE_T) for head in STAGE3_HEADS
        ]

    def test_a_file_missing_a_column_outright_is_refused(self, tree: Path) -> None:
        run_dir = _find(tree, "fm_prelinear_ce")
        payload = json.loads((run_dir / "rowspace.json").read_text(encoding="utf-8"))
        payload["points"].pop("margin_before")
        (run_dir / "rowspace.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(stage3.RowspaceMismatchError, match="margin_before"):
            stage3.load_rowspace(run_dir, name="fm_prelinear_ce T=4")

    def test_an_example_the_block_did_not_move_is_counted_not_divided_by_zero(
        self, tree: Path, tmp_path: Path
    ) -> None:
        """A zero displacement has no fraction. Dropping those points silently
        would shift the distribution the left panel is there to show."""
        run_dir = _find(tree, "fm_prelinear_ce")
        payload = json.loads((run_dir / "rowspace.json").read_text(encoding="utf-8"))
        for column in ("total_norm", "row_norm", "null_norm"):
            payload["points"][column][0] = 0.0
        (run_dir / "rowspace.json").write_text(json.dumps(payload), encoding="utf-8")

        group = stage3.load_rowspace(run_dir, name="fm_prelinear_ce T=4")
        assert group["undefined"] == 1
        assert group["row_fraction"].shape[0] == len(payload["points"]["total_norm"]) - 1
        assert np.isfinite(group["row_fraction"]).all()

        out = figures.plot_displacement(
            groups=[group], title="t", out=tmp_path / "p4.png"
        )
        assert out.stat().st_size > 0

    def test_an_untrained_field_is_labelled_rather_than_raising(
        self, tree: Path, tmp_path: Path
    ) -> None:
        """Before any training the field is the identity (ADR-030), so the whole
        displacement is zero and no fraction is defined anywhere. P4 says so on
        the panel instead of dividing by zero or refusing to draw."""
        run_dir = _find(tree, "fm_prelinear_ce")
        payload = json.loads((run_dir / "rowspace.json").read_text(encoding="utf-8"))
        n = len(payload["points"]["total_norm"])
        for column in ("total_norm", "row_norm", "null_norm"):
            payload["points"][column] = [0.0] * n
        (run_dir / "rowspace.json").write_text(json.dumps(payload), encoding="utf-8")

        group = stage3.load_rowspace(run_dir, name="fm_prelinear_ce T=4")
        assert group["undefined"] == n
        assert group["row_fraction"].shape[0] == 0

        out = figures.plot_displacement(groups=[group], title="t", out=tmp_path / "p4.png")
        assert out.stat().st_size > 0

    def test_the_scatter_is_subsampled_deterministically(self) -> None:
        """One point per test example is unreadable at the real split size and,
        drawn from an RNG, would not regenerate byte-identically either."""
        first = figures.displacement_scatter_rows(400, 50)
        second = figures.displacement_scatter_rows(400, 50)
        assert np.array_equal(first, second)
        assert first.shape[0] == 50
        assert np.array_equal(figures.displacement_scatter_rows(30, 50), np.arange(30))


# --------------------------------------------------------------------------
# missing runs, generation, reproducibility
# --------------------------------------------------------------------------


class TestMissingRuns:
    def test_an_absent_variant_refuses_by_name_and_writes_nothing(
        self, tree: Path, spec_path: Path
    ) -> None:
        for run in (tree / "results").glob("fm_prelinear_guided-T12-*"):
            shutil.rmtree(run)

        with pytest.raises(figures.MissingRunError, match="fm_prelinear_guided@T12"):
            _generate(tree, spec_path)
        assert not (tree / "assets").exists()

    def test_a_cell_short_of_the_protocol_s_runs_refuses_by_name(
        self, tree: Path, spec_path: Path
    ) -> None:
        """Averaging two runs where the protocol wants three would put a number
        on the panel nobody could reproduce from the stated protocol."""
        shutil.rmtree(tree / "results" / "fm_prelinear_ce-T4-5-2")

        with pytest.raises(IncompleteCellError, match="fm_prelinear_ce@T4"):
            _generate(tree, spec_path)
        assert not (tree / "assets").exists()

    def test_a_cell_missing_a_whole_k_refuses_naming_it(
        self, tree: Path, spec_path: Path
    ) -> None:
        for run in (tree / "results").glob("fm_prelinear_ce-T4-5-*"):
            shutil.rmtree(run)

        with pytest.raises(figures.MissingRunError, match="K=5"):
            _generate(tree, spec_path)
        assert not (tree / "assets").exists()

    def test_an_absent_baseline_refuses_by_name(self, tree: Path, spec_path: Path) -> None:
        """A Stage 3 panel with no probe line is not a Stage 3 panel."""
        for run in (tree / "results").glob("linear_probe-*"):
            shutil.rmtree(run)

        with pytest.raises(figures.MissingRunError, match="linear_probe"):
            _generate(tree, spec_path)
        assert not (tree / "assets").exists()

    def test_an_absent_representative_run_refuses_by_name(
        self, tree: Path, spec_path: Path
    ) -> None:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        spec["stage3"]["representative_subset_seed"] = 7
        spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

        with pytest.raises(figures.MissingRunError, match="subset_seed=7"):
            _generate(tree, spec_path)

    def test_the_refusal_names_stage_3_and_not_stage_2(
        self, tree: Path, spec_path: Path
    ) -> None:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        spec["stage3"]["representative_subset_seed"] = 7
        spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

        with pytest.raises(figures.MissingRunError, match="Stage 3"):
            _generate(tree, spec_path)


class TestGeneration:
    def test_writes_every_family(self, tree: Path, spec_path: Path) -> None:
        written = _generate(tree, spec_path)
        names = {path.name for path in written}
        assert f"stage3_size_curve_{ENCODER}.png" in names
        assert f"stage3_curves_{ENCODER}.png" in names
        assert f"stage3_features_{ENCODER}.png" in names
        assert f"stage3_displacement_{ENCODER}.png" in names
        assert all(path.exists() and path.stat().st_size > 0 for path in written)

    def test_nothing_a_stage_1_or_stage_2_run_wrote_is_overwritten(
        self, tree: Path, spec_path: Path
    ) -> None:
        """P and S and F are separate letters precisely so all three stages'
        assets can sit in one directory."""
        written = _generate(tree, spec_path)
        assert all(path.name.startswith("stage3_") for path in written)

    def test_an_encoder_that_is_not_his_main_cell_marks_nothing(
        self, tree: Path, spec_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """He narrowed Stage 3 to one encoder per dataset. Marking every panel
        would say the Main Comparison is wherever the reader is looking."""
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        spec["stage3"]["main_encoder"] = {DATASET: "some_other_encoder"}
        spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

        captured: list[dict] = []

        def record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            captured.append(kwargs)
            return kwargs["out"]

        monkeypatch.setattr(figures, "plot_stage3_size_curve", record)
        _generate(tree, spec_path)
        assert captured[0]["marked_k"] is None

    def test_a_marked_cell_outside_the_protocol_s_k_values_is_refused(
        self, tree: Path
    ) -> None:
        cells = stage3.load_stage3_cells(tree / "results")
        with pytest.raises(ValueError, match="marked cell"):
            figures.plot_stage3_size_curve(
                cells=cells,
                dataset=DATASET,
                encoder=ENCODER,
                lines=figures.stage3_lines(T_VALUES),
                marked_k="7",
                out=tree / "bad.png",
            )

    def test_regeneration_is_byte_identical(self, tree: Path, spec_path: Path) -> None:
        written = _generate(tree, spec_path)
        before = {path: path.read_bytes() for path in written}
        _generate(tree, spec_path)
        assert all(path.read_bytes() == before[path] for path in written)


class TestCells:
    def test_the_grid_is_keyed_by_head_and_variant(self, tree: Path) -> None:
        """`report.load_cells` already carries the variant, so Stage 3 needs no
        second aggregation the way Stage 2 did: T=4 and T=12 are two cells
        because they are two rows of TABLE.md."""
        cells = stage3.load_stage3_cells(tree / "results")
        heads = {cell.head for cell in cells}
        assert heads == set(figures.stage3_lines(T_VALUES))
        assert "prototype" not in heads
