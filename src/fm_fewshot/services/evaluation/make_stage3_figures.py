"""Drive the Stage 3 figure families P1 to P4 (PRD_stage3_figures, ADR-025, ADR-036).

P1 accuracy against K for the probe and both strategies, P2 the training and
validation curves at the representative setting, P3 the before-and-after
features on one joint projection, P4 the row(W) and null(W) decomposition of the
displacement. They serve the three things the write-up asks to be prepared to
present, plus the diagnostic ADR-036 adds.

Reads results/ and the feature caches, writes assets/<dataset>/. As in Stage 2,
no field weights are stored, so any figure needing transported features refits
the run deterministically from its stored config.yaml, through the same `refit`
Stage 2 uses rather than a second implementation of it.

Two rules run through the family. The baseline is the linear probe, taken from
the registry (ADR-035): Stage 1's F1 and Stage 2's S1 lead with the image
prototype row, and a Stage 3 panel that did so would put the reader's comparison
on a number Stage 3 is not measured on. And the displacement decomposition is read
from the run's stored `rowspace.json`, never recomputed here, for the reason S5
reads `reverse.json`: the number in the phase note and the number on the figure
have to come from one file.

Validation runs to completion before anything is drawn. A missing run, or a
`rowspace.json` that does not describe the split it claims to, has to refuse by
name and leave no half-written assets directory behind.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from fm_fewshot.services.evaluation import figures
from fm_fewshot.services.evaluation import make_stage2_figures as stage2
from fm_fewshot.services.evaluation.report import expected_runs, load_cells
from fm_fewshot.services.features.cache import read_features
from fm_fewshot.shared.contracts import CellSummary

STAGE3_HEADS = figures.STAGE3_HEADS

# ADR-025 again, and the same implementation: a second refit path would be a
# second answer to "what field produced this picture".
Refit = stage2.Refit
refit = stage2.refit

ROWSPACE_COLUMNS = ("total_norm", "row_norm", "null_norm", "margin_before", "margin_after")


class RowspaceMismatchError(ValueError):
    """Raised when a stored rowspace.json does not describe an orthogonal split."""


# --------------------------------------------------------------------------
# cells and runs
# --------------------------------------------------------------------------


def load_stage3_cells(results_dir: Path, *, t_values=(4, 12)) -> list[CellSummary]:
    """The Stage 3 lines out of `report.load_cells`, and nothing else.

    Unlike Stage 2 this needs no aggregation of its own. T is a config variant
    here, so `method_label` already separates T=4 from T=12 and the panel reads
    exactly the cells TABLE.md reports.
    """
    wanted = set(figures.stage3_lines(t_values))
    return [cell for cell in load_cells(results_dir) if cell.head in wanted]


def require_stage3_cells(
    cells: list[CellSummary], dataset: str, encoder: str, t_values
) -> None:
    """Fail naming the absent line rather than plotting a partial panel."""
    present = [c for c in cells if c.dataset == dataset and c.encoder == encoder]
    for label in figures.stage3_lines(t_values):
        matching = [c for c in present if c.head == label]
        if not matching:
            raise figures.MissingRunError(
                f"no runs for ({dataset}, {encoder}, {label}); the Stage 3 figure "
                "would be partial, so nothing was written"
            )
        covered = {figures.k_label(c.k) for c in matching}
        absent = [k for k in figures.K_ORDER if k not in covered]
        if absent:
            raise figures.MissingRunError(
                f"({dataset}, {encoder}, {label}) has no runs at K={', '.join(absent)}; "
                "the Stage 3 figure would be partial, so nothing was written"
            )
        for cell in matching:
            wanted = expected_runs(cell.head, cell.k)
            if cell.n_runs != wanted:
                raise figures.MissingRunError(
                    f"({dataset}, {encoder}, {label}, K={figures.k_label(cell.k)}) holds "
                    f"{cell.n_runs} of the {wanted} runs the protocol requires; the "
                    "Stage 3 figure would be partial, so nothing was written"
                )


def find_run(results_dir: Path, **match) -> Path:
    """The stored run directory for one Stage 3 grid point, or a refusal naming it."""
    return stage2.find_run(results_dir, stage=3, **match)


# --------------------------------------------------------------------------
# P3, the joint projection
# --------------------------------------------------------------------------


def three_way_panels(test_x, rows, labels, schemes, projector) -> stage2.ProjectedPanels:
    """P3. The original test features and the same rows after each strategy.

    No prototypes, in the fit or on the panel. Stage 2 transported toward the
    image prototypes and S3 drew them; Stage 3 transports toward nothing, and a
    star here would suggest a target that does not exist.
    """
    original = test_x[rows]
    names = ["original features"]
    blocks = [figures.as_array(original)]
    for name, head in schemes:
        names.append(name)
        blocks.append(figures.as_array(head.transport(original)))
    projected, _ = figures.project_blocks(blocks, projector)
    return stage2.ProjectedPanels(
        panels=tuple(zip(names, projected, strict=True)),
        labels=np.asarray(labels),
        prototypes=None,
        indices=figures.as_array(rows),
    )


# --------------------------------------------------------------------------
# P4, the stored decomposition
# --------------------------------------------------------------------------


def load_rowspace(run_dir: Path, *, name: str) -> dict:
    """One P4 group from the run's stored ADR-036 diagnostic.

    The figure never recomputes the decomposition, for the reason S5 never
    recomputes the reverse metrics. What it does check is that the file
    describes the split it claims to: the two components are orthogonal, so
    their norms and the total are a right triangle, and a file where they are
    not describes some other decomposition with both of P4's axes mislabelled.

    A displacement of exactly zero has no defined fraction. Those examples are
    counted and reported on the panel rather than divided by zero or quietly
    dropped.
    """
    run_dir = Path(run_dir)
    path = run_dir / "rowspace.json"
    if not path.exists():
        raise figures.MissingRunError(
            f"run {run_dir.name} has no rowspace.json, so P4 cannot be drawn; run "
            f"`rowspace --config {run_dir / 'config.yaml'}` first, because no figure "
            "recomputes a diagnostic the metrics did not see. Nothing was written"
        )
    points = json.loads(path.read_text(encoding="utf-8"))["points"]

    columns: dict[str, np.ndarray] = {}
    for key in ROWSPACE_COLUMNS:
        if key not in points:
            raise RowspaceMismatchError(f"{path} has no {key} column, so P4 cannot be drawn")
        columns[key] = np.asarray(points[key], dtype=np.float64)
    length = columns["total_norm"].shape[0]
    for key, values in columns.items():
        if values.shape[0] != length:
            raise RowspaceMismatchError(
                f"{path} holds {length} points but {values.shape[0]} {key} values; the "
                "columns do not describe one set of test examples"
            )

    total = columns["total_norm"]
    composed = np.hypot(columns["row_norm"], columns["null_norm"])
    if not np.allclose(composed, total, rtol=1e-4, atol=1e-6):
        worst = int(np.argmax(np.abs(composed - total)))
        raise RowspaceMismatchError(
            f"{path} point {worst}: row {columns['row_norm'][worst]:.6g} and null "
            f"{columns['null_norm'][worst]:.6g} compose {composed[worst]:.6g}, not the "
            f"stored total {total[worst]:.6g}. The two components are orthogonal by "
            "construction, so this file describes some other split and P4 would "
            "mislabel both of its axes"
        )

    moved = total > 0.0
    return {
        "name": name,
        "row_fraction": columns["row_norm"][moved] / total[moved],
        "null_fraction": columns["null_norm"][moved] / total[moved],
        "row_norm": columns["row_norm"],
        "margin_change": columns["margin_after"] - columns["margin_before"],
        "undefined": int((~moved).sum()),
    }


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Plan:
    """Everything one (dataset, encoder) figure set needs, resolved up front."""

    dataset: str
    encoder: str
    viz_classes: list[str]
    viz_ids: list[int]
    colors: dict[str, str]
    runs: dict[str, Path]
    # The Stage 1 probe run for this cell. The block starts as the identity, so
    # this run's kept validation top-1 is where every Stage 3 curve begins.
    probe_run: Path
    test_x: torch.Tensor
    test_y: torch.Tensor
    # The K whose cell is the Main Comparison here, or None on an encoder that
    # is not the one he narrowed this dataset to.
    marked_k: str | None


def _settings(spec: dict) -> dict:
    stage3 = spec["stage3"]
    representative_k = stage3["representative_k"]
    return {
        "t_values": tuple(int(t) for t in stage3["t_values"]),
        "feature_t": int(stage3["feature_t"]),
        "k": None if representative_k == "full" else int(representative_k),
        "subset_seed": int(stage3["representative_subset_seed"]),
        "main_encoder": dict(stage3.get("main_encoder") or {}),
    }


def _plan(spec: dict, results_dir: Path, data_root: Path, cells: list[CellSummary]) -> list[_Plan]:
    settings = _settings(spec)

    encoders: dict[str, list[str]] = {}
    for cell in cells:
        encoders.setdefault(cell.dataset, [])
        if cell.encoder not in encoders[cell.dataset]:
            encoders[cell.dataset].append(cell.encoder)

    plans: list[_Plan] = []
    for dataset, dataset_spec in sorted(spec["datasets"].items()):
        if dataset not in encoders:
            raise figures.MissingRunError(
                f"no stored Stage 3 runs for dataset {dataset!r} under {results_dir}; "
                "run the sweep before generating figures"
            )
        viz_classes = list(dataset_spec["viz_classes"])
        colors = figures.class_colors(viz_classes)
        for encoder in encoders[dataset]:
            require_stage3_cells(cells, dataset, encoder, settings["t_values"])
            test_x, test_y, meta = read_features(
                dataset, "test", encoder, data_root=data_root, l2_normalize=False
            )
            class_names = list(meta["class_names"])
            figures.validate_viz_classes(viz_classes, class_names)

            runs: dict[str, Path] = {}
            for head in STAGE3_HEADS:
                run_dir = find_run(
                    results_dir, dataset=dataset, encoder=encoder, head=head,
                    sample_steps=settings["feature_t"], k=settings["k"],
                    subset_seed=settings["subset_seed"],
                )
                if not (run_dir / "loss_curve.csv").exists():
                    raise figures.MissingRunError(
                        f"run {run_dir.name} has no loss_curve.csv, so the "
                        f"{figures.scheme_label(head, settings['feature_t'])} training "
                        "curve cannot be drawn; nothing was written"
                    )
                # Read now, so a broken or absent diagnostic refuses before the
                # first file is written rather than half way through the set.
                load_rowspace(run_dir, name=figures.scheme_label(head, settings["feature_t"]))
                runs[head] = run_dir

            # Resolved here rather than at draw time, with everything else, so a
            # cell whose probe run is missing refuses before any file is written.
            probe_run = find_run(
                results_dir, dataset=dataset, encoder=encoder, head="linear_probe",
                k=settings["k"], subset_seed=settings["subset_seed"],
            )
            figures.read_probe_validation(probe_run)

            plans.append(
                _Plan(
                    dataset=dataset,
                    encoder=encoder,
                    viz_classes=viz_classes,
                    viz_ids=[class_names.index(name) for name in viz_classes],
                    colors=colors,
                    runs=runs,
                    probe_run=probe_run,
                    test_x=test_x,
                    test_y=test_y,
                    marked_k=(
                        figures.k_label(settings["k"])
                        if settings["main_encoder"].get(dataset) == encoder
                        else None
                    ),
                )
            )
    return plans


def make_stage3_figures(
    figure_config: Path,
    *,
    results_dir: Path = Path("results"),
    data_root: Path = Path("data"),
    assets_dir: Path = Path("assets"),
) -> list[Path]:
    spec = yaml.safe_load(Path(figure_config).read_text(encoding="utf-8"))
    settings = _settings(spec)
    feature_t = settings["feature_t"]
    dpi = int(spec["dpi"])
    kinds, viz_seed = figures.projection_kinds(spec), int(spec["viz_seed"])

    cells = load_stage3_cells(results_dir, t_values=settings["t_values"])
    # Everything is resolved and checked before a single file is written.
    plans = _plan(spec, Path(results_dir), Path(data_root), cells)

    refits: dict[Path, Refit] = {}

    def head_of(run_dir: Path):  # noqa: ANN202 - a fitted FmPreLinearHead
        if run_dir not in refits:
            refits[run_dir] = refit(run_dir, data_root=data_root)
        return refits[run_dir].head

    lines = figures.stage3_lines(settings["t_values"])
    k_tag = figures.k_label(settings["k"])
    written: list[Path] = []
    for plan in plans:
        out_dir = Path(assets_dir) / plan.dataset
        encoder, stem = plan.encoder, f"{plan.dataset} / {plan.encoder}"
        names = [figures.scheme_label(head, feature_t) for head in STAGE3_HEADS]

        # P1, accuracy against K against the probe, with his cell marked.
        written.append(
            figures.plot_stage3_size_curve(
                cells=cells, dataset=plan.dataset, encoder=encoder, lines=lines,
                marked_k=plan.marked_k,
                out=out_dir / f"stage3_size_curve_{encoder}.png", dpi=dpi,
            )
        )
        written.append(
            figures.write_series_csv(
                cells, plan.dataset, encoder,
                out_dir / f"stage3_size_curve_{encoder}.csv", order=lines,
            )
        )

        # P2, training loss and the validation grid, read from the stored run.
        written.append(
            figures.plot_selection(
                panels=[
                    {"name": name, **figures.read_selection(plan.runs[head])}
                    for name, head in zip(names, STAGE3_HEADS, strict=True)
                ],
                title=f"{stem}, K={k_tag}",
                caption=figures.STAGE3_CURVES_CAPTION,
                reference={
                    "value": figures.read_probe_validation(plan.probe_run),
                    "label": "linear probe, the identity start",
                },
                out=out_dir / f"stage3_curves_{encoder}.png", dpi=dpi,
            )
        )

        # P3, the three-way feature comparison on one joint projection.
        heads = [head_of(plan.runs[head]) for head in STAGE3_HEADS]
        rows, labels = figures.viz_test_rows(
            plan.test_y, plan.viz_ids, int(spec["max_per_class"])
        )
        # "PCA or t-SNE may be used": one figure per projection, PCA keeping the
        # bare filename. No unit-sphere twin here, unlike S3. Stage 2 classifies
        # by cosine, so normalizing shows the geometry the rule reads; the frozen
        # probe is affine and its logits move with scale, so a normalized panel
        # would draw a decision this classifier never makes.
        for kind in kinds:
            projector = figures.make_projector(kind, viz_seed)
            panels = three_way_panels(
                plan.test_x, rows, labels,
                tuple((f"after {name}", head) for name, head in zip(names, heads, strict=True)),
                projector,
            )
            # Either head's probe: both are the Stage 1 probe refitted on the same
            # subset at the same init_seed, which makes them bit-identical.
            reduced = figures.reduced_probe(
                projector, heads[0].probe.weight, heads[0].probe.bias
            )
            written.append(
                figures.plot_feature_panels(
                    panels=panels.panels,
                    labels=panels.labels,
                    prototypes=None,
                    boundary=None if reduced is None else (*reduced, plan.viz_ids),
                    class_names=plan.viz_classes,
                    colors=plan.colors,
                    note=figures.variance_note(projector, kind),
                    title=f"{stem}, K={k_tag} ({kind.upper()}, one joint fit)",
                    out=out_dir / f"stage3_features_{encoder}{figures.projection_suffix(kind)}.png",
                    dpi=dpi,
                )
            )

        # P4, where the displacement went, from what the rowspace command stored.
        written.append(
            figures.plot_displacement(
                groups=[
                    load_rowspace(plan.runs[head], name=name)
                    for name, head in zip(names, STAGE3_HEADS, strict=True)
                ],
                title=f"{stem}, K={k_tag}",
                out=out_dir / f"stage3_displacement_{encoder}.png",
                dpi=dpi,
            )
        )

    return written


__all__ = [
    "Refit",
    "RowspaceMismatchError",
    "find_run",
    "load_rowspace",
    "load_stage3_cells",
    "make_stage3_figures",
    "refit",
    "require_stage3_cells",
    "three_way_panels",
]
