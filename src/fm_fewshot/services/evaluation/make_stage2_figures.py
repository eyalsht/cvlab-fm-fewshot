"""Drive the Stage 2 figure families S1 to S4 (PRD_stage2_figures, ADR-025).

S1 accuracy against K for the write-up's five configurations, S2 the two
schemes' training curves, S3 the three-way feature comparison, S4 the flow
trajectories. They serve items 1 to 4 of the write-up's "Results to present".

Reads results/ and the feature caches, writes assets/<dataset>/. Nothing comes
from notebook state: no field weights are stored, so any figure needing
transported features refits the run deterministically from its stored
config.yaml, which the seed alone reproduces.

Two rules the write-up states outright are enforced here rather than left to
the caller. The projection is fitted once, jointly, over every compared feature
set and the prototypes, because fitting per panel produces unrelated pictures.
And T is carried in the cell key, because report groups by head alone and a
Stage 2 panel built on that grouping would average T=4 and T=12 into one line.

Validation runs to completion before anything is drawn. A missing run has to
refuse by name and leave no half-written assets directory behind.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from fm_fewshot.services.data.subsets import balanced_subset
from fm_fewshot.services.evaluation import figures
from fm_fewshot.services.evaluation.metrics import aggregate_cell
from fm_fewshot.services.evaluation.report import expected_runs
from fm_fewshot.services.features.cache import read_features
from fm_fewshot.services.heads import make_head
from fm_fewshot.shared.config import load_config
from fm_fewshot.shared.contracts import CellSummary, ExperimentConfig

STAGE2_HEADS = figures.STAGE2_HEADS


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------


def scheme_labels(t_values) -> list[str]:
    """The write-up's five configurations, in the order a panel lists them."""
    labels = ["prototype"]
    for head in STAGE2_HEADS:
        labels += [figures.scheme_label(head, int(t)) for t in t_values]
    return labels


def _sample_steps(cfg: dict) -> int:
    return int((cfg.get("head_params") or {}).get("sample_steps", 4))


def load_stage2_cells(results_dir: Path, *, t_values=(4, 12)) -> list[CellSummary]:
    """Aggregate the Stage 2 grid, keyed by scheme rather than by head.

    Stage 1 heads sharing the results directory are skipped: the Stage 2 panels
    are defined by the write-up's five configurations, and a linear probe is
    not one of them.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    wanted = {int(t) for t in t_values}

    grouped: dict[tuple, list[tuple[str, float]]] = {}
    for run_dir in sorted(results_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        if not run_dir.is_dir() or not summary_path.exists():
            continue  # half-written or unrelated directory
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        head = cfg["head"]
        if head == "prototype":
            label = "prototype"
        elif head in STAGE2_HEADS:
            steps = _sample_steps(cfg)
            if steps not in wanted:
                continue
            label = figures.scheme_label(head, steps)
        else:
            continue
        key = (cfg["dataset"], cfg["encoder"], label, cfg["k"])
        grouped.setdefault(key, []).append((payload["run_id"], payload["test_top1"]))

    cells: list[CellSummary] = []
    for key in sorted(grouped, key=lambda k: (k[0], k[1], k[2], figures.k_label(k[3]))):
        dataset, encoder, label, k = key
        runs = sorted(grouped[key])
        cells.append(
            aggregate_cell(
                dataset=dataset,
                encoder=encoder,
                head=label,
                k=k,
                run_ids=tuple(run_id for run_id, _ in runs),
                accuracies=[accuracy for _, accuracy in runs],
            )
        )
    return cells


def require_schemes(cells: list[CellSummary], dataset: str, encoder: str, t_values) -> None:
    """Fail naming the absent configuration rather than plotting a partial panel."""
    present = [c for c in cells if c.dataset == dataset and c.encoder == encoder]
    for label in scheme_labels(t_values):
        matching = [c for c in present if c.head == label]
        if not matching:
            raise figures.MissingRunError(
                f"no runs for ({dataset}, {encoder}, {label}); the Stage 2 figure "
                "would be partial, so nothing was written"
            )
        covered = {figures.k_label(c.k) for c in matching}
        absent = [k for k in figures.K_ORDER if k not in covered]
        if absent:
            raise figures.MissingRunError(
                f"({dataset}, {encoder}, {label}) has no runs at K={', '.join(absent)}; "
                "the Stage 2 figure would be partial, so nothing was written"
            )
        for cell in matching:
            wanted = expected_runs(cell.head, cell.k)
            if cell.n_runs != wanted:
                raise figures.MissingRunError(
                    f"({dataset}, {encoder}, {label}, K={figures.k_label(cell.k)}) holds "
                    f"{cell.n_runs} of the {wanted} runs the protocol requires; the "
                    "Stage 2 figure would be partial, so nothing was written"
                )


# --------------------------------------------------------------------------
# runs and refits
# --------------------------------------------------------------------------


def find_run(
    results_dir: Path,
    *,
    dataset: str,
    encoder: str,
    head: str,
    k: int | None,
    subset_seed: int,
    sample_steps: int | None = None,
) -> Path:
    """The stored run directory for one grid point, or a refusal naming it.

    sample_steps lives inside head_params rather than at the top of the config,
    so it cannot be matched by the plain key comparison the Stage 1 finder uses.
    """
    results_dir = Path(results_dir)
    if results_dir.exists():
        for run_dir in sorted(results_dir.iterdir()):
            summary_path = run_dir / "summary.json"
            if not run_dir.is_dir() or not summary_path.exists():
                continue
            cfg = json.loads(summary_path.read_text(encoding="utf-8"))["config"]
            if (cfg["dataset"], cfg["encoder"], cfg["head"]) != (dataset, encoder, head):
                continue
            if cfg["k"] != k or cfg["subset_seed"] != subset_seed:
                continue
            if sample_steps is not None and _sample_steps(cfg) != sample_steps:
                continue
            return run_dir
    steps = "" if sample_steps is None else f", sample_steps={sample_steps}"
    raise figures.MissingRunError(
        f"no stored run for dataset={dataset}, encoder={encoder}, head={head}{steps}, "
        f"K={figures.k_label(k)}, subset_seed={subset_seed} under {results_dir}; the "
        "Stage 2 figure would be partial, so nothing was written"
    )


@dataclass(frozen=True)
class Refit:
    """A run's head, rebuilt from its stored config (ADR-025)."""

    run_dir: Path
    config: ExperimentConfig
    head: object


def refit(run_dir: Path, *, data_root: Path) -> Refit:
    """Reproduce a stored run's fitted head from its config alone.

    No field weights are written by the loop, so a figure showing transported
    features or trajectories is only reproducible if this is. It mirrors the
    evaluation loop's fit exactly: same subset draw, same seeds, same head.
    """
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    train_x, train_y, meta = read_features(
        cfg.dataset, "train", cfg.encoder, data_root=data_root, l2_normalize=cfg.l2_normalize
    )
    n_classes = len(meta["class_names"])
    val_x, val_y, _ = read_features(
        cfg.dataset, "val", cfg.encoder, data_root=data_root, l2_normalize=cfg.l2_normalize
    )
    subset = balanced_subset(train_y, cfg.k, cfg.subset_seed, n_classes, cfg.dataset)
    head = make_head(cfg, n_classes)
    head.fit(train_x[subset.idx], subset.labels, val_x, val_y)
    return Refit(run_dir=run_dir, config=cfg, head=head)


# --------------------------------------------------------------------------
# joint projection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectedPanels:
    """Panels sharing one projection, plus the rows they were built from."""

    panels: tuple[tuple[str, np.ndarray], ...]
    labels: np.ndarray
    prototypes: np.ndarray
    indices: np.ndarray


def _as_array(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _project_blocks(blocks, prototypes, projector):  # noqa: ANN202 - arrays and an estimator
    """One fit over every block and the prototypes, then split back.

    The write-up: "Compute the projection jointly over the feature sets and
    prototypes being compared so that the different views correspond to the
    same low-dimensional representation." Transforming a set that did not help
    define the space puts it in coordinates it never earned.
    """
    flat = [block.reshape(-1, block.shape[-1]) for block in blocks]
    protos = _as_array(prototypes)
    embedded = np.asarray(projector.fit_transform(np.concatenate([*flat, protos], axis=0)))

    split, start = [], 0
    for block, rows in zip(blocks, flat, strict=True):
        stop = start + rows.shape[0]
        split.append(embedded[start:stop].reshape(*block.shape[:-1], embedded.shape[1]))
        start = stop
    return split, embedded[start:]


def three_way_panels(test_x, rows, labels, schemes, prototypes, projector) -> ProjectedPanels:
    """S3. The original test features and the same rows after each scheme."""
    original = test_x[rows]
    names = ["original features"]
    blocks = [_as_array(original)]
    for name, head in schemes:
        names.append(name)
        blocks.append(_as_array(head.transport(original)))
    projected, protos = _project_blocks(blocks, prototypes, projector)
    return ProjectedPanels(
        panels=tuple(zip(names, projected, strict=True)),
        labels=np.asarray(labels),
        prototypes=protos,
        indices=_as_array(rows),
    )


def trajectory_panels(test_x, rows, labels, schemes, prototypes, projector) -> ProjectedPanels:
    """S4. Every Euler state of the same rows, one panel per scheme.

    Each head integrates at its own T, so the T=4 and T=12 panels hold
    different numbers of states; they only contrast because the single fit
    covers both.
    """
    original = test_x[rows]
    names, blocks = [], []
    for name, head in schemes:
        names.append(name)
        blocks.append(_as_array(head.trajectory(original)))
    projected, protos = _project_blocks(blocks, prototypes, projector)
    return ProjectedPanels(
        panels=tuple(zip(names, projected, strict=True)),
        labels=np.asarray(labels),
        prototypes=protos,
        indices=_as_array(rows),
    )


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
    runs: dict[tuple[str, int], Path]
    test_x: torch.Tensor
    test_y: torch.Tensor


def _read_loss_curve(path: Path) -> tuple[list[int], list[float]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [int(r["step"]) for r in rows], [float(r["train_loss"]) for r in rows]


def _read_selection(run_dir: Path) -> dict:
    """Everything S7 draws for one run, straight from what the run stored.

    `best_epoch` comes from `summary.json` rather than from the validation
    column's argmax so the marked step is the step the head actually restored,
    including its tie rule. Recomputing it here would let the figure and the
    table disagree without either being obviously wrong.
    """
    run_dir = Path(run_dir)
    steps, losses = _read_loss_curve(run_dir / "loss_curve.csv")
    with (run_dir / "loss_curve.csv").open(encoding="utf-8") as handle:
        validation = [
            (int(r["step"]), float(r["val_top1"]))
            for r in csv.DictReader(handle)
            if r.get("val_top1")
        ]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "steps": steps,
        "losses": losses,
        "validation": validation,
        "selected_step": summary.get("best_epoch"),
    }


def _needed_steps(head: str, feature_t: int, t_values) -> list[int]:
    """Which T each scheme has to be refit at.

    Standard training also carries the T=4 against T=12 contrast, which is
    where the discretization story is visible; rolled-out training is only
    needed at the representative T.
    """
    if head == "fm_standard":
        return sorted({feature_t, *(int(t) for t in t_values)})
    return [feature_t]


def _plan(spec, results_dir: Path, data_root: Path, cells: list[CellSummary]) -> list[_Plan]:
    stage2 = spec["stage2"]
    t_values = tuple(int(t) for t in stage2["t_values"])
    feature_t = int(stage2["feature_t"])
    representative_k = stage2["representative_k"]
    representative_k = None if representative_k == "full" else int(representative_k)
    subset_seed = int(stage2["representative_subset_seed"])

    encoders: dict[str, list[str]] = {}
    for cell in cells:
        encoders.setdefault(cell.dataset, [])
        if cell.encoder not in encoders[cell.dataset]:
            encoders[cell.dataset].append(cell.encoder)

    plans: list[_Plan] = []
    for dataset, dataset_spec in sorted(spec["datasets"].items()):
        if dataset not in encoders:
            raise figures.MissingRunError(
                f"no stored Stage 2 runs for dataset {dataset!r} under {results_dir}; "
                "run the sweep before generating figures"
            )
        viz_classes = list(dataset_spec["viz_classes"])
        colors = figures.class_colors(viz_classes)
        for encoder in encoders[dataset]:
            require_schemes(cells, dataset, encoder, t_values)
            test_x, test_y, meta = read_features(
                dataset, "test", encoder, data_root=data_root, l2_normalize=False
            )
            class_names = list(meta["class_names"])
            figures.validate_viz_classes(viz_classes, class_names)
            runs: dict[tuple[str, int], Path] = {}
            for head in STAGE2_HEADS:
                for t in _needed_steps(head, feature_t, t_values):
                    run_dir = find_run(
                        results_dir, dataset=dataset, encoder=encoder, head=head,
                        sample_steps=t, k=representative_k, subset_seed=subset_seed,
                    )
                    if t == feature_t and not (run_dir / "loss_curve.csv").exists():
                        raise figures.MissingRunError(
                            f"run {run_dir.name} has no loss_curve.csv, so the "
                            f"{figures.scheme_label(head, t)} training curve cannot be "
                            "drawn; nothing was written"
                        )
                    runs[(head, t)] = run_dir
            plans.append(
                _Plan(
                    dataset=dataset,
                    encoder=encoder,
                    viz_classes=viz_classes,
                    viz_ids=[class_names.index(name) for name in viz_classes],
                    colors=colors,
                    runs=runs,
                    test_x=test_x,
                    test_y=test_y,
                )
            )
    return plans


def make_stage2_figures(
    figure_config: Path,
    *,
    results_dir: Path = Path("results"),
    data_root: Path = Path("data"),
    assets_dir: Path = Path("assets"),
) -> list[Path]:
    spec = yaml.safe_load(Path(figure_config).read_text(encoding="utf-8"))
    stage2 = spec["stage2"]
    t_values = tuple(int(t) for t in stage2["t_values"])
    feature_t = int(stage2["feature_t"])
    representative_k = stage2["representative_k"]
    representative_k = None if representative_k == "full" else int(representative_k)
    dpi = int(spec["dpi"])
    projection, viz_seed = spec["projection"], int(spec["viz_seed"])

    cells = load_stage2_cells(results_dir, t_values=t_values)
    # Everything is resolved and checked before a single file is written.
    plans = _plan(spec, Path(results_dir), Path(data_root), cells)

    refits: dict[Path, Refit] = {}

    def head_of(run_dir: Path):  # noqa: ANN202 - a fitted FmHead
        if run_dir not in refits:
            refits[run_dir] = refit(run_dir, data_root=data_root)
        return refits[run_dir].head

    k_tag = figures.k_label(representative_k)
    written: list[Path] = []
    for plan in plans:
        out_dir = Path(assets_dir) / plan.dataset
        encoder, stem = plan.encoder, f"{plan.dataset} / {plan.encoder}"

        # S1, accuracy against K for the write-up's five configurations.
        written.append(
            figures.plot_size_curve(
                cells, plan.dataset, encoder,
                out_dir / f"stage2_size_curve_{encoder}.png", dpi=dpi,
            )
        )
        written.append(
            figures.write_series_csv(
                cells, plan.dataset, encoder, out_dir / f"stage2_size_curve_{encoder}.csv"
            )
        )

        # S2, the two schemes' training curves, read from the stored run.
        curves = [
            (figures.scheme_label(head, feature_t),
             *_read_loss_curve(plan.runs[(head, feature_t)] / "loss_curve.csv"))
            for head in STAGE2_HEADS
        ]
        written.append(
            figures.plot_stage2_loss_curves(
                curves=curves,
                title=f"{stem}, K={k_tag}",
                out=out_dir / f"stage2_loss_curves_{encoder}.png",
                dpi=dpi,
            )
        )

        # S7, why each run stopped where it did. Same runs as S2, no refit.
        written.append(
            figures.plot_selection(
                panels=[
                    {"name": figures.scheme_label(head, feature_t),
                     **_read_selection(plan.runs[(head, feature_t)])}
                    for head in STAGE2_HEADS
                ],
                title=f"{stem}, K={k_tag}",
                out=out_dir / f"stage2_selection_{encoder}.png",
                dpi=dpi,
            )
        )

        # S3, the three-way feature comparison on one joint projection.
        standard = head_of(plan.runs[("fm_standard", feature_t)])
        rolled = head_of(plan.runs[("fm_rolled", feature_t)])
        rows, labels = figures.viz_test_rows(
            plan.test_y, plan.viz_ids, int(spec["max_per_class"])
        )
        panels = three_way_panels(
            plan.test_x, rows, labels,
            ((f"after fm_standard, T={feature_t}", standard),
             (f"after fm_rolled, T={feature_t}", rolled)),
            standard.prototypes[plan.viz_ids],
            figures.make_projector(projection, viz_seed),
        )
        written.append(
            figures.plot_feature_panels(
                panels=panels.panels,
                labels=panels.labels,
                prototypes=panels.prototypes,
                class_names=plan.viz_classes,
                colors=plan.colors,
                title=f"{stem}, K={k_tag} ({projection.upper()}, one joint fit)",
                out=out_dir / f"stage2_features_{encoder}.png",
                dpi=dpi,
            )
        )

        # S4, trajectories: one figure per scheme, then the T contrast.
        n_classes = int(stage2["trajectory_classes"])
        traj_ids = plan.viz_ids[:n_classes]
        traj_names = plan.viz_classes[:n_classes]
        traj_rows, traj_labels = figures.viz_test_rows(
            plan.test_y, traj_ids, int(stage2["trajectory_per_class"])
        )

        def _trajectories(schemes, out_name, title, plan=plan, ids=traj_ids,
                          names=traj_names, rows=traj_rows, labels=traj_labels):
            drawn = trajectory_panels(
                plan.test_x, rows, labels, schemes,
                schemes[0][1].prototypes[ids],
                figures.make_projector(projection, viz_seed),
            )
            return figures.plot_trajectory_panels(
                panels=drawn.panels,
                labels=drawn.labels,
                prototypes=drawn.prototypes,
                class_names=names,
                colors=plan.colors,
                title=title,
                out=Path(assets_dir) / plan.dataset / out_name,
                dpi=dpi,
            )

        for head, fitted in (("fm_standard", standard), ("fm_rolled", rolled)):
            label = figures.scheme_label(head, feature_t)
            written.append(
                _trajectories(
                    ((label, fitted),),
                    f"stage2_trajectories_{encoder}_{head}_T{feature_t}.png",
                    f"{stem}, K={k_tag}, {label}",
                )
            )

        contrast = tuple(
            (figures.scheme_label("fm_standard", t), head_of(plan.runs[("fm_standard", t)]))
            for t in t_values
        )
        tags = "_vs_".join(f"T{t}" for t in t_values)
        written.append(
            _trajectories(
                contrast,
                f"stage2_trajectories_{encoder}_fm_standard_{tags}.png",
                f"{stem}, K={k_tag}, fm_standard, {tags.replace('_vs_', ' against ')}",
            )
        )

    return written


__all__ = [
    "ProjectedPanels",
    "Refit",
    "find_run",
    "load_stage2_cells",
    "make_stage2_figures",
    "refit",
    "require_schemes",
    "scheme_labels",
    "three_way_panels",
    "trajectory_panels",
]
