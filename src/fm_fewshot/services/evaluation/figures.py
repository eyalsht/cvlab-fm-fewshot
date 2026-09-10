"""The four Stage 1 figure families per PRD_figures (ADR-015).

F1 accuracy against training-set size, F2 linear-probe loss curves, F3
row-normalized confusion matrices, F4 joint-projection feature visualizations.

Everything regenerates from stored results and the feature caches, never from
notebook state, because the write-up makes these graded deliverables.

Three rules from the write-up are load-bearing and are enforced here rather
than left to the caller: the same classes and colors across encoders on one
dataset, the projection fitted jointly to features and prototypes, and the
figures treated as qualitative.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from fm_fewshot.services.data.subsets import balanced_subset  # noqa: E402
from fm_fewshot.services.evaluation.metrics import confusion_matrix  # noqa: E402
from fm_fewshot.services.evaluation.report import head_key  # noqa: E402
from fm_fewshot.services.features.cache import read_features  # noqa: E402
from fm_fewshot.services.heads import dacc_baseline  # noqa: E402
from fm_fewshot.shared.contracts import CellSummary  # noqa: E402

K_ORDER = ("5", "10", "full")
# Tableau 10 plus two, enough for the write-up's 8 to 10 classes.
PALETTE = (
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1b9e77", "#7570b3",
)


class MissingRunError(RuntimeError):
    """Raised when a figure needs a cell that no run produced."""


# F1, S1 and P1 take their line styling from here rather than from matplotlib's
# default cycle. The rule is his, from the Stage 2 meeting: the two step counts
# of one scheme are a couple and have to read as one, so they share a hue and
# are separated by the line style. Alphabetical order through the default cycle
# gave a pair two unrelated colors and interleaved the pairs.
SERIES_PALETTE = ("#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759")
BASELINE_COLOR = "#333333"
# Indexed by the step count's rank inside its scheme, so the shorter solve is
# dotted and the longer one solid. A scheme with a single T stays solid; a
# lone dotted line would suggest a partner that is not on the panel.
T_LINESTYLES = (":", "-", "-.", (0, (3, 1, 1, 1)))


def split_scheme(label: str) -> tuple[str, int | None]:
    """Scheme name and step count, from either stage's series label.

    Stage 2 labels a line `fm_standard T=4` (`scheme_label`) and Stage 3
    `fm_prelinear_ce@T4` (`report.method_label`). One parser for both, so the
    pairing cannot hold in one family and quietly fail in the other.
    """
    for separator in ("@T", " T="):
        scheme, found, steps = label.partition(separator)
        if found and steps.isdigit():
            return scheme, int(steps)
    return label, None


def series_order(labels, baseline: str | None = None) -> list[str]:
    """Baseline first, then each scheme with its step counts ascending.

    The legend is the panel's key, so it reads in the order the reader compares
    in: the row being measured against, then a scheme's couple together.
    """
    def key(label: str) -> tuple[int, str, int]:
        scheme, steps = split_scheme(label)
        return (0 if label == baseline else 1, scheme, -1 if steps is None else steps)

    return sorted(labels, key=key)


def series_styles(labels, *, baseline: str | None = None) -> dict[str, dict]:
    """One hue per scheme; inside it, the step counts differ only by line style."""
    labels = list(labels)
    steps: dict[str, list[int]] = {}
    for label in labels:
        if label == baseline:
            continue
        scheme, count = split_scheme(label)
        entry = steps.setdefault(scheme, [])
        if count is not None and count not in entry:
            entry.append(count)
    for entry in steps.values():
        entry.sort()
    hues = {
        scheme: SERIES_PALETTE[i % len(SERIES_PALETTE)] for i, scheme in enumerate(steps)
    }

    styles: dict[str, dict] = {}
    for label in labels:
        if label == baseline:
            styles[label] = {
                "color": BASELINE_COLOR, "linestyle": "--", "marker": "s", "linewidth": 1.7,
            }
            continue
        scheme, count = split_scheme(label)
        paired = len(steps[scheme]) > 1
        rank = steps[scheme].index(count) if paired else 1
        styles[label] = {
            "color": hues[scheme],
            "linestyle": T_LINESTYLES[rank % len(T_LINESTYLES)] if paired else "-",
            "marker": "o",
            "linewidth": 1.6,
        }
    return styles


def draw_caption(fig, caption: str) -> None:  # noqa: ANN001
    """Put a caption under the axes and reserve the room it needs.

    `tight_layout` lays out the axes without knowing a `fig.text` exists, so
    a caption at the bottom, and a two-line one in particular, ends up under
    the x label. The reserve is per line and generous by a hair.
    """
    lines = caption.count(chr(10)) + 1
    fig.tight_layout(rect=(0.0, 0.035 * lines + 0.015, 1.0, 1.0))
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=6)


def pairing_note(styles: dict[str, dict]) -> str:
    """The one line that tells the reader how to read a couple, or nothing.

    Kept short deliberately. It is drawn at 6pt across a 5 inch figure, and a
    sentence long enough to explain itself is a sentence wide enough to be
    clipped at both ends.
    """
    dotted = {split_scheme(label)[1] for label, style in styles.items()
              if style["linestyle"] == T_LINESTYLES[0]}
    solid = {split_scheme(label)[1] for label, style in styles.items()
             if style["linestyle"] == "-" and split_scheme(label)[1] is not None}
    if not dotted:
        return ""
    shorter = ", ".join(f"T={value}" for value in sorted(dotted))
    longer = ", ".join(f"T={value}" for value in sorted(solid))
    return f"one colour per scheme; {shorter} dotted, {longer} solid"


def k_label(k: int | None) -> str:
    return "full" if k is None else str(k)


def class_colors(class_names: list[str]) -> dict[str, str]:
    """Stable class-to-color mapping.

    Sorted, not call-ordered, so the same class keeps its color when the two
    encoders on a dataset are compared. Letting matplotlib assign per call
    would silently break the write-up's comparability rule.
    """
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(sorted(class_names))}


def validate_viz_classes(chosen: list[str], available: list[str]) -> None:
    if not 8 <= len(chosen) <= 10:
        raise ValueError(f"the write-up asks for 8 to 10 classes, got {len(chosen)}")
    missing = [name for name in chosen if name not in available]
    if missing:
        raise ValueError(f"classes not in the dataset: {missing}")


def require_cells(
    cells: list[CellSummary], dataset: str, encoder: str, heads: list[str]
) -> list[CellSummary]:
    """Fail naming what is absent rather than plotting a partial panel."""
    present = [c for c in cells if c.dataset == dataset and c.encoder == encoder]
    for head in heads:
        if not any(c.head == head for c in present):
            raise MissingRunError(
                f"no runs for ({dataset}, {encoder}, {head}); the figure would be "
                "partial, so nothing was written"
            )
    return present


def project_jointly(features: np.ndarray, prototypes: np.ndarray, projector):  # noqa: ANN001
    """Fit the projection on features and prototypes together.

    The write-up is explicit: "fit the projection jointly to the image features
    and prototypes shown in the plot". Fitting on features and transforming
    prototypes afterwards places them in a space they did not help define.
    """
    stacked = np.concatenate([features, prototypes], axis=0)
    embedded = projector.fit_transform(stacked)
    return embedded[: features.shape[0]], embedded[features.shape[0] :]


def size_curve_series(
    cells: list[CellSummary], dataset: str, encoder: str
) -> dict[str, dict]:
    """One series per head, ordered 5 / 10 / full, error bars from the stored std."""
    series: dict[str, dict] = {}
    for cell in cells:
        if cell.dataset != dataset or cell.encoder != encoder:
            continue
        entry = series.setdefault(cell.head, {"x": [], "y": [], "yerr": []})
        entry["x"].append(k_label(cell.k))
        entry["y"].append(cell.mean)
        # None, not 0.0: a single run has no measured spread to draw.
        entry["yerr"].append(cell.std if cell.n_runs > 1 else None)
    for entry in series.values():
        order = sorted(range(len(entry["x"])), key=lambda i: K_ORDER.index(entry["x"][i]))
        for key in ("x", "y", "yerr"):
            entry[key] = [entry[key][i] for i in order]
    return series


def top_confusions(
    matrix: torch.Tensor, class_names: list[str], limit: int = 10
) -> list[tuple[str, str, float]]:
    """Worst off-diagonal cells, for the note to name in prose.

    At 47 and 100 classes the matrix image is unreadable, so the numbers have
    to reach the reader some other way.
    """
    pairs = []
    n = matrix.shape[0]
    for true in range(n):
        for predicted in range(n):
            if true != predicted and matrix[true, predicted] > 0:
                pairs.append((class_names[true], class_names[predicted],
                              float(matrix[true, predicted])))
    return sorted(pairs, key=lambda p: -p[2])[:limit]


def plot_size_curve(
    cells, dataset: str, encoder: str, out: Path, *, dpi: int = 200,
    baseline: str | None = None,
) -> Path:
    """F1 and S1. `baseline`, when given, is drawn as the row being measured against.

    The line ordering and colors come from `series_order` and `series_styles`,
    not from `sorted()` through matplotlib's default cycle. Alphabetically,
    `fm_rolled T=12` and `fm_rolled T=4` are adjacent but were given unrelated
    hues, and the two schemes interleaved; his Stage 2 note asked for the
    couples to be visible as couples.
    """
    series = size_curve_series(cells, dataset, encoder)
    order = series_order(series, baseline)
    styles = series_styles(order, baseline=baseline)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=dpi)
    positions = range(len(K_ORDER))
    for head in order:
        entry = series[head]
        xs = [K_ORDER.index(label) for label in entry["x"]]
        errs = [e if e is not None else 0.0 for e in entry["yerr"]]
        ax.errorbar(xs, entry["y"], yerr=errs, capsize=3, label=head, **styles[head])
    ax.set_xticks(list(positions))
    ax.set_xticklabels(K_ORDER)
    ax.set_xlabel("training images per class (K)")
    ax.set_ylabel("top-1 accuracy, official test split")
    ax.set_title(f"{dataset} / {encoder}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    note = pairing_note(styles)
    if note:
        draw_caption(fig, note)
    else:
        fig.tight_layout()
    return _save(fig, out)


def plot_loss_curves(epochs_csv: Path, best_epoch: int, title: str, out: Path) -> Path:
    rows = list(csv.DictReader(epochs_csv.open(encoding="utf-8")))
    epochs = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200)
    ax.plot(epochs, [float(r["train_loss"]) for r in rows], label="train loss")
    ax.plot(epochs, [float(r["val_loss"]) for r in rows], label="val loss")
    ax.axvline(best_epoch, color="0.4", linestyle="--", linewidth=1,
               label=f"selected epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_confusion(matrix: torch.Tensor, title: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    image = ax.imshow(matrix.numpy(), cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, label="row-normalized rate")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_feature_space(
    points: np.ndarray,
    point_labels: np.ndarray,
    protos: np.ndarray,
    class_names: list[str],
    colors: dict[str, str],
    title: str,
    out: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=200)
    for i, name in enumerate(class_names):
        mask = point_labels == i
        ax.scatter(points[mask, 0], points[mask, 1], s=8, alpha=0.55,
                   color=colors[name], label=name, linewidths=0)
        ax.scatter(protos[i, 0], protos[i, 1], s=180, marker="*",
                   color=colors[name], edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6, loc="best", framealpha=0.85, ncol=2)
    fig.text(0.5, 0.005, "qualitative; stars are class prototypes", ha="center", fontsize=6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def prototypes_for(train_x: torch.Tensor, train_y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """The specified Option A formula, reused so figure and table agree."""
    z = F.normalize(train_x, dim=1)
    mu = torch.stack([z[train_y == c].mean(0) for c in range(n_classes)])
    return F.normalize(mu, dim=1)


def feature_space_inputs(
    dataset: str,
    encoder: str,
    viz_class_ids: list[int],
    *,
    data_root: Path,
    k: int | None = None,
    seed: int = 0,
    max_per_class: int = 60,
):
    """Test features and prototypes for the chosen classes, as raw arrays."""
    train_x, train_y, meta = read_features(
        dataset, "train", encoder, data_root=data_root, l2_normalize=False
    )
    test_x, test_y, _ = read_features(
        dataset, "test", encoder, data_root=data_root, l2_normalize=False
    )
    n_classes = len(meta["class_names"])
    if k is not None:
        subset = balanced_subset(train_y, k, seed, n_classes, dataset)
        train_x, train_y = train_x[subset.idx], subset.labels
    protos = prototypes_for(train_x, train_y, n_classes)

    rows, labels = [], []
    for local, class_id in enumerate(viz_class_ids):
        idx = torch.nonzero(test_y == class_id, as_tuple=False).flatten()[:max_per_class]
        rows.append(F.normalize(test_x[idx], dim=1))
        labels.append(torch.full((idx.shape[0],), local, dtype=torch.int64))
    return (
        torch.cat(rows).numpy(),
        torch.cat(labels).numpy(),
        protos[viz_class_ids].numpy(),
        meta["class_names"],
    )


# --------------------------------------------------------------------------
# Stage 2 (PRD_stage2_figures). S1 to S4 reuse the Stage 1 helpers above rather
# than copying them: one color mapping, one series aggregation, one joint-fit
# rule, so a class cannot drift color between the two stages.
# --------------------------------------------------------------------------

STAGE2_HEADS = ("fm_standard", "fm_rolled")

# Suppress the toolchain version string matplotlib otherwise stamps into the
# PNG. These are graded deliverables that must regenerate byte-identically from
# stored results, and a metadata chunk that moves with the matplotlib version
# would break that without changing a single pixel.
PNG_METADATA = {"Software": None}


def scheme_label(head: str, sample_steps: int) -> str:
    """Name one Stage 2 configuration by its head and its step count.

    T is not part of report's cell key, so a Stage 2 panel that grouped by head
    alone would average T=4 and T=12 into one line and hide exactly the
    discretization effect the write-up asks about.
    """
    return f"{head} T={sample_steps}"


def viz_test_rows(
    test_y: torch.Tensor, class_ids: list[int], max_per_class: int
) -> tuple[torch.Tensor, np.ndarray]:
    """The test rows every compared panel shares, and their local class ids.

    Rows into the test cache, not features: the write-up requires the same test
    examples in each compared plot, and that is only checkable if the panels
    carry the indices they were built from.
    """
    if not class_ids:
        raise ValueError("no visualization classes were given")
    rows, labels = [], []
    for local, class_id in enumerate(class_ids):
        idx = torch.nonzero(test_y == class_id, as_tuple=False).flatten()[:max_per_class]
        rows.append(idx)
        labels.append(np.full(int(idx.shape[0]), local, dtype=np.int64))
    return torch.cat(rows), np.concatenate(labels)


def projection_kinds(spec) -> list[str]:  # noqa: ANN001 - the parsed figure config
    """Which projections a figure config asks for.

    Both write-ups say "PCA or t-SNE may be used" and we only ever produced
    PCA. A list rather than a swap, because the two answer different
    questions: PCA is a linear map with a preimage, which is what lets P3 draw
    the frozen probe's boundary in the plotted plane, and t-SNE is a neighbour
    embedding that can separate what two linear axes cannot hold. The old
    scalar key still parses, so an existing config keeps working.
    """
    value = spec.get("projections", spec.get("projection", "pca"))
    kinds = [value] if isinstance(value, str) else [str(kind) for kind in value]
    if not kinds:
        raise ValueError("no projection asked for; give at least 'pca'")
    return kinds


def projection_suffix(kind: str) -> str:
    """PCA keeps the bare filename every note and report already cites."""
    return "" if kind == "pca" else f"_{kind}"


def variance_note(projector, kind: str) -> str:  # noqa: ANN001 - an sklearn estimator
    """How much of the feature variance the two drawn axes actually carry.

    His question was why PCA helped on some cells and not others. Without this
    number the answer is an impression about how the panel looks; with it, a
    panel that is a blob says so on the panel.
    """
    ratio = getattr(projector, "explained_variance_ratio_", None)
    if ratio is None:
        return f"{kind} is a neighbour embedding: its axes carry no share of the variance"
    share = float(np.sum(np.asarray(ratio)[:2]))
    return f"the two PCA axes carry {share:.1%} of the variance of the projected sets"


def unit_rows(x):  # noqa: ANN001, ANN201 - an array or a Tensor
    """Rows on the unit sphere, which is where the cosine rule reads them.

    The prototypes are unit norm by his own Stage 1 formula while the features
    enter the flow raw, so a panel drawn in raw coordinates is dominated by a
    scale gap the decision rule never sees.
    """
    x = np.asarray(as_array(x), dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


NORMALIZED_NOTE = (
    "every point and prototype is projected onto the unit sphere first, so the "
    "panel shows the directions the cosine rule compares and not the raw scale"
)


def make_projector(kind: str, seed: int):  # noqa: ANN202 - an sklearn estimator
    """The projector both stages fit. Seeded, because the figures must repeat."""
    if kind == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed)
    if kind == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, random_state=seed, init="pca", perplexity=30)
    raise ValueError(f"unknown projection {kind!r}; use 'pca' or 'tsne'")


def write_series_csv(cells, dataset: str, encoder: str, out: Path, *, order=None) -> Path:
    """The numbers behind an accuracy-versus-K panel, so the plot is checkable.

    `order` names the rows and their sequence, for a panel whose lines are the
    protocol's rather than whatever the run store happens to hold. Stage 1 and
    Stage 2 draw every head they find and pass nothing.
    """
    series = size_curve_series(cells, dataset, encoder)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(series.items()) if order is None else [(name, series[name]) for name in order]
    lines = ["head,k,top1,std"]
    for head, entry in rows:
        for x, y, err in zip(entry["x"], entry["y"], entry["yerr"], strict=True):
            lines.append(f"{head},{x},{y:.6f},{'' if err is None else f'{err:.6f}'}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _save(fig, out: Path) -> Path:  # noqa: ANN001 - a matplotlib Figure
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata=PNG_METADATA)
    plt.close(fig)
    return out


def stability_note(losses: list[float]) -> str:
    """What S2 exists to say: did this scheme train stably, and if not, where.

    The write-up's stated purpose for the training curves is verifying
    stability, so the answer is written on the figure instead of being left for
    a reader to infer from the shape of a line.
    """
    if not losses:
        return "no recorded steps"
    for step, value in enumerate(losses, start=1):
        if not math.isfinite(value):
            return f"diverged at step {step}"
    trend = "no net decrease" if losses[-1] > losses[0] else "stable"
    return f"{trend}: {losses[0]:.3g} to {losses[-1]:.3g} over {len(losses)} steps"


def plot_stage2_loss_curves(*, curves, title: str, out: Path, dpi: int = 200) -> Path:
    """S2. One panel per scheme, never a shared axis.

    A velocity residual and a squared endpoint distance are not comparable in
    value, so drawing them on one axis would invite a comparison the numbers do
    not support. Separate panels, and the caption says why.
    """
    fig, axes = plt.subplots(
        1, len(curves), figsize=(5.0 * len(curves), 3.8), dpi=dpi, squeeze=False
    )
    for ax, (name, steps, losses) in zip(axes[0], curves, strict=True):
        ax.plot(steps, losses, linewidth=1.1)
        ax.set_xlabel("training step")
        ax.set_ylabel("training loss")
        ax.set_title(f"{name}\n{stability_note(losses)}", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.text(
        0.5, 0.005,
        "the two losses are a velocity residual and a squared endpoint distance; "
        "their values are not comparable, only their stability",
        ha="center", fontsize=6,
    )
    fig.tight_layout()
    return _save(fig, out)


def selection_note(
    *,
    steps: list[int],
    losses: list[float],
    validation: list[tuple[int, float]],
    selected_step: int | None,
) -> str:
    """What S7 exists to say: which step the run kept, and where the loss bottomed.

    The two are reported together because the interesting fact about this grid
    is that they disagree. Selection is on validation accuracy, and in every run
    of the 108 the training loss is still falling at the step the run keeps, so
    a reader who assumes the kept step is the loss minimum is reading the
    opposite of what happened.
    """
    if not validation or selected_step is None:
        return "no selection grid recorded; the field is the one the last step left"
    loss_min_step = steps[losses.index(min(losses))]
    # The value at the kept step, not the maximum of the curve. They agree when
    # selection did its job, and when they do not the figure should show it
    # rather than paper over it with the number the rule was supposed to find.
    at_kept = dict(validation).get(selected_step)
    scored = f", val top-1 {at_kept:.3f}" if at_kept is not None else ""
    kept = f"kept step {selected_step} of {steps[-1]}{scored}"
    if loss_min_step > selected_step:
        return f"{kept}; loss still falling, its minimum at step {loss_min_step}"
    return f"{kept}; loss minimum at step {loss_min_step}"


# Why the x axis counts steps and not epochs, which he asked at the Stage 2
# meeting: the step budget is one number for the whole grid, while an epoch is a
# different amount of optimizer work in every cell, so epochs would not hold the
# two schemes and the three K values to the same budget.
EPOCH_NOTE = (
    "the unit is the optimizer step, not the epoch: the step budget is fixed and "
    "identical in every cell, while at batch 64 one epoch is about 7 batches at "
    "K=10 on DTD and about 52 at K=full on Aircraft, so an epoch is a different "
    "amount of work in every cell and would not compare across the grid"
)

SELECTION_CAPTION = (
    "selection is on validation top-1, not on the loss; the dashed rule is the "
    f"step the run kept. {EPOCH_NOTE}"
)


def plot_selection(
    *, panels, title: str, out: Path, dpi: int = 200, caption: str = SELECTION_CAPTION
) -> Path:
    """S7. Two rows sharing one x axis per scheme, never a twin y axis.

    A loss spanning three decades and an accuracy in [0, 1] have no common
    scale. Drawing them against two y axes on one plot would put a crossing
    point on the page that is an artifact of where the two scales were pinned,
    and a reader would take it for a fact about the run.
    """
    columns = len(panels)
    fig, axes = plt.subplots(
        2, columns,
        figsize=(5.0 * columns, 5.2),
        dpi=dpi,
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    for column, panel in enumerate(panels):
        steps = list(panel["steps"])
        losses = list(panel["losses"])
        validation = list(panel["validation"])
        selected = panel["selected_step"]
        if selected is not None and not (steps[0] <= selected <= steps[-1]):
            raise ValueError(
                f"selected step {selected} is outside the recorded run, "
                f"steps {steps[0]} to {steps[-1]}; a marker there would "
                "misdescribe the run rather than annotate it"
            )

        top, bottom = axes[0][column], axes[1][column]
        top.plot(steps, losses, linewidth=1.1, color="#333333")
        top.set_yscale("log")
        top.set_ylabel("training loss (log)")
        note = selection_note(
            steps=steps, losses=losses, validation=validation, selected_step=selected
        )
        top.set_title(f"{panel['name']}\n{note}", fontsize=8)
        top.grid(alpha=0.3)

        if validation:
            bottom.plot(
                [step for step, _ in validation],
                [value for _, value in validation],
                linewidth=1.1, marker="o", markersize=2.5, color="#0E6F7B",
            )
        else:
            bottom.text(
                0.5, 0.5, "no validation recorded (eval_every: 0)",
                ha="center", va="center", fontsize=7, transform=bottom.transAxes,
            )
        bottom.set_ylabel("validation top-1")
        bottom.set_xlabel("training step")
        bottom.grid(alpha=0.3)

        # One rule through both rows, so the kept step is read at the same x.
        if selected is not None:
            for ax in (top, bottom):
                ax.axvline(selected, color="#A2432C", linewidth=1.0, linestyle="--")
            bottom.annotate(
                f"kept step {selected}",
                xy=(selected, 0), xycoords=("data", "axes fraction"),
                xytext=(3, 4), textcoords="offset points",
                fontsize=6, color="#A2432C",
            )

    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.005, caption, ha="center", fontsize=6)
    fig.tight_layout()
    return _save(fig, out)


def _panel_grid(count: int, dpi: int):  # noqa: ANN202 - a matplotlib Figure and Axes
    fig, axes = plt.subplots(1, count, figsize=(4.7 * count, 4.7), dpi=dpi, squeeze=False)
    return fig, axes[0]


def _draw_prototypes(ax, prototypes, class_names, colors, *, label: bool) -> None:  # noqa: ANN001
    for i, class_name in enumerate(class_names):
        ax.scatter(
            prototypes[i, 0], prototypes[i, 1], s=170, marker="*",
            color=colors[class_name], edgecolors="black", linewidths=0.8, zorder=5,
            label=class_name if label else None,
        )


def plot_feature_panels(
    *, panels, labels, class_names, colors, title: str, out: Path,
    prototypes=None, boundary=None, note: str = "", dpi: int = 200,
) -> Path:
    """S3 and P3. The original features and the same test examples after each scheme.

    The points arrive already projected. Fitting the projection is the caller's
    job because it has to happen once over every panel, which is a property of
    the comparison and not of the drawing.

    `prototypes` is Stage 2's. Stage 3 passes none: it transports toward nothing,
    and a star on the panel would suggest a target that does not exist. What
    replaces them, when the projection is linear, is `boundary`, the frozen
    probe restricted to the plotted plane (see `reduced_probe`) as
    `(weight, bias, class_ids)` with the plotted classes' global ids.
    """
    fig, axes = _panel_grid(len(panels), dpi)
    # The box is computed once over every panel, so the boundary is the same map
    # in each. Only drawn when there is one: the Stage 2 panels are unchanged.
    extent = None if boundary is None else _panel_extent(panels)
    for column, (ax, (name, points)) in enumerate(zip(axes, panels, strict=True)):
        if boundary is not None:
            _draw_decision_regions(ax, boundary, class_names, colors, extent)
        for i, class_name in enumerate(class_names):
            mask = labels == i
            ax.scatter(
                points[mask, 0], points[mask, 1], s=10, alpha=0.6,
                color=colors[class_name], linewidths=0,
                zorder=None if boundary is None else 2,
                label=class_name if column == 0 else None,
            )
        if prototypes is not None:
            _draw_prototypes(ax, prototypes, class_names, colors, label=False)
        if boundary is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=6, loc="best", framealpha=0.85, ncol=2)
    fig.suptitle(title, fontsize=10)
    draw_caption(fig, _feature_caption(prototypes, boundary, note))
    return _save(fig, out)


def _feature_caption(prototypes, boundary, note: str = "") -> str:  # noqa: ANN001
    text = "qualitative; one projection fitted jointly over all panels"
    if prototypes is not None:
        text += " and the prototypes"
    text += ", so the panels share coordinates"
    if prototypes is not None:
        text += "; stars are class prototypes"
    if boundary is not None:
        text += (
            "; the shading is the frozen probe's decision regions in the plotted "
            "plane, unshaded where the winner is a class this panel does not draw"
        )
    return text if not note else chr(10).join([text, note])


def _panel_extent(panels) -> tuple[float, float, float, float]:  # noqa: ANN001
    """The box every panel is drawn in, so a boundary is the same map in each."""
    stacked = np.concatenate([np.asarray(points).reshape(-1, 2) for _, points in panels])
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    pad = 0.04 * np.maximum(high - low, 1e-9)
    return (
        float(low[0] - pad[0]), float(high[0] + pad[0]),
        float(low[1] - pad[1]), float(high[1] + pad[1]),
    )


DECISION_GRID = 200


def _draw_decision_regions(ax, boundary, class_names, colors, extent) -> None:  # noqa: ANN001
    """The frozen probe's argmax over the plotted plane, at low contrast.

    The argmax runs over every class the probe fits, not only the plotted ones:
    restricting it would draw a classifier the run never used. Regions won by a
    class this panel does not draw are left unshaded rather than recolored.
    """
    weight, bias, class_ids = boundary
    xs = np.linspace(extent[0], extent[1], DECISION_GRID)
    ys = np.linspace(extent[2], extent[3], DECISION_GRID)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    winner = np.argmax(grid @ np.asarray(weight).T + np.asarray(bias), axis=1)
    local = np.zeros(winner.shape[0], dtype=np.int64)
    for i, class_id in enumerate(class_ids):
        local[winner == class_id] = i + 1
    ax.imshow(
        local.reshape(DECISION_GRID, DECISION_GRID),
        origin="lower", extent=extent, aspect="auto", zorder=0, alpha=0.18,
        interpolation="nearest", vmin=0, vmax=len(class_names),
        cmap=ListedColormap(["#ffffff", *(colors[name] for name in class_names)]),
    )


def plot_trajectory_panels(
    *, panels, labels, prototypes, class_names, colors, title: str, out: Path,
    note: str = "", dpi: int = 200,
) -> Path:
    """S4. Every Euler state of a few test examples, in the class colors."""
    fig, axes = _panel_grid(len(panels), dpi)
    for ax, (name, states) in zip(axes, panels, strict=True):
        for row in range(states.shape[1]):
            color = colors[class_names[int(labels[row])]]
            path = states[:, row, :]
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=0.9, alpha=0.8, zorder=2)
            ax.scatter(path[0, 0], path[0, 1], s=22, marker="o", color=color,
                       edgecolors="black", linewidths=0.4, zorder=3)
            ax.scatter(path[-1, 0], path[-1, 1], s=36, marker="X", color=color,
                       edgecolors="black", linewidths=0.4, zorder=4)
        _draw_prototypes(ax, prototypes, class_names, colors, label=ax is axes[0])
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=6, loc="best", framealpha=0.85, ncol=2)
    fig.suptitle(title, fontsize=10)
    caption = (
        "qualitative; circles are the original features, crosses the transported "
        "features, stars the class prototypes"
    )
    draw_caption(fig, caption if not note else chr(10).join([caption, note]))
    return _save(fig, out)


# --------------------------------------------------------------------------
# Shared with both later stages: one joint-fit rule and one reader for what a
# run stored about its own selection, so the two drivers cannot drift apart.
# --------------------------------------------------------------------------


def as_array(x) -> np.ndarray:  # noqa: ANN001 - a Tensor or anything array-like
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def project_blocks(blocks, projector, extra=None):  # noqa: ANN001, ANN201 - arrays
    """One fit over every compared block, then split back.

    The write-up: "Compute the projection jointly over the feature sets and
    prototypes being compared so that the different views correspond to the same
    low-dimensional representation." Transforming a set that did not help define
    the space puts it in coordinates it never earned.

    `extra` is Stage 2's prototypes, which are part of that comparison. Stage 3
    has none, passes nothing, and gets None back in their place.
    """
    flat = [block.reshape(-1, block.shape[-1]) for block in blocks]
    stacked = [*flat] if extra is None else [*flat, as_array(extra)]
    embedded = np.asarray(projector.fit_transform(np.concatenate(stacked, axis=0)))

    split, start = [], 0
    for block, rows in zip(blocks, flat, strict=True):
        stop = start + rows.shape[0]
        split.append(embedded[start:stop].reshape(*block.shape[:-1], embedded.shape[1]))
        start = stop
    return split, (None if extra is None else embedded[start:])


def read_loss_curve(path: Path) -> tuple[list[int], list[float]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [int(r["step"]) for r in rows], [float(r["train_loss"]) for r in rows]


def read_selection(run_dir: Path) -> dict:
    """Everything a selection figure draws for one run, straight from the run.

    `best_epoch` comes from `summary.json` rather than from the validation
    column's argmax, so the marked step is the step the head actually restored,
    including its tie rule. Recomputing it here would let the figure and the
    table disagree without either being obviously wrong.
    """
    run_dir = Path(run_dir)
    steps, losses = read_loss_curve(run_dir / "loss_curve.csv")
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


# --------------------------------------------------------------------------
# Stage 3 (PRD_stage3_figures). P1 accuracy against K, P2 the training and
# validation curves, P3 the before-and-after features, P4 the displacement
# decomposition (ADR-036). P2 and P3 are S7 and S3 with their Stage 2
# assumptions made optional; P1 and P4 are new, and both exist because Stage 3
# is measured against a different row and moves in a constrained space.
# --------------------------------------------------------------------------

STAGE3_HEADS = ("fm_prelinear_ce", "fm_prelinear_guided")

STAGE3_CURVES_CAPTION = (
    "the two training losses are a cross-entropy and a velocity residual; their "
    "values are not comparable, so the columns share no y axis. The kept "
    "checkpoint is the best validation top-1 and the dashed rule is its step. "
    + EPOCH_NOTE
)

DISPLACEMENT_CAPTION = (
    "row(W) is the only part of the displacement the frozen probe can see, so the "
    "null(W) fraction is capacity the loss cannot reward. fm_prelinear_guided is "
    "confined to row(W) by construction and fm_prelinear_ce is not; both near zero, "
    "or both large, falsifies that reading"
)

DISPLACEMENT_SCATTER_POINTS = 400


def stage3_baseline() -> str:
    """The row the Stage 3 lines are measured against, taken from the registry.

    ADR-035 puts the answer on the head, which is what keeps this figure and
    TABLE.md's dAcc column from disagreeing. Stage 1 and Stage 2 report against
    the image prototype row; Stage 3 reports against the linear probe the block
    sits in front of, and a panel that led with the prototypes would put the
    reader's comparison on a number Stage 3 is not measured on.
    """
    baselines = {dacc_baseline(head) for head in STAGE3_HEADS}
    if len(baselines) != 1 or "" in baselines:
        raise ValueError(
            f"the Stage 3 heads disagree about their baseline: {sorted(baselines)}; "
            "P1 has no single row to draw the comparison against"
        )
    return baselines.pop()


def stage3_lines(t_values) -> list[str]:  # noqa: ANN001
    """P1's lines: the baseline first, then both strategies at each T.

    The labels are `report.method_label`'s, `head@variant`. Stage 3 runs T as a
    config variant, so `load_cells` already separates T=4 from T=12 and this
    family needs none of the private aggregation Stage 2 had to build.
    """
    lines = [stage3_baseline()]
    for head in STAGE3_HEADS:
        lines += [f"{head}@T{int(t)}" for t in t_values]
    return lines


def check_stage3_lines(lines, baseline: str) -> None:  # noqa: ANN001
    """Refuse a panel drawn against the wrong row before anything is plotted."""
    lines = list(lines)
    wrong = [line for line in lines if head_key(line) == "prototype"]
    if wrong:
        raise ValueError(
            f"the prototype row {wrong} is Stage 1's and Stage 2's baseline, not "
            f"Stage 3's; P1 is drawn against {baseline!r} (ADR-035)"
        )
    if not lines or lines[0] != baseline:
        first = lines[0] if lines else "no lines at all"
        raise ValueError(
            f"P1's first line has to be {baseline!r}, the row the Stage 3 heads are "
            f"measured against (ADR-035); got {first}"
        )


def plot_stage3_size_curve(
    *, cells, dataset: str, encoder: str, lines, out: Path,
    marked_k: str | None = None, dpi: int = 200,
) -> Path:
    """P1. One panel per (dataset, encoder), the baseline drawn as the baseline.

    The Main Comparison cell is marked here rather than drawn as its own figure,
    so his required comparison and the extended grid are one picture and nobody
    has to reconcile two.
    """
    baseline = stage3_baseline()
    check_stage3_lines(lines, baseline)
    series = size_curve_series(cells, dataset, encoder)
    absent = [line for line in lines if line not in series]
    if absent:
        raise MissingRunError(
            f"no cells for {absent} at ({dataset}, {encoder}); the Stage 3 figure "
            "would be partial, so nothing was written"
        )

    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=dpi)
    if marked_k is not None:
        if marked_k not in K_ORDER:
            raise ValueError(f"the marked cell must be one of {K_ORDER}, got {marked_k!r}")
        position = K_ORDER.index(marked_k)
        ax.axvspan(position - 0.42, position + 0.42, color="#e6ecf2", zorder=0)
        ax.annotate(
            "Main Comparison", xy=(position, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, -9), textcoords="offset points",
            ha="center", fontsize=6, color="#4a5a68",
        )
    styles = series_styles(lines, baseline=baseline)
    for line in lines:
        entry = series[line]
        xs = [K_ORDER.index(label) for label in entry["x"]]
        errs = [0.0 if e is None else e for e in entry["yerr"]]
        ax.errorbar(
            xs, entry["y"], yerr=errs, capsize=3, label=line, zorder=3, **styles[line]
        )
    ax.set_xticks(list(range(len(K_ORDER))))
    ax.set_xticklabels(K_ORDER)
    ax.set_xlabel("training images per class (K)")
    ax.set_ylabel("top-1 accuracy, official test split")
    ax.set_title(f"{dataset} / {encoder}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    # Two short lines rather than one long one: his write-up makes the probe the
    # direct baseline, and the reader needs to know it is the black dashed line.
    caption = f"the dashed black line is {baseline}, the direct baseline his write-up names"
    note = pairing_note(styles)
    if note:
        caption = '\n'.join([caption, note])
    draw_caption(fig, caption)
    return _save(fig, out)


def reduced_probe(projector, weight, bias):  # noqa: ANN001, ANN201
    """The frozen probe restricted to the two plotted components, or None.

    A PCA embeds z as u = (z - mean_) components_^T, so the plane's preimage is
    z = mean_ + u components_ and

        W z + b = u (W components_^T) + (W mean_ + b),

    a linear rule in the plotted coordinates and therefore a boundary that can
    honestly be drawn on the panel. None when the projector exposes no such map:
    a t-SNE panel has no linear preimage to substitute, so P3 draws nothing
    rather than drawing it somewhere it does not belong.
    """
    components = getattr(projector, "components_", None)
    mean = getattr(projector, "mean_", None)
    if components is None or mean is None:
        return None
    w = as_array(weight).astype(np.float64)
    reduced_weight = w @ as_array(components).astype(np.float64).T
    reduced_bias = w @ as_array(mean).astype(np.float64) + as_array(bias).astype(np.float64)
    return reduced_weight, reduced_bias


def displacement_scatter_rows(n: int, limit: int) -> np.ndarray:
    """Evenly spaced rows of a point cloud, never a random draw.

    P4's right panel is one point per test example and the split is too large to
    read at that density. A subsample drawn from an RNG would regenerate
    identically only as long as nothing else touched the same stream, and these
    figures have to be byte-identical on regeneration.
    """
    if n <= limit:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, limit).round().astype(np.int64))


def plot_displacement(
    *, groups, title: str, out: Path, dpi: int = 200,
    scatter_points: int = DISPLACEMENT_SCATTER_POINTS,
) -> Path:
    """P4. Where the displacement went, and what it bought (ADR-036).

    Left, each component's norm as a fraction of the total: only the row(W) part
    can change a logit, so the null(W) fraction is capacity spent where the loss
    cannot see it. Right, the change in the frozen classifier's margin against
    the row-space displacement that produced it.

    The numbers arrive from the run's stored `rowspace.json` and are never
    recomputed here, so the figure and the number in the phase note cannot
    disagree.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=dpi)

    data, positions, faces, ticks, tick_labels, notes = [], [], [], [], [], []
    for index, group in enumerate(groups):
        base = 2.0 * index
        data += [np.asarray(group["row_fraction"]), np.asarray(group["null_fraction"])]
        positions += [base, base + 0.75]
        faces += [PALETTE[index % len(PALETTE)], "#ffffff"]
        ticks.append(base + 0.375)
        tick_labels.append(group["name"])
        if group.get("undefined"):
            notes.append(f"{group['name']}: {group['undefined']} examples were not moved")

    boxes = left.boxplot(
        data, positions=positions, widths=0.62, patch_artist=True,
        medianprops={"color": "#222222", "linewidth": 1.2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.4},
    )
    for patch, face in zip(boxes["boxes"], faces, strict=True):
        patch.set_facecolor(face)
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.9)
    left.set_xticks(ticks)
    left.set_xticklabels(tick_labels, fontsize=7)
    left.set_ylim(0.0, 1.0)
    left.set_ylabel("component norm as a fraction of the displacement")
    left.set_title("where the displacement went", fontsize=9)
    left.grid(alpha=0.3, axis="y")
    left.legend(
        handles=[
            Patch(facecolor="#9a9a9a", edgecolor="#333333", label="row(W), the visible part"),
            Patch(facecolor="#ffffff", edgecolor="#333333", label="null(W), the invisible part"),
        ],
        fontsize=6, loc="best", framealpha=0.85,
    )
    if notes:
        left.set_xlabel("; ".join(notes), fontsize=6)

    for index, group in enumerate(groups):
        row_norm = np.asarray(group["row_norm"])
        rows = displacement_scatter_rows(row_norm.shape[0], scatter_points)
        right.scatter(
            row_norm[rows], np.asarray(group["margin_change"])[rows],
            s=9, alpha=0.5, linewidths=0, color=PALETTE[index % len(PALETTE)],
            label=group["name"],
        )
    right.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    right.set_xlabel("row(W) displacement norm")
    right.set_ylabel("change in the frozen probe's margin")
    right.set_title("what it bought", fontsize=9)
    right.grid(alpha=0.3)
    right.legend(fontsize=6, loc="best", framealpha=0.85)

    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.005, DISPLACEMENT_CAPTION, ha="center", fontsize=6)
    fig.tight_layout()
    return _save(fig, out)


__all__ = [
    "DISPLACEMENT_CAPTION",
    "MissingRunError",
    "PNG_METADATA",
    "SELECTION_CAPTION",
    "STAGE2_HEADS",
    "STAGE3_CURVES_CAPTION",
    "STAGE3_HEADS",
    "as_array",
    "check_stage3_lines",
    "class_colors",
    "displacement_scatter_rows",
    "confusion_matrix",
    "feature_space_inputs",
    "k_label",
    "make_projector",
    "plot_confusion",
    "plot_displacement",
    "plot_feature_panels",
    "plot_feature_space",
    "plot_loss_curves",
    "NORMALIZED_NOTE",
    "draw_caption",
    "pairing_note",
    "plot_size_curve",
    "plot_selection",
    "plot_stage2_loss_curves",
    "plot_stage3_size_curve",
    "plot_trajectory_panels",
    "project_blocks",
    "project_jointly",
    "prototypes_for",
    "read_loss_curve",
    "read_selection",
    "reduced_probe",
    "require_cells",
    "scheme_label",
    "projection_kinds",
    "projection_suffix",
    "series_order",
    "series_styles",
    "size_curve_series",
    "split_scheme",
    "unit_rows",
    "variance_note",
    "stability_note",
    "stage3_baseline",
    "stage3_lines",
    "top_confusions",
    "validate_viz_classes",
    "viz_test_rows",
    "write_series_csv",
]
