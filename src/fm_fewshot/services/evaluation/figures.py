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
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fm_fewshot.services.data.subsets import balanced_subset  # noqa: E402
from fm_fewshot.services.evaluation.metrics import confusion_matrix  # noqa: E402
from fm_fewshot.services.features.cache import read_features  # noqa: E402
from fm_fewshot.shared.contracts import CellSummary  # noqa: E402

K_ORDER = ("5", "10", "full")
# Tableau 10 plus two, enough for the write-up's 8 to 10 classes.
PALETTE = (
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1b9e77", "#7570b3",
)


class MissingRunError(RuntimeError):
    """Raised when a figure needs a cell that no run produced."""


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


def plot_size_curve(cells, dataset: str, encoder: str, out: Path) -> Path:
    series = size_curve_series(cells, dataset, encoder)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200)
    positions = range(len(K_ORDER))
    for head, entry in sorted(series.items()):
        xs = [K_ORDER.index(label) for label in entry["x"]]
        errs = [e if e is not None else 0.0 for e in entry["yerr"]]
        ax.errorbar(xs, entry["y"], yerr=errs, marker="o", capsize=3, label=head)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(K_ORDER)
    ax.set_xlabel("training images per class (K)")
    ax.set_ylabel("top-1 accuracy, official test split")
    ax.set_title(f"{dataset} / {encoder}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


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


__all__ = [
    "MissingRunError",
    "class_colors",
    "confusion_matrix",
    "feature_space_inputs",
    "k_label",
    "plot_confusion",
    "plot_feature_space",
    "plot_loss_curves",
    "plot_size_curve",
    "project_jointly",
    "prototypes_for",
    "require_cells",
    "size_curve_series",
    "top_confusions",
    "validate_viz_classes",
]
