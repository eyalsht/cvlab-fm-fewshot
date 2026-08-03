"""Drive all four Stage 1 figure families from stored results (ADR-015).

Reads results/ and the feature caches, writes assets/<dataset>/. Fails naming
what is missing rather than writing a partial panel.
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from fm_fewshot.services.evaluation import figures
from fm_fewshot.services.evaluation.report import load_cells
from fm_fewshot.services.features.cache import read_features


def _projector(kind: str, seed: int):  # noqa: ANN202
    if kind == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed)
    if kind == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, random_state=seed, init="pca", perplexity=30)
    raise ValueError(f"unknown projection {kind!r}; use 'pca' or 'tsne'")


def _find_run(results_dir: Path, **match) -> tuple[str, dict]:
    for run_dir in sorted(results_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        if all(cfg.get(key) == value for key, value in match.items()):
            return run_dir.name, payload
    raise figures.MissingRunError(f"no stored run matching {match}")


def make_figures(
    figure_config: Path,
    *,
    results_dir: Path = Path("results"),
    data_root: Path = Path("data"),
    assets_dir: Path = Path("assets"),
) -> list[Path]:
    spec = yaml.safe_load(Path(figure_config).read_text(encoding="utf-8"))
    cells = load_cells(results_dir)
    written: list[Path] = []

    encoders_by_dataset: dict[str, list[str]] = {}
    for cell in cells:
        encoders_by_dataset.setdefault(cell.dataset, [])
        if cell.encoder not in encoders_by_dataset[cell.dataset]:
            encoders_by_dataset[cell.dataset].append(cell.encoder)

    for dataset, dataset_spec in spec["datasets"].items():
        if dataset not in encoders_by_dataset:
            raise figures.MissingRunError(
                f"no stored runs for dataset {dataset!r} under {results_dir}; "
                "run the sweep before generating figures"
            )
        out_dir = Path(assets_dir) / dataset
        _, _, meta = read_features(
            dataset, "test", encoders_by_dataset[dataset][0],
            data_root=data_root, l2_normalize=False,
        )
        class_names = list(meta["class_names"])
        viz_classes = dataset_spec["viz_classes"]
        figures.validate_viz_classes(viz_classes, class_names)
        viz_ids = [class_names.index(name) for name in viz_classes]
        colors = figures.class_colors(viz_classes)

        for encoder in encoders_by_dataset[dataset]:
            heads = sorted({c.head for c in cells if c.dataset == dataset})
            figures.require_cells(cells, dataset, encoder, heads)

            # F1
            written.append(
                figures.plot_size_curve(
                    cells, dataset, encoder, out_dir / f"size_curve_{encoder}.png"
                )
            )
            _write_series_csv(cells, dataset, encoder, out_dir / f"size_curve_{encoder}.csv")

            # F2, the representative 10-shot linear-probe run the write-up asks for
            run_id, payload = _find_run(
                results_dir, dataset=dataset, encoder=encoder, head="linear_probe", k=10,
                subset_seed=0,
            )
            written.append(
                figures.plot_loss_curves(
                    results_dir / run_id / "epochs.csv",
                    payload["best_epoch"],
                    f"{dataset} / {encoder} / linear probe, K=10",
                    out_dir / f"loss_curves_{encoder}.png",
                )
            )

            # F4
            points, labels, protos, _ = figures.feature_space_inputs(
                dataset, encoder, viz_ids, data_root=data_root,
                max_per_class=spec["max_per_class"],
            )
            projected, projected_protos = figures.project_jointly(
                points, protos, _projector(spec["projection"], spec["viz_seed"])
            )
            written.append(
                figures.plot_feature_space(
                    projected, labels, projected_protos, viz_classes, colors,
                    f"{dataset} / {encoder} ({spec['projection'].upper()})",
                    out_dir / f"features_{encoder}_{spec['projection']}.png",
                )
            )

        # F3, one per dataset at the configured representative setting
        setting = dataset_spec["confusion_setting"]
        k = None if setting["k"] == "full" else int(setting["k"])
        run_id, payload = _find_run(
            results_dir, dataset=dataset, encoder=setting["encoder"],
            head=setting["head"], k=k,
        )
        predictions = torch.from_numpy(np.load(results_dir / run_id / "preds.npy"))
        _, test_y, _ = read_features(
            dataset, "test", setting["encoder"], data_root=data_root, l2_normalize=False
        )
        matrix = figures.confusion_matrix(predictions, test_y, len(class_names))
        written.append(
            figures.plot_confusion(
                matrix,
                f"{dataset} / {setting['encoder']} / {setting['head']}, K={setting['k']}",
                out_dir / f"confusion_{setting['encoder']}_{setting['head']}.png",
            )
        )
        _write_confusions(
            figures.top_confusions(matrix, class_names, limit=15),
            out_dir / "top_confusions.csv",
        )

    return written


def _write_series_csv(cells, dataset: str, encoder: str, out: Path) -> None:
    series = figures.size_curve_series(cells, dataset, encoder)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["head,k,top1,std"]
    for head, entry in sorted(series.items()):
        for x, y, err in zip(entry["x"], entry["y"], entry["yerr"], strict=True):
            lines.append(f"{head},{x},{y:.6f},{'' if err is None else f'{err:.6f}'}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_confusions(pairs, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["true,predicted,rate"]
    lines += [f"{t},{p},{r:.4f}" for t, p, r in pairs]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
