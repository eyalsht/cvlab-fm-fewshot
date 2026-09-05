"""Cell aggregation and TABLE.md regeneration per PRD_evaluation_protocol.

A cell is (dataset, encoder, head, k). The write-up fixes how many runs each
cell holds and what is reported: mean and sample std over three runs, except
the full-data prototype cell, which is a single number.

report refuses to emit an incomplete cell. Averaging two runs where the
protocol wants three would silently produce a number nobody could reproduce
from the stated protocol.
"""

import json
from pathlib import Path

from fm_fewshot.services.evaluation.metrics import aggregate_cell
from fm_fewshot.services.heads import dacc_baseline
from fm_fewshot.shared.contracts import CellSummary

CellKey = tuple[str, str, str, object]


class IncompleteCellError(RuntimeError):
    """Raised when a cell holds fewer runs than the protocol requires."""


def expected_runs(head: str, k: int | None) -> int:
    """How many runs the write-up expects for this cell.

    5-shot and 10-shot vary the subset seed over {0,1,2} for every head. The
    full setting varies the initialization seed, which only means something for
    a head that initializes anything: the prototype head has no subset draw and
    no initialization there, so three seeds would give three identical numbers.
    """
    if k is None and head == "prototype":
        return 1
    return 3


def method_label(cfg: dict) -> str:
    """What the table calls this run: the head, plus its variant when it has one.

    Two configurations of one head are two methods as far as the protocol is
    concerned, so they are two cells and two rows.
    """
    variant = cfg.get("variant", "")
    return f"{cfg['head']}@{variant}" if variant else cfg["head"]


def _k_label(k: int | None) -> str:
    return "full" if k is None else str(k)


def load_cells(results_dir: Path, *, require_complete: bool = True) -> list[CellSummary]:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    grouped: dict[CellKey, list[tuple[str, float]]] = {}
    for run_dir in sorted(results_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        if not run_dir.is_dir() or not summary_path.exists():
            continue  # half-written or unrelated directory
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        key: CellKey = (cfg["dataset"], cfg["encoder"], method_label(cfg), cfg["k"])
        grouped.setdefault(key, []).append((payload["run_id"], payload["test_top1"]))

    cells: list[CellSummary] = []
    for key in sorted(grouped, key=lambda k: (k[0], k[1], k[2], _k_label(k[3]))):
        dataset, encoder, head, k = key
        runs = sorted(grouped[key])
        wanted = expected_runs(head, k)
        if require_complete and len(runs) != wanted:
            raise IncompleteCellError(
                f"cell ({dataset}, {encoder}, {head}, k={_k_label(k)}) holds "
                f"{len(runs)} of {wanted} runs the protocol requires; run the "
                "missing seeds before reporting"
            )
        cells.append(
            aggregate_cell(
                dataset=dataset,
                encoder=encoder,
                head=head,
                k=k,  # type: ignore[arg-type]
                run_ids=tuple(run_id for run_id, _ in runs),
                accuracies=[accuracy for _, accuracy in runs],
            )
        )
    return cells


def head_key(method_label: str) -> str:
    """The registry key inside a table label, which may carry a variant."""
    return method_label.split("@", 1)[0]


def _cell_means(cells: list[CellSummary]) -> dict[tuple[str, str, str, object], float]:
    """Every reported cell's mean, keyed so a row can find its own baseline."""
    return {
        (cell.dataset, cell.encoder, cell.head, cell.k): cell.mean for cell in cells
    }


def _dacc(cell: CellSummary, means: dict[tuple[str, str, str, object], float]) -> str:
    """Acc_method - Acc_baseline, against whichever row this head is measured on.

    Which row that is comes from the head itself, through the registry
    (ADR-035): Stage 2 reports against the image prototypes and Stage 3 against
    the linear probe. `report` therefore has no per-stage branch, and adding a
    stage adds no case here.

    Blank on a row that is a baseline with nothing under it, and blank when the
    cell's own baseline has not been reported yet, rather than guessing at a
    number report cannot support or falling back on whichever baseline happens
    to exist.
    """
    baseline = dacc_baseline(head_key(cell.head))
    if not baseline:
        return ""
    value = means.get((cell.dataset, cell.encoder, baseline, cell.k))
    if value is None:
        return ""
    return f"{cell.mean - value:+.3f}"


def render_table(cells: list[CellSummary]) -> str:
    means = _cell_means(cells)
    lines = [
        "# Results",
        "",
        "Top-1 accuracy on the complete official test split. Mean and sample",
        "standard deviation over the protocol's three runs, except where noted.",
        "dAcc is Acc_method - Acc_baseline at the same dataset, encoder and K,",
        "against the baseline the method is measured on: the prototype row for",
        "the Stage 1 probe and the Stage 2 heads, the linear probe row for the",
        "Stage 3 heads. Blank on a row with no baseline under it, and blank",
        "where that baseline has not been reported.",
        "Regenerated by `uv run python -m fm_fewshot report`; do not edit by hand.",
        "",
        "| Dataset | Encoder | Head | K | Top-1 | Std | Runs | dAcc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        std = f"{cell.std:.3f}" if cell.n_runs > 1 else "single run"
        lines.append(
            f"| {cell.dataset} | {cell.encoder} | {cell.head} | {_k_label(cell.k)} "
            f"| {cell.mean:.3f} | {std} | {cell.n_runs} | {_dacc(cell, means)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_table(
    results_dir: Path, table_path: Path, *, require_complete: bool = True
) -> Path:
    cells = load_cells(results_dir, require_complete=require_complete)
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(render_table(cells), encoding="utf-8")
    return table_path
