"""Cross-head subset guard: the same rows under every head at one setting.

Training subsets come from one seeded sampler shared by every method, so any
two runs at the same (dataset, encoder, k, subset_seed) must have fitted on
byte-identical rows. That is what lets a difference between two heads be read
as a difference between the heads. `RunSummary` records the rows themselves
rather than a hash of them (ADR-012), so the check is a read over finished
results and needs no rerun.

Order and length are part of the comparison. The sampler returns ascending
indices, so two runs whose rows differ in order did not come from one draw
either.

A missing or empty results tree is not a failure. `results/` is gitignored and
a fresh checkout has none, and the gate has to be runnable there.
"""

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SettingKey = tuple[str, str, object, int]


@dataclass(frozen=True)
class RunRows:
    run_id: str
    head: str
    subset_idx: tuple[int, ...]


@dataclass(frozen=True)
class SubsetDisagreement:
    """One setting whose runs did not all fit on the same rows."""

    dataset: str
    encoder: str
    k: object  # int, or None for the full official train split
    subset_seed: int
    runs: tuple[RunRows, ...]


def _k_label(k: object) -> str:
    return "full" if k is None else str(k)


def read_runs(results_dir: Path) -> dict[SettingKey, list[RunRows]]:
    """Group every summary under results_dir by the setting it belongs to."""
    results_dir = Path(results_dir)
    grouped: dict[SettingKey, list[RunRows]] = {}
    if not results_dir.is_dir():
        return grouped

    for run_dir in sorted(results_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        if not run_dir.is_dir() or not summary_path.exists():
            continue  # half-written run, or a file such as TABLE.md
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        key: SettingKey = (
            cfg["dataset"],
            cfg["encoder"],
            cfg["k"],
            int(cfg["subset_seed"]),
        )
        grouped.setdefault(key, []).append(
            RunRows(
                run_id=payload["run_id"],
                head=cfg["head"],
                subset_idx=tuple(int(i) for i in payload["subset_idx"]),
            )
        )
    return grouped


def find_subset_disagreements(results_dir: Path) -> list[SubsetDisagreement]:
    """Every setting whose runs disagree, one entry per setting."""
    disagreements: list[SubsetDisagreement] = []
    grouped = read_runs(results_dir)
    for key in sorted(grouped, key=lambda k: (k[0], k[1], _k_label(k[2]), k[3])):
        runs = sorted(grouped[key], key=lambda run: run.run_id)
        if len({run.subset_idx for run in runs}) > 1:
            dataset, encoder, k, subset_seed = key
            disagreements.append(
                SubsetDisagreement(
                    dataset=dataset,
                    encoder=encoder,
                    k=k,
                    subset_seed=subset_seed,
                    runs=tuple(runs),
                )
            )
    return disagreements


def format_disagreement(disagreement: SubsetDisagreement) -> str:
    """The setting, then every run in it with the rows it actually fitted on."""
    header = (
        f"{disagreement.dataset} / {disagreement.encoder} / "
        f"k={_k_label(disagreement.k)} / subset_seed={disagreement.subset_seed}"
    )
    lines = [header]
    for run in disagreement.runs:
        lines.append(
            f"    {run.run_id}  head={run.head}  "
            f"n={len(run.subset_idx)}  rows={_preview(run.subset_idx)}"
        )
    return "\n".join(lines)


def _preview(idx: Iterable[int], limit: int = 12) -> str:
    values = list(idx)
    shown = ", ".join(str(i) for i in values[:limit])
    return f"[{shown}]" if len(values) <= limit else f"[{shown}, ... +{len(values) - limit}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_subsets",
        description="assert every head at one setting fitted on identical rows",
    )
    parser.add_argument("results_dir", nargs="?", default="results", type=Path)
    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"no {args.results_dir}/ to check")
        return 0

    disagreements = find_subset_disagreements(args.results_dir)
    if disagreements:
        print(f"subset mismatch in {len(disagreements)} setting(s):")
        for disagreement in disagreements:
            print(format_disagreement(disagreement))
        print(
            "the runs above did not fit on the same rows, so their accuracies "
            "are not comparable and the table must not aggregate them"
        )
        return 1

    checked = sum(len(runs) for runs in read_runs(args.results_dir).values())
    print(f"{checked} run(s) checked, every head at each setting on identical rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
