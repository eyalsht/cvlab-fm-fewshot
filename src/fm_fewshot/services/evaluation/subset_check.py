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

Stage 3 adds a second guard over the same store. Its two heads refit the
Stage 1 probe inside themselves at the same `init_seed`, so at one setting the
probe run and both Stage 3 runs must have fitted the identical classifier
(ADR-031). Each such run records a digest of its fitted (W, b), and the check
compares them. The grouping is finer by one field: the classifier depends on
`init_seed`, and the full setting varies exactly that, so comparing across it
would fail on a correct grid.

Runs written before that digest existed carry none. They are skipped rather
than failed, and counted in the output, so it is visible which cells still
have to be rerun before the guard binds over them.

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
# The classifier depends on init_seed as well, and the full setting varies
# exactly that, so the two guards cannot share one grouping.
ClassifierKey = tuple[str, str, object, int, int]


@dataclass(frozen=True)
class RunRows:
    run_id: str
    head: str
    subset_idx: tuple[int, ...]
    init_seed: int = 0
    classifier_digest: str | None = None


@dataclass(frozen=True)
class ClassifierDisagreement:
    """One setting whose runs did not all fit the same frozen classifier."""

    dataset: str
    encoder: str
    k: object
    subset_seed: int
    init_seed: int
    runs: tuple[RunRows, ...]


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
                init_seed=int(cfg.get("init_seed", 0)),
                classifier_digest=payload.get("classifier_digest"),
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


# Heads that fit a linear classifier internally and therefore record a digest.
# Named rather than inferred from the digest's absence, so a run written by a
# head that should have recorded one and did not is visible as a gap.
CLASSIFIER_HEADS = ("linear_probe", "fm_prelinear_ce", "fm_prelinear_guided")


def find_classifier_disagreements(results_dir: Path) -> list[ClassifierDisagreement]:
    """Every setting whose runs did not all fit the same frozen classifier.

    Stage 3 refits the Stage 1 probe inside its own head at the same
    init_seed, so at one (dataset, encoder, k, subset_seed, init_seed) the
    probe run and both Stage 3 runs must carry one digest between them. A run
    with no digest is skipped: the store holds runs written before the field
    existed, and they cannot bind the check.
    """
    disagreements: list[ClassifierDisagreement] = []
    by_classifier: dict[ClassifierKey, list[RunRows]] = {}
    for (dataset, encoder, k, subset_seed), runs in read_runs(results_dir).items():
        for run in runs:
            if run.classifier_digest is None:
                continue
            key: ClassifierKey = (dataset, encoder, k, subset_seed, run.init_seed)
            by_classifier.setdefault(key, []).append(run)

    for key in sorted(by_classifier, key=lambda k: (k[0], k[1], _k_label(k[2]), k[3], k[4])):
        runs = sorted(by_classifier[key], key=lambda run: run.run_id)
        if len({run.classifier_digest for run in runs}) > 1:
            dataset, encoder, k, subset_seed, init_seed = key
            disagreements.append(
                ClassifierDisagreement(
                    dataset=dataset,
                    encoder=encoder,
                    k=k,
                    subset_seed=subset_seed,
                    init_seed=init_seed,
                    runs=tuple(runs),
                )
            )
    return disagreements


def count_missing_digests(results_dir: Path) -> int:
    """Runs by a classifier-fitting head that carry no digest, so bind nothing."""
    return sum(
        1
        for runs in read_runs(results_dir).values()
        for run in runs
        if run.head in CLASSIFIER_HEADS and run.classifier_digest is None
    )


def format_classifier_disagreement(disagreement: ClassifierDisagreement) -> str:
    header = (
        f"{disagreement.dataset} / {disagreement.encoder} / "
        f"k={_k_label(disagreement.k)} / subset_seed={disagreement.subset_seed} / "
        f"init_seed={disagreement.init_seed}"
    )
    lines = [header]
    for run in disagreement.runs:
        lines.append(
            f"    {run.run_id}  head={run.head}  classifier={run.classifier_digest[:16]}"
        )
    return "\n".join(lines)


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

    classifier_disagreements = find_classifier_disagreements(args.results_dir)
    if classifier_disagreements:
        print(f"classifier mismatch in {len(classifier_disagreements)} setting(s):")
        for disagreement in classifier_disagreements:
            print(format_classifier_disagreement(disagreement))
        print(
            "a Stage 3 run must transport into the classifier its Stage 1 row "
            "reports, refitted at the same init_seed (ADR-031); the runs above "
            "did not, so their dAcc is against a baseline they did not share"
        )
        return 1

    checked = sum(len(runs) for runs in read_runs(args.results_dir).values())
    print(f"{checked} run(s) checked, every head at each setting on identical rows")
    missing = count_missing_digests(args.results_dir)
    if missing:
        print(
            f"{missing} head that fits a classifier carries no digest, written "
            "before the field existed; rerun those cells to bind the classifier check"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
