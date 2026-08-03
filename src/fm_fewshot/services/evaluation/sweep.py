"""The Stage 1 grid: every cell the write-up asks for, in one command.

The grid is derived from the protocol rather than enumerated by hand, so a
sweep cannot silently omit a seed. Which seed varies depends on the setting:
5-shot and 10-shot vary the subset seed, and the full setting varies the
initialization seed, because there is no subset to draw.
"""

import dataclasses
from pathlib import Path

import yaml

from fm_fewshot.services.evaluation.loop import run_experiment
from fm_fewshot.services.evaluation.report import expected_runs
from fm_fewshot.shared.contracts import ExperimentConfig, RunSummary

SEEDS = (0, 1, 2)


def grid(
    datasets: list[str],
    encoders: dict[str, list[str]],
    heads: list[str],
    ks: list[int | None],
    head_params: dict[str, dict] | None = None,
) -> list[ExperimentConfig]:
    """Expand the protocol into one config per run.

    encoders maps an encoder name to the datasets it runs on, because the
    write-up puts ResNet-18 on both datasets and DINOv2 on only one.
    """
    head_params = head_params or {}
    configs: list[ExperimentConfig] = []
    for dataset in datasets:
        for encoder, encoder_datasets in encoders.items():
            if dataset not in encoder_datasets:
                continue
            for head in heads:
                for k in ks:
                    for seed in SEEDS[: expected_runs(head, k)]:
                        subset_seed, init_seed = (0, seed) if k is None else (seed, 0)
                        label = "full" if k is None else str(k)
                        configs.append(
                            ExperimentConfig(
                                run_name=f"{dataset}-{encoder}-{head}-k{label}-s{seed}",
                                dataset=dataset,
                                encoder=encoder,
                                head=head,
                                head_params=dict(head_params.get(head, {})),
                                k=k,
                                subset_seed=subset_seed,
                                init_seed=init_seed,
                            )
                        )
    return configs


def load_grid(path: Path) -> list[ExperimentConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return grid(
        datasets=raw["datasets"],
        encoders=raw["encoders"],
        heads=raw["heads"],
        ks=[None if k in ("full", None) else int(k) for k in raw["ks"]],
        head_params=raw.get("head_params"),
    )


def run_sweep(
    configs: list[ExperimentConfig],
    *,
    data_root: Path = Path("data"),
    results_dir: Path = Path("results"),
    progress: bool = False,
) -> list[RunSummary]:
    summaries = []
    for i, cfg in enumerate(configs, start=1):
        summary = run_experiment(cfg, data_root=data_root, results_dir=results_dir)
        summaries.append(summary)
        if progress:
            print(
                f"[{i}/{len(configs)}] {cfg.run_name}: "
                f"top1={summary.test_top1:.4f} fit={summary.fit_seconds:.1f}s",
                flush=True,
            )
    return summaries


def as_dicts(configs: list[ExperimentConfig]) -> list[dict]:
    return [dataclasses.asdict(cfg) for cfg in configs]
