"""Command-line dispatch: features | run | sweep | report | figures | reverse.

Thin argument parsing only; behavior lives behind the sdk.
"""

import argparse
from pathlib import Path

from fm_fewshot import sdk
from fm_fewshot.shared.gatekeeper import HeavyJobOnCpuError


def _features_parser(subparsers) -> None:  # noqa: ANN001
    p = subparsers.add_parser("features", help="build a feature cache")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--encoder", default="resnet18")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-heavy-on-cpu", action="store_true")


def _run_parser(subparsers) -> None:  # noqa: ANN001
    p = subparsers.add_parser("run", help="run one experiment from a config file")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))


def _sweep_parser(subparsers) -> None:  # noqa: ANN001
    p = subparsers.add_parser("sweep", help="run every cell of a grid config")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--dry-run", action="store_true", help="list the runs and exit")


def _reverse_parser(subparsers) -> None:  # noqa: ANN001
    p = subparsers.add_parser(
        "reverse", help="integrate a Stage 2 run's field backwards (FR18, diagnostics only)"
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--run-id",
        default=None,
        help="the stored run to annotate; found from the config when omitted",
    )
    p.add_argument(
        "--mode",
        action="append",
        choices=["cycle", "basin", "meet", "volume", "all"],
        help="repeatable; defaults to cycle, basin and meet",
    )
    p.add_argument("--reverse-steps", type=int, nargs="+", default=[4, 12, 32])
    p.add_argument("--reverse-method", nargs="+", default=["euler", "midpoint"])
    p.add_argument("--sigma-scale", type=float, default=1.0)
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--meet-times", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--n-hutchinson", type=int, default=1)


def _report_parser(subparsers) -> None:  # noqa: ANN001
    p = subparsers.add_parser("report", help="regenerate results/TABLE.md")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--table", type=Path, default=None)
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="report cells holding fewer runs than the protocol requires",
    )


def _figures_parser(subparsers) -> None:  # noqa: ANN001
    p = subparsers.add_parser("figures", help="regenerate the figure families")
    p.add_argument("--config", type=Path, default=Path("config/figures.yaml"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--assets-dir", type=Path, default=Path("assets"))
    p.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 for the Stage 1 families F1 to F4, 2 for the Stage 2 families S1 to S4",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fm_fewshot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _features_parser(subparsers)
    _run_parser(subparsers)
    _sweep_parser(subparsers)
    _report_parser(subparsers)
    _figures_parser(subparsers)
    _reverse_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "features":
        try:
            path = sdk.build_features(
                args.dataset,
                args.split,
                encoder=args.encoder,
                data_root=args.data_root,
                batch_size=args.batch_size,
                device=args.device,
                allow_heavy_on_cpu=args.allow_heavy_on_cpu,
            )
        except HeavyJobOnCpuError as error:
            raise SystemExit(str(error)) from error
        print(path)

    elif args.command == "run":
        from fm_fewshot.shared.config import load_config

        summary = sdk.run_experiment(
            load_config(args.config),
            data_root=args.data_root,
            results_dir=args.results_dir,
        )
        print(f"{summary.run_id} top1={summary.test_top1:.4f}")

    elif args.command == "sweep":
        from fm_fewshot.services.evaluation.sweep import load_grid

        configs = load_grid(args.config)
        if args.dry_run:
            for cfg in configs:
                print(cfg.run_name)
            print(f"{len(configs)} runs")
            return
        sdk.run_sweep(
            configs,
            data_root=args.data_root,
            results_dir=args.results_dir,
            progress=True,
        )

    elif args.command == "figures":
        from fm_fewshot.services.evaluation.figures import MissingRunError

        generate = sdk.make_figures if args.stage == 1 else sdk.make_stage2_figures
        try:
            for path in generate(
                args.config,
                results_dir=args.results_dir,
                data_root=args.data_root,
                assets_dir=args.assets_dir,
            ):
                print(path)
        except MissingRunError as error:
            raise SystemExit(str(error)) from error

    elif args.command == "reverse":
        from fm_fewshot.services.evaluation.reverse_run import (
            DEFAULT_MODES,
            MissingRunError,
            NotAnFmRunError,
        )
        from fm_fewshot.services.flow.reverse import ReverseConfig
        from fm_fewshot.shared.config import load_config

        chosen = args.mode or list(DEFAULT_MODES)
        modes = list(DEFAULT_MODES) if "all" in chosen else chosen
        primary = "volume" if modes == ["volume"] else "cycle"
        try:
            path = sdk.run_reverse(
                load_config(args.config),
                reverse_cfg=ReverseConfig(
                    mode=primary,
                    reverse_steps=tuple(args.reverse_steps),
                    reverse_method=tuple(args.reverse_method),
                    sigma_scale=args.sigma_scale,
                    n_samples=args.n_samples,
                    meet_times=tuple(args.meet_times),
                    n_hutchinson=args.n_hutchinson,
                ),
                modes=modes,
                data_root=args.data_root,
                results_dir=args.results_dir,
                run_id=args.run_id,
            )
        except (MissingRunError, NotAnFmRunError, ValueError) as error:
            raise SystemExit(str(error)) from error
        print(path)

    elif args.command == "report":
        from fm_fewshot.services.evaluation.report import IncompleteCellError, write_table

        table = args.table or args.results_dir / "TABLE.md"
        try:
            path = write_table(
                args.results_dir, table, require_complete=not args.allow_incomplete
            )
        except IncompleteCellError as error:
            raise SystemExit(str(error)) from error
        print(path)


if __name__ == "__main__":
    main()
