"""Command-line dispatch: run | features | sweep | report.

Thin argument parsing only; behavior lives behind the sdk. run, sweep, and
report arrive with the evaluation harness.
"""

import argparse
from pathlib import Path

from fm_fewshot import sdk
from fm_fewshot.shared.gatekeeper import HeavyJobOnCpuError


def _features_parser(subparsers) -> None:  # noqa: ANN001
    p = subparsers.add_parser("features", help="build a feature cache")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--encoder", default="clip_vit_b32")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-heavy-on-cpu", action="store_true")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fm_fewshot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _features_parser(subparsers)
    for name in ("run", "sweep", "report"):
        subparsers.add_parser(name, help="arrives with the evaluation harness")
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
    else:
        raise SystemExit(f"{args.command} is not implemented yet")


if __name__ == "__main__":
    main()
