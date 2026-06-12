from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import describe_config, load_config
from .utils import module_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluation entrypoint for the EuroSAT project scaffold.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    print(describe_config(config))
    print("Evaluation scaffold is ready.")
    print(f"- torch installed: {module_available('torch')}")
    print("Actual evaluation will be implemented in the next stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

