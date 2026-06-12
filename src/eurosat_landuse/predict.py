from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import describe_config, load_config
from .utils import module_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Prediction entrypoint for the EuroSAT project scaffold.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--image", default=None, help="Path to one image for future prediction.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    print(describe_config(config))
    print("Prediction scaffold is ready.")
    print(f"- streamlit installed: {module_available('streamlit')}")
    if args.image:
        print(f"- image path: {args.image}")
    print("Actual prediction will be implemented in the next stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

