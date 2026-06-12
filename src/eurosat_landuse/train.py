from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import describe_config, load_config
from .data import build_data_pipeline
from .model import build_model_summary
from .paths import ensure_directory, project_path
from .utils import module_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Training entrypoint for the EuroSAT project scaffold.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--show-config", action="store_true", help="Print the loaded config and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.show_config:
        print(describe_config(config))
        return 0

    ensure_directory(project_path("outputs"))
    print(describe_config(config))
    print("Training scaffold is ready.")
    print(f"- torch installed: {module_available('torch')}")
    print(f"- torchvision installed: {module_available('torchvision')}")
    print(f"- timm installed: {module_available('timm')}")
    print(f"- planned data pipeline: {build_data_pipeline(config)}")
    print(f"- planned model summary: {build_model_summary(config)}")
    print("Actual training will be implemented in the next stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

