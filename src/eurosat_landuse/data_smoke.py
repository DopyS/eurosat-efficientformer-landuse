from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import describe_config, load_config
from .data import build_data_pipeline, build_dataloaders, build_datasets, summarize_splits
from .utils import module_available


REQUIRED_MODULES = ("torch", "torchvision")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-smoke",
        description="Check the EuroSAT data pipeline and optionally load one batch.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--download", action="store_true", help="Download EuroSAT if it is not present.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for the smoke test.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only print config and dependency status without importing torchvision datasets.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    print(describe_config(config))
    print(f"- planned data pipeline: {build_data_pipeline(config)}")

    dependency_status = {module: module_available(module) for module in REQUIRED_MODULES}
    print(f"- dependency status: {dependency_status}")
    if args.check_only:
        return 0

    missing = [module for module, available in dependency_status.items() if not available]
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install dependencies with `pip install -r requirements.txt` before running the data smoke test.")
        return 2

    datasets = build_datasets(config, download=args.download)
    print(f"- split sizes: {summarize_splits(datasets)}")

    loaders = build_dataloaders(config, download=False, batch_size=args.batch_size)
    images, labels = next(iter(loaders["train"]))
    print(f"- train batch image shape: {tuple(images.shape)}")
    print(f"- train batch label shape: {tuple(labels.shape)}")
    print("EuroSAT data smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
