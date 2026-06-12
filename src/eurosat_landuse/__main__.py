from __future__ import annotations

from .config import describe_config, load_config


def main() -> int:
    config = load_config()
    print(describe_config(config))
    print("Use one of the dedicated entrypoints:")
    print("  python -m src.eurosat_landuse.train --config configs/default.yaml")
    print("  python -m src.eurosat_landuse.evaluate --config configs/default.yaml")
    print("  python -m src.eurosat_landuse.predict --config configs/default.yaml --image path/to/image.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

