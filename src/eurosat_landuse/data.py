from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_data_pipeline(config: Mapping[str, Any]) -> dict[str, Any]:
    """Placeholder for the future EuroSAT dataset and augmentation pipeline."""
    data_cfg = dict(config.get("data", {}))
    augmentation_cfg = dict(config.get("augmentation", {}))
    return {
        "dataset_name": data_cfg.get("dataset_name", "EuroSAT"),
        "image_size": data_cfg.get("image_size", 224),
        "num_workers": data_cfg.get("num_workers", 2),
        "augmentation": augmentation_cfg,
        "status": "scaffold_only",
    }


def build_datasets(config: Mapping[str, Any]) -> None:
    raise NotImplementedError("Dataset construction will be implemented in the next stage.")


def build_dataloaders(config: Mapping[str, Any]) -> None:
    raise NotImplementedError("DataLoader construction will be implemented in the next stage.")

