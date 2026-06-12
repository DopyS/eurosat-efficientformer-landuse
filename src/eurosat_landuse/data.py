from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .paths import project_path
from .utils import require_module

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


def build_data_pipeline(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight summary of the configured data pipeline."""
    data_cfg = dict(config.get("data", {}))
    augmentation_cfg = dict(config.get("augmentation", {}))
    return {
        "dataset_name": data_cfg.get("dataset_name", "EuroSAT"),
        "root": data_cfg.get("root", "data/eurosat"),
        "image_size": data_cfg.get("image_size", 224),
        "num_workers": data_cfg.get("num_workers", 2),
        "classes": EUROSAT_CLASSES,
        "augmentation": augmentation_cfg,
        "status": "implemented_requires_torchvision",
    }


def split_indices(length: int, config: Mapping[str, Any]) -> SplitIndices:
    """Create deterministic train/val/test indices."""
    if length <= 0:
        raise ValueError("Dataset length must be positive.")

    data_cfg = dict(config.get("data", {}))
    project_cfg = dict(config.get("project", {}))
    train_ratio = float(data_cfg.get("train_ratio", 0.7))
    val_ratio = float(data_cfg.get("val_ratio", 0.15))
    test_ratio = float(data_cfg.get("test_ratio", 0.15))
    seed = int(project_cfg.get("seed", 42))

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}.")

    indices = list(range(length))
    random.Random(seed).shuffle(indices)

    train_end = int(length * train_ratio)
    val_end = train_end + int(length * val_ratio)
    return SplitIndices(
        train=indices[:train_end],
        val=indices[train_end:val_end],
        test=indices[val_end:],
    )


def build_transforms(config: Mapping[str, Any], *, train: bool):
    """Build torchvision transforms for EuroSAT images."""
    transforms = require_module(
        "torchvision.transforms",
        "Install dependencies with `pip install -r requirements.txt`.",
    )
    data_cfg = dict(config.get("data", {}))
    augmentation_cfg = dict(config.get("augmentation", {}))
    image_size = int(data_cfg.get("image_size", 224))

    if train:
        steps = []
        if augmentation_cfg.get("random_resized_crop", True):
            steps.append(transforms.RandomResizedCrop(image_size))
        else:
            steps.append(transforms.Resize((image_size, image_size)))
        if augmentation_cfg.get("horizontal_flip", True):
            steps.append(transforms.RandomHorizontalFlip())
        if augmentation_cfg.get("color_jitter", True):
            steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    else:
        steps = [transforms.Resize((image_size, image_size))]

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


def build_datasets(config: Mapping[str, Any], *, download: bool = False) -> dict[str, Any]:
    """Build deterministic train/val/test EuroSAT dataset subsets."""
    datasets_module = require_module(
        "torchvision.datasets",
        "Install dependencies with `pip install -r requirements.txt`.",
    )
    torch_utils_data = require_module(
        "torch.utils.data",
        "Install dependencies with `pip install -r requirements.txt`.",
    )

    data_cfg = dict(config.get("data", {}))
    root = project_path(data_cfg.get("root", "data/eurosat"))
    train_dataset = datasets_module.EuroSAT(
        root=str(root),
        transform=build_transforms(config, train=True),
        download=download,
    )
    eval_dataset = datasets_module.EuroSAT(
        root=str(root),
        transform=build_transforms(config, train=False),
        download=False,
    )
    splits = split_indices(len(train_dataset), config)

    return {
        "train": torch_utils_data.Subset(train_dataset, splits.train),
        "val": torch_utils_data.Subset(eval_dataset, splits.val),
        "test": torch_utils_data.Subset(eval_dataset, splits.test),
    }


def build_dataloaders(
    config: Mapping[str, Any],
    *,
    download: bool = False,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Build DataLoader objects for the EuroSAT splits."""
    torch_utils_data = require_module(
        "torch.utils.data",
        "Install dependencies with `pip install -r requirements.txt`.",
    )

    data_cfg = dict(config.get("data", {}))
    training_cfg = dict(config.get("training", {}))
    datasets = build_datasets(config, download=download)
    loader_batch_size = int(batch_size or training_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))

    return {
        "train": torch_utils_data.DataLoader(
            datasets["train"],
            batch_size=loader_batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": torch_utils_data.DataLoader(
            datasets["val"],
            batch_size=loader_batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": torch_utils_data.DataLoader(
            datasets["test"],
            batch_size=loader_batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def summarize_splits(datasets: Mapping[str, Any]) -> dict[str, int]:
    """Return split sizes for logging and smoke tests."""
    return {split: len(dataset) for split, dataset in datasets.items()}
