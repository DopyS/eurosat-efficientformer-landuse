from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np

from .utils import require_module


def set_seed(config: Mapping[str, Any]) -> int:
    """Set random seeds for reproducible experiments."""
    torch = require_module("torch", "Install dependencies with `pip install -r requirements.txt`.")
    project_cfg = dict(config.get("project", {}))
    seed = int(project_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def select_device(preferred: str = "auto"):
    """Select torch device with Apple MPS support when available."""
    torch = require_module("torch", "Install dependencies with `pip install -r requirements.txt`.")
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

