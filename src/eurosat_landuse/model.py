from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_model(config: Mapping[str, Any]) -> None:
    raise NotImplementedError("EfficientFormerV2 model construction will be implemented in the next stage.")


def build_model_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight summary for documentation and CLI output."""
    model_cfg = dict(config.get("model", {}))
    return {
        "name": model_cfg.get("name", "efficientformerv2_s0.snap_dist_in1k"),
        "num_classes": model_cfg.get("num_classes", 10),
        "pretrained": model_cfg.get("pretrained", True),
        "status": "scaffold_only",
    }

