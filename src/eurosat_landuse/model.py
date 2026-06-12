from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .utils import require_module

FALLBACK_MODEL_NAME = "efficientformerv2_s0"


def build_model(config: Mapping[str, Any]):
    """Build the EfficientFormerV2 classifier from the project config."""
    timm = require_module("timm", "Install dependencies with `pip install -r requirements.txt`.")
    model_cfg = dict(config.get("model", {}))
    requested_name = model_cfg.get("name", FALLBACK_MODEL_NAME)
    model_name = resolve_model_name(str(requested_name))
    num_classes = int(model_cfg.get("num_classes", 10))
    pretrained = bool(model_cfg.get("pretrained", True))
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def resolve_model_name(model_name: str) -> str:
    """Resolve project model names to names available in the installed timm version."""
    timm = require_module("timm", "Install dependencies with `pip install -r requirements.txt`.")
    available = set(timm.list_models("*efficientformerv2*"))
    if model_name in available:
        return model_name
    base_name = model_name.split(".", maxsplit=1)[0]
    if base_name in available:
        return base_name
    raise ValueError(
        f"Model `{model_name}` is not available in this timm installation. "
        f"Available EfficientFormerV2 models: {sorted(available)}"
    )


def build_model_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight summary for documentation and CLI output."""
    model_cfg = dict(config.get("model", {}))
    requested_name = str(model_cfg.get("name", FALLBACK_MODEL_NAME))
    try:
        resolved_name = resolve_model_name(requested_name)
    except Exception:
        resolved_name = "unresolved"
    return {
        "name": requested_name,
        "resolved_name": resolved_name,
        "num_classes": model_cfg.get("num_classes", 10),
        "pretrained": model_cfg.get("pretrained", True),
        "status": "implemented",
    }


def count_parameters(model: Any) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}
