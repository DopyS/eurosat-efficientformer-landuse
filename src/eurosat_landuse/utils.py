from __future__ import annotations

import importlib
from types import ModuleType


def module_available(module_name: str) -> bool:
    """Return True when a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError:
        return False


def require_module(module_name: str, install_hint: str | None = None) -> ModuleType:
    """Import a module or raise a clear error with an install hint."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        hint = install_hint or "Install the project dependencies first."
        raise RuntimeError(f"Missing dependency: {module_name}. {hint}") from exc

