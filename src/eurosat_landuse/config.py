from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .paths import project_path

DEFAULT_CONFIG_PATH = project_path("configs", "default.yaml")


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve a config path relative to the repository root when needed."""
    if config_path is None:
        return DEFAULT_CONFIG_PATH

    path = Path(config_path)
    if path.is_file():
        return path.resolve()

    root_candidate = project_path(path)
    if root_candidate.is_file():
        return root_candidate.resolve()

    raise FileNotFoundError(f"Config file not found: {config_path}")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary."""
    yaml = _load_yaml_module()
    resolved = resolve_config_path(config_path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Config file must contain a mapping at the top level.")
    return data


def describe_config(config: Mapping[str, Any]) -> str:
    """Create a compact human-readable summary for CLI output."""
    project = config.get("project", {})
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    outputs = config.get("outputs", {})

    lines = [
        "Configuration summary:",
        f"- project: {project.get('name', 'unknown')}",
        f"- title: {project.get('title', 'unknown')}",
        f"- dataset: {data.get('dataset_name', 'unknown')}",
        f"- model: {model.get('name', 'unknown')}",
        f"- epochs: {training.get('epochs', 'unknown')}",
        f"- batch_size: {training.get('batch_size', 'unknown')}",
        f"- output_root: {outputs.get('root', 'unknown')}",
    ]
    return "\n".join(lines)


def _load_yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to read configuration files. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return yaml

