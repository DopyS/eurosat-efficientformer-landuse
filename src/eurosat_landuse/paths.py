from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    """Build a path relative to the repository root."""
    return project_root().joinpath(*map(str, parts))


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

