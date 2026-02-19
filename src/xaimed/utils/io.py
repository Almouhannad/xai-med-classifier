"""I/O helpers for configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file is invalid or unsupported."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If YAML parser is unavailable or file content is not a mapping.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
        raise ConfigError(
            "PyYAML is required to read YAML config files. Install it with `pip install pyyaml`."
        ) from exc

    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file must contain a top-level mapping, got {type(data).__name__}."
        )

    return data