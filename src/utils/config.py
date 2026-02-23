"""Configuration loading from YAML files.

Reads configs/default.yaml (or a given path) and returns a nested dict.
All paths in config are relative to the project root; this module does not
resolve them to absolute paths (callers may do so using project root).
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. If None, uses
            configs/default.yaml relative to the project root (current working
            directory or parent of 'src' when run from within the project).

    Returns:
        Nested dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if config_path is None:
        # Assume run from project root or from project subdir (e.g. src)
        base = Path.cwd()
        if (base / "configs" / "default.yaml").exists():
            config_path = base / "configs" / "default.yaml"
        elif (base.parent / "configs" / "default.yaml").exists():
            config_path = base.parent / "configs" / "default.yaml"
        else:
            config_path = base / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}
    return config
