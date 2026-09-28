"""Configuration loading.

Reads ``configs/config.yaml`` and resolves every entry under ``paths`` to an
absolute :class:`pathlib.Path`. The data root can be overridden with the
``DR_DATA_ROOT`` environment variable so the same config works both locally
and inside a Kaggle Notebook.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root is two levels above this file: src/utils/config.py -> project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML config and resolve all paths.

    Args:
        path: Path to the YAML config file.

    Returns:
        The config as a nested dict. ``cfg["paths"]`` values are absolute Paths.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_root = os.environ.get("DR_DATA_ROOT")
    if env_root:
        cfg["paths"]["data_root"] = env_root

    # Preprocessed image cache: environment variable, else Kaggle's local temp disk.
    env_cache = os.environ.get("DR_PROCESSED_DIR")
    if env_cache:
        cfg["paths"]["processed_dir"] = env_cache
    elif Path("/kaggle/temp").exists():
        cfg["paths"]["processed_dir"] = "/kaggle/temp/processed"

    resolved = {}
    for key, value in cfg["paths"].items():
        p = Path(value)
        resolved[key] = p if p.is_absolute() else PROJECT_ROOT / p
    cfg["paths"] = resolved
    return cfg
