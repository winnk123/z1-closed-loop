"""Configuration loading for the Z1 dry-run planner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML configuration file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")

    for section in ("camera", "relay", "task", "planning", "artifacts"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing mapping: {section}")

    serial = str(config["camera"].get("serial", ""))
    if not serial:
        raise ValueError("camera.serial is required")
    if not str(config["relay"].get("url", "")):
        raise ValueError("relay.url is required")
    return config
