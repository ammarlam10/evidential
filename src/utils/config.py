"""Thin helpers for loading YAML configs and JSON norm stats."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_norm_stats(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def save_norm_stats(stats: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=2)
