from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(payload: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    resolved = deepcopy(config)
    paths = resolved.setdefault("paths", {})
    for key, value in list(paths.items()):
        paths[key] = str((ROOT / value).resolve())
    return resolved
