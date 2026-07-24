from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_model_registry(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def enabled_models(registry: dict[str, Any], role: str = "target") -> list[dict[str, Any]]:
    key = "target_models" if role == "target" else "judge_models"
    models = registry.get(key) or []
    if not isinstance(models, list):
        raise ValueError(f"{key} must be a list")
    return [dict(model) for model in models if model.get("enabled", False)]


def find_model(registry: dict[str, Any], model_id: str, role: str = "target") -> dict[str, Any]:
    for model in enabled_models(registry, role=role):
        if model.get("id") == model_id:
            return model
    raise ValueError(f"enabled {role} model not found: {model_id}")
