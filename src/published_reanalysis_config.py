"""Versioned input-path configuration for deposited-data reanalyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


CONFIG_SCHEMA_VERSION = "1.0"
REQUIRED_DATASETS = {
    "figure5_dry",
    "figure5_total",
    "Ensure",
    "Sucralose",
    "Sucrose",
    "EnsureQuinineCohort",
    "QuinineEnsureMatched",
    "Water",
    "QuinineWaterMatched",
}


def load_published_reanalysis_config(path: str | Path) -> dict[str, object]:
    """Load and validate a published-data path configuration."""

    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "unsupported published-reanalysis config schema: "
            f"{config.get('schema_version')!r}"
        )
    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("published-reanalysis config must contain a datasets mapping")
    missing = REQUIRED_DATASETS.difference(datasets)
    if missing:
        raise ValueError(f"published-reanalysis config is missing datasets: {sorted(missing)}")
    for name, relative_path in datasets.items():
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"dataset {name!r} must be a safe relative path")
    return config


def resolve_dataset_paths(
    data_root: str | Path, datasets: Mapping[str, str]
) -> dict[str, Path]:
    """Resolve configured dataset paths beneath a user-supplied data root."""

    root = Path(data_root).resolve()
    resolved = {}
    for name, relative_path in datasets.items():
        path = (root / relative_path).resolve()
        if root not in path.parents:
            raise ValueError(f"dataset {name!r} resolves outside the data root")
        resolved[name] = path
    return resolved
