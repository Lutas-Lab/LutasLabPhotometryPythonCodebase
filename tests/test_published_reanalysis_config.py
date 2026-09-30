import json
from pathlib import Path
import shutil

import pytest

from src.published_reanalysis_config import (
    load_published_reanalysis_config,
    resolve_dataset_paths,
)


def test_repository_config_loads_and_resolves_below_root():
    config = load_published_reanalysis_config("config/published_reanalysis.json")
    paths = resolve_dataset_paths("C:/Depository Data", config["datasets"])

    assert "figure5_total" in paths
    assert paths["figure5_total"].name.endswith("Finalsave.mat")


def test_config_rejects_parent_traversal():
    folder = Path("tests/_published_config_tmp")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "bad.json"
        datasets = {
            key: "safe.mat"
            for key in load_published_reanalysis_config(
                "config/published_reanalysis.json"
            )["datasets"]
        }
        datasets["figure5_total"] = "../outside.mat"
        path.write_text(
            json.dumps({"schema_version": "1.0", "datasets": datasets}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="safe relative path"):
            load_published_reanalysis_config(path)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
