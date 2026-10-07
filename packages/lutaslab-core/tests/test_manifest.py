import shutil
from pathlib import Path

import pytest

from lutaslab_core.manifest import load_session_manifest


def test_manifest_preserves_identifiers_and_extra_columns():
    folder = Path("packages/lutaslab-core/tests/_manifest_tmp")
    path = folder / "sessions.csv"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "mouse,date,run,condition\nM1,001234,002,rewarded\n",
            encoding="utf-8",
        )
        assert load_session_manifest(path) == [
            {"condition": "rewarded", "mouse": "M1", "date": "001234", "run": 2}
        ]
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_manifest_rejects_duplicate_sessions():
    folder = Path("packages/lutaslab-core/tests/_manifest_tmp")
    path = folder / "sessions.csv"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "mouse,date,run\nM1,260101,1\nM1,260101,1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_session_manifest(path)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
