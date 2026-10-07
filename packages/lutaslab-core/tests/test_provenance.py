import json
import shutil
from pathlib import Path

from lutaslab_core.provenance import build_run_record, describe_path, write_run_record


def _test_folder() -> Path:
    return Path("packages/lutaslab-core/tests/_provenance_tmp")


def test_describe_path_hashes_file():
    folder = _test_folder()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "input.txt"
        source.write_text("photometry", encoding="utf-8")
        description = describe_path(source)

        assert description["kind"] == "file"
        assert description["size_bytes"] == 10
        assert len(description["sha256"]) == 64
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_build_run_record_serializes_paths_and_missing_outputs():
    folder = _test_folder()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "input.txt"
        source.write_text("data", encoding="utf-8")
        output = folder / "future.csv"
        record = build_run_record(
            workflow="test",
            parameters={"source": source, "folds": 5},
            inputs=[source],
            outputs=[output],
            started_at="2026-01-01T00:00:00+00:00",
            package_names=(),
        )

        assert record["schema_version"] == "1.0"
        assert record["parameters"]["source"] == str(source.resolve())
        assert record["inputs"][0]["exists"] is True
        assert record["outputs"][0]["exists"] is False
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_write_run_record_creates_json():
    folder = _test_folder()
    try:
        destination = folder / "run_metadata.json"
        write_run_record(
            destination,
            workflow="test",
            parameters={},
            inputs=[],
            outputs=[],
            package_names=(),
        )

        assert json.loads(destination.read_text(encoding="utf-8"))["workflow"] == "test"
    finally:
        shutil.rmtree(folder, ignore_errors=True)
