import json
from pathlib import Path

import numpy as np

import lutaslab_photometry.processed_provenance as provenance


def _write_processed(path, record, schema="1.0"):
    np.savez_compressed(
        path,
        processed_schema_version=np.asarray(schema),
        provenance_json=np.asarray(provenance.provenance_json(record)),
    )


def _record(parameters=None, *, commit="a" * 40, dirty=False, schema="1.0"):
    parameters = parameters or {"threshold": 1.5}
    return {
        "provenance_schema_version": "1.0",
        "processed_schema_version": schema,
        "workflow": "test",
        "processing_utc": "2026-10-09T12:00:00+00:00",
        "code": {"commit": commit, "dirty": dirty},
        "software": {},
        "parameters": parameters,
        "parameters_sha256": provenance.parameter_fingerprint(parameters),
    }


def test_build_processed_provenance_records_commit_dirty_state_and_settings(monkeypatch):
    monkeypatch.setattr(provenance, "local_git_commit", lambda root: "b" * 40)
    monkeypatch.setattr(provenance, "_git_dirty", lambda root: False)

    record = provenance.build_processed_provenance(
        "iflip3",
        {"tau1": 0.7},
        processed_schema_version="1.0",
        project_root=".",
    )

    assert record["code"] == {"commit": "b" * 40, "dirty": False}
    assert record["parameters"] == {"tau1": 0.7}
    assert record["parameters_sha256"] == provenance.parameter_fingerprint({"tau1": 0.7})
    json.dumps(record)


def test_classify_processed_provenance_distinguishes_reprocessing_reasons():
    path = Path("packages/lutaslab-photometry/tests/_provenance-processed.npz")
    parameters = {"threshold": 1.5}
    commit = "a" * 40
    try:
        _write_processed(path, _record(parameters, commit=commit))

        current = provenance.classify_processed_provenance(
            path,
            current_commit=commit,
            expected_schema_version="1.0",
            expected_parameters=parameters,
        )
        assert current.state == "current"

        changed = provenance.classify_processed_provenance(
            path,
            current_commit=commit,
            expected_schema_version="1.0",
            expected_parameters={"threshold": 2.0},
        )
        assert changed.state == "settings_changed"

        different_commit = provenance.classify_processed_provenance(
            path,
            current_commit="c" * 40,
            expected_schema_version="1.0",
            expected_parameters=parameters,
        )
        assert different_commit.state == "different_commit"

        older_schema = provenance.classify_processed_provenance(
            path,
            current_commit=commit,
            expected_schema_version="2.0",
            expected_parameters=parameters,
        )
        assert older_schema.state == "older_schema"

        _write_processed(path, _record(parameters, commit=commit, dirty=True))
        dirty = provenance.classify_processed_provenance(
            path,
            current_commit=commit,
            expected_schema_version="1.0",
            expected_parameters=parameters,
        )
        assert dirty.state == "dirty_code"
    finally:
        path.unlink(missing_ok=True)


def test_classify_processed_provenance_marks_legacy_file_unknown():
    path = Path("packages/lutaslab-photometry/tests/_provenance-legacy.npz")
    try:
        np.savez_compressed(path, processed_schema_version=np.asarray("1.0"))

        status = provenance.classify_processed_provenance(
            path,
            current_commit="a" * 40,
            expected_schema_version="1.0",
            expected_parameters={},
        )

        assert status.state == "unknown_version"
    finally:
        path.unlink(missing_ok=True)
