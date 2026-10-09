"""Compact, machine-readable provenance embedded in processed session files."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .update_check import local_git_commit

PROVENANCE_SCHEMA_VERSION = "1.0"
DEFAULT_PACKAGES = (
    "lutaslab-core",
    "lutaslab-photometry",
    "fluopulse-analysis",
    "iflip3-analysis",
    "numpy",
    "scipy",
    "matplotlib",
    "pynapple",
    "nemos",
)

ProvenanceState = Literal[
    "current",
    "settings_changed",
    "older_schema",
    "different_commit",
    "unknown_version",
    "dirty_code",
    "missing",
]


@dataclass(frozen=True)
class ProcessedProvenanceStatus:
    state: ProvenanceState
    summary: str
    detail: str
    provenance: dict[str, Any] | None = None


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _package_versions(package_names=DEFAULT_PACKAGES) -> dict[str, str | None]:
    versions = {}
    for package in package_names:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_dirty(project_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def parameter_fingerprint(parameters: Mapping[str, Any]) -> str:
    """Return a stable fingerprint for a JSON-compatible parameter mapping."""

    canonical = json.dumps(
        _json_value(parameters),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_processed_provenance(
    workflow: str,
    parameters: Mapping[str, Any],
    *,
    processed_schema_version: str,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Describe the code and settings that produced one processed session."""

    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[4]
    )
    clean_parameters = _json_value(parameters)
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "processed_schema_version": str(processed_schema_version),
        "workflow": str(workflow),
        "processing_utc": datetime.now(UTC).isoformat(),
        "code": {
            "commit": local_git_commit(root),
            "dirty": _git_dirty(root),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "parameters": clean_parameters,
        "parameters_sha256": parameter_fingerprint(clean_parameters),
    }


def provenance_json(provenance: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(provenance), sort_keys=True, separators=(",", ":"))


def read_processed_provenance(path: str | Path) -> dict[str, Any] | None:
    """Read embedded provenance, returning ``None`` for legacy processed files."""

    processed_path = Path(path)
    if not processed_path.is_file():
        return None
    with np.load(processed_path, allow_pickle=False) as data:
        if "provenance_json" in data.files:
            return json.loads(str(data["provenance_json"].item()))
        if "code_commit" not in data.files:
            return None
        commit = str(data["code_commit"].item())
        schema = (
            str(data["processed_schema_version"].item())
            if "processed_schema_version" in data.files
            else "legacy"
        )
        processing_utc = (
            str(data["processing_utc"].item())
            if "processing_utc" in data.files
            else None
        )
    return {
        "provenance_schema_version": "legacy",
        "processed_schema_version": schema,
        "workflow": "conventional",
        "processing_utc": processing_utc,
        "code": {"commit": None if commit == "unknown" else commit, "dirty": None},
        "software": {},
        "parameters": None,
        "parameters_sha256": None,
    }


def classify_processed_provenance(
    path: str | Path,
    *,
    current_commit: str | None,
    expected_schema_version: str,
    expected_parameters: Mapping[str, Any] | None = None,
) -> ProcessedProvenanceStatus:
    """Classify whether a processed file is current enough to reuse."""

    processed_path = Path(path)
    if not processed_path.is_file():
        return ProcessedProvenanceStatus(
            "missing",
            "Not processed",
            "No processed file exists for this session.",
        )
    try:
        record = read_processed_provenance(processed_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return ProcessedProvenanceStatus(
            "unknown_version",
            "Version unknown",
            f"The processed file provenance could not be read: {error}",
        )
    if record is None:
        return ProcessedProvenanceStatus(
            "unknown_version",
            "Legacy or unknown version",
            "This file has no embedded code provenance; consider reprocessing it.",
        )
    if str(record.get("processed_schema_version")) != str(expected_schema_version):
        return ProcessedProvenanceStatus(
            "older_schema",
            "Processed-file schema differs",
            "Reprocessing is required before relying on current downstream workflows.",
            record,
        )
    code = record.get("code") or {}
    if code.get("dirty") is True:
        return ProcessedProvenanceStatus(
            "dirty_code",
            "Uncommitted code was used",
            "The exact code cannot be recovered from the commit alone; "
            "reprocessing is recommended.",
            record,
        )
    if expected_parameters is not None:
        stored_fingerprint = record.get("parameters_sha256")
        if not stored_fingerprint:
            return ProcessedProvenanceStatus(
                "unknown_version",
                "Preprocessing settings unknown",
                "This older file records a commit but not its complete settings; "
                "consider reprocessing it.",
                record,
            )
        expected_fingerprint = parameter_fingerprint(expected_parameters)
        if stored_fingerprint != expected_fingerprint:
            return ProcessedProvenanceStatus(
                "settings_changed",
                "Preprocessing settings differ",
                "Reprocess with overwrite enabled to apply the currently selected settings.",
                record,
            )
    stored_commit = code.get("commit")
    if not stored_commit:
        return ProcessedProvenanceStatus(
            "unknown_version",
            "Code version unknown",
            "The processed file cannot be tied to a Git commit; consider reprocessing it.",
            record,
        )
    if not current_commit:
        return ProcessedProvenanceStatus(
            "unknown_version",
            "Current code version unknown",
            "This repository copy cannot be tied to a Git commit, so versions cannot be compared.",
            record,
        )
    if stored_commit != current_commit:
        return ProcessedProvenanceStatus(
            "different_commit",
            "Different code commit",
            "Review preprocessing changes since this commit; reprocessing is not "
            "automatically required.",
            record,
        )
    return ProcessedProvenanceStatus(
        "current",
        "Current code and settings",
        "The processed file matches the current commit and selected preprocessing settings.",
        record,
    )
