"""Machine-readable provenance records for analysis runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping


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


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for a run record."""

    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def describe_path(path: str | Path, *, hash_file: bool = True) -> dict[str, Any]:
    """Describe an input or output path without recursively scanning directories."""

    resolved = Path(path).resolve()
    description: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        return description
    stat = resolved.stat()
    description.update(
        {
            "kind": "directory" if resolved.is_dir() else "file",
            "modified_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    )
    if resolved.is_file():
        description["size_bytes"] = stat.st_size
        if hash_file:
            description["sha256"] = _sha256(resolved)
    return description


def _package_versions(package_names: Iterable[str]) -> dict[str, str | None]:
    versions = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_state(repository_root: Path | None) -> dict[str, Any] | None:
    if repository_root is None:
        return None
    root = repository_root.resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"repository_root": str(root), "available": False}
    return {
        "repository_root": str(root),
        "available": True,
        "commit": commit,
        "dirty": bool(status.strip()),
    }


def build_run_record(
    *,
    workflow: str,
    parameters: Mapping[str, Any],
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    repository_root: str | Path | None = None,
    started_at: str | None = None,
    package_names: Iterable[str] = DEFAULT_PACKAGES,
) -> dict[str, Any]:
    """Build a JSON-compatible record of an analysis invocation."""

    root = None if repository_root is None else Path(repository_root)
    return {
        "schema_version": "1.0",
        "workflow": workflow,
        "started_at_utc": started_at or utc_now(),
        "completed_at_utc": utc_now(),
        "parameters": _json_value(parameters),
        "inputs": [describe_path(path) for path in inputs],
        "outputs": [describe_path(path, hash_file=False) for path in outputs],
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(package_names),
            "git": _git_state(root),
        },
    }


def write_run_record(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Build and write a run record, returning the serialized content."""

    record = build_run_record(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
