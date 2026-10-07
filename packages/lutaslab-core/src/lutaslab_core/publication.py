"""Small, portable publication bundles for figure reconstruction.

The bundle format intentionally uses only gzip-compressed CSV, JSON, Markdown,
and SHA-256 checksums.  It is a figure-data export, not a replacement for raw
data or the richer processed-session representation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "1.0"


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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


def _safe_relative_path(value: str | Path) -> Path:
    text = str(value).replace("\\", "/")
    posix = PurePosixPath(text)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"bundle paths must be relative and contained: {value!r}")
    return Path(*posix.parts)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_gzip_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> int:
    """Write a deterministic compressed CSV and return its row count."""

    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, extrasaction="raise")
                writer.writeheader()
                for row in rows:
                    unknown = set(row) - set(fieldnames)
                    if unknown:
                        raise ValueError(f"row contains undeclared columns: {sorted(unknown)}")
                    writer.writerow({name: _json_value(row.get(name)) for name in fieldnames})
                    count += 1
    return count


@dataclass(frozen=True)
class BundleValidation:
    """Result returned by :func:`validate_publication_bundle`."""

    valid: bool
    errors: tuple[str, ...]
    file_count: int
    size_bytes: int


class PublicationBundle:
    """Build a versioned, self-describing directory of figure source data."""

    def __init__(
        self,
        root: str | Path,
        *,
        title: str,
        description: str = "",
        creators: Sequence[str] = (),
        code_url: str | None = None,
        code_version: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "title": title,
            "description": description,
            "creators": list(creators),
            "code": {"url": code_url, "version": code_version},
            "files": [],
            "figures": {},
        }
        self._paths: set[str] = set()

    def _reserve(self, relative_path: str | Path) -> tuple[Path, str]:
        relative = _safe_relative_path(relative_path)
        normalized = relative.as_posix()
        if normalized in self._paths:
            raise ValueError(f"bundle path already exists: {normalized}")
        self._paths.add(normalized)
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination, normalized

    def add_table(
        self,
        name: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        columns: Sequence[str],
        description: str,
        units: Mapping[str, str] | None = None,
        primary_key: Sequence[str] = (),
        panel_ids: Sequence[str] = (),
    ) -> Path:
        """Add a gzip-compressed CSV table.

        ``columns`` is explicit so empty tables remain self-describing and
        accidental column-order changes do not silently alter deposits.
        """

        if not name or not columns or len(set(columns)) != len(columns):
            raise ValueError("name and unique table columns are required")
        unknown_keys = set(primary_key) - set(columns)
        if unknown_keys:
            raise ValueError(f"primary-key columns not present: {sorted(unknown_keys)}")
        destination, relative = self._reserve(Path("tables") / f"{name}.csv.gz")
        row_count = _write_gzip_csv(destination, list(columns), rows)
        self._manifest["files"].append(
            {
                "path": relative,
                "type": "table",
                "format": "csv+gzip",
                "description": description,
                "columns": list(columns),
                "units": dict(units or {}),
                "primary_key": list(primary_key),
                "panel_ids": list(panel_ids),
                "row_count": row_count,
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
        return destination

    def add_json(
        self,
        name: str,
        value: Mapping[str, Any] | Sequence[Any],
        *,
        description: str,
        panel_ids: Sequence[str] = (),
    ) -> Path:
        """Add structured metadata or a model specification as JSON."""

        destination, relative = self._reserve(Path("metadata") / f"{name}.json")
        destination.write_text(
            json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._manifest["files"].append(
            {
                "path": relative,
                "type": "metadata",
                "format": "json",
                "description": description,
                "panel_ids": list(panel_ids),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
        return destination

    def add_figure(
        self,
        panel_id: str,
        *,
        title: str,
        sources: Sequence[str | Path],
        command: str,
        notes: str = "",
    ) -> None:
        """Map one figure or panel to bundle files and a reconstruction command."""

        if panel_id in self._manifest["figures"]:
            raise ValueError(f"figure panel already registered: {panel_id}")
        normalized_sources = [_safe_relative_path(source).as_posix() for source in sources]
        missing = set(normalized_sources) - self._paths
        if missing:
            raise ValueError(f"figure sources have not been added: {sorted(missing)}")
        self._manifest["figures"][panel_id] = {
            "title": title,
            "sources": normalized_sources,
            "command": command,
            "notes": notes,
        }

    def finalize(self, *, readme_notes: str = "") -> Path:
        """Write the manifest, human-readable README, and checksums."""

        if not self._manifest["files"]:
            raise ValueError("a publication bundle must contain at least one data file")
        manifest_path = self.root / "figure_manifest.json"
        manifest_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        readme = [
            f"# {self._manifest['title']}",
            "",
            self._manifest["description"],
            "",
            "This is a slim figure-reconstruction bundle, not the raw acquisition data.",
            "Tables are UTF-8 CSV files compressed with gzip and can be opened by most",
            "statistical software without the original analysis environment.",
            "",
            "## Contents",
            "",
        ]
        for entry in self._manifest["files"]:
            readme.append(f"- `{entry['path']}` — {entry['description']}")
        if self._manifest["figures"]:
            readme.extend(["", "## Figure reconstruction", ""])
            for panel_id, figure in self._manifest["figures"].items():
                readme.append(f"- **{panel_id}: {figure['title']}** — `{figure['command']}`")
        if readme_notes:
            readme.extend(["", "## Additional notes", "", readme_notes.strip()])
        readme.extend(
            [
                "",
                "Run `python -m lutaslab_core.publication PATH --validate` to verify all files.",
                "",
            ]
        )
        (self.root / "README.md").write_text("\n".join(readme), encoding="utf-8")

        checksum_paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        checksum_text = "".join(
            f"{_sha256(path)}  {path.relative_to(self.root).as_posix()}\n"
            for path in checksum_paths
        )
        (self.root / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
        return manifest_path


def validate_publication_bundle(root: str | Path) -> BundleValidation:
    """Validate schema, declared table structure, and every stored checksum."""

    root = Path(root)
    errors: list[str] = []
    manifest_path = root / "figure_manifest.json"
    checksum_path = root / "checksums.sha256"
    if not manifest_path.is_file():
        return BundleValidation(False, ("missing figure_manifest.json",), 0, 0)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BundleValidation(False, (f"invalid figure_manifest.json: {exc}",), 0, 0)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {manifest.get('schema_version')!r}")
    declared_paths: set[str] = set()
    size_bytes = 0
    for entry in manifest.get("files", []):
        relative = entry.get("path", "")
        try:
            path = root / _safe_relative_path(relative)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        declared_paths.add(Path(relative).as_posix())
        if not path.is_file():
            errors.append(f"missing declared file: {relative}")
            continue
        size_bytes += path.stat().st_size
        if _sha256(path) != entry.get("sha256"):
            errors.append(f"manifest checksum mismatch: {relative}")
        if path.stat().st_size != entry.get("size_bytes"):
            errors.append(f"size mismatch: {relative}")
        if entry.get("type") == "table":
            try:
                with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                    reader = csv.reader(stream)
                    header = next(reader)
                    row_count = sum(1 for _ in reader)
            except (OSError, StopIteration, UnicodeDecodeError) as exc:
                errors.append(f"invalid table {relative}: {exc}")
                continue
            if header != entry.get("columns"):
                errors.append(f"column mismatch: {relative}")
            if row_count != entry.get("row_count"):
                errors.append(f"row-count mismatch: {relative}")
    for panel_id, figure in manifest.get("figures", {}).items():
        missing = set(figure.get("sources", [])) - declared_paths
        if missing:
            errors.append(f"{panel_id} has missing sources: {sorted(missing)}")
    if not checksum_path.is_file():
        errors.append("missing checksums.sha256")
    else:
        checked_paths: set[str] = set()
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(checksum_lines, 1):
            try:
                expected, relative = line.split("  ", 1)
                path = root / _safe_relative_path(relative)
            except ValueError:
                errors.append(f"invalid checksum line {line_number}")
                continue
            checked_paths.add(Path(relative).as_posix())
            if not path.is_file() or _sha256(path) != expected:
                errors.append(f"checksum mismatch: {relative}")
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        }
        if checked_paths != actual_paths:
            checksum_missing = sorted(actual_paths - checked_paths)
            extra = sorted(checked_paths - actual_paths)
            if checksum_missing:
                errors.append(
                    f"files absent from checksums.sha256: {checksum_missing}"
                )
            if extra:
                errors.append(f"checksums reference absent files: {extra}")
    return BundleValidation(
        not errors,
        tuple(errors),
        len(declared_paths),
        size_bytes,
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a Lutas Lab publication bundle")
    parser.add_argument("path", type=Path)
    parser.add_argument("--validate", action="store_true", help="Validate the bundle")
    args = parser.parse_args()
    if not args.validate:
        parser.error("specify --validate")
    result = validate_publication_bundle(args.path)
    print(json.dumps({"valid": result.valid, "errors": result.errors}, indent=2))
    raise SystemExit(0 if result.valid else 1)


if __name__ == "__main__":
    _main()
