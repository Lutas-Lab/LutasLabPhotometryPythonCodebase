"""CSV session-manifest loading shared by analysis packages."""

from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_SESSION_COLUMNS = ("mouse", "date", "run")


def load_session_manifest(path: str | Path) -> list[dict[str, object]]:
    """Load a unique mouse/date/run manifest while preserving extra columns."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Session manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Session manifest is missing a header row")
        fieldnames = tuple(name.strip() for name in reader.fieldnames)
        missing = set(REQUIRED_SESSION_COLUMNS).difference(fieldnames)
        if missing:
            raise ValueError(f"Session manifest is missing columns: {sorted(missing)}")
        sessions = []
        seen = set()
        for line_number, row in enumerate(reader, start=2):
            cleaned = {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            if not any(cleaned.values()):
                continue
            mouse, date, run_text = (cleaned.get(name, "") for name in REQUIRED_SESSION_COLUMNS)
            if not mouse or not date or not run_text:
                raise ValueError(
                    f"Session manifest line {line_number} has an empty required value"
                )
            try:
                run = int(run_text)
            except ValueError as error:
                raise ValueError(
                    f"Session manifest line {line_number} has invalid run {run_text!r}"
                ) from error
            if run < 0:
                raise ValueError(
                    f"Session manifest line {line_number} has a negative run number"
                )
            identifier = (mouse, date, run)
            if identifier in seen:
                raise ValueError(
                    "Session manifest contains a duplicate session: "
                    f"{mouse} {date} run {run}"
                )
            seen.add(identifier)
            session = {
                key: value
                for key, value in cleaned.items()
                if key not in REQUIRED_SESSION_COLUMNS and value
            }
            session.update({"mouse": mouse, "date": date, "run": run})
            sessions.append(session)
    if not sessions:
        raise ValueError("Session manifest contains no sessions")
    return sessions
