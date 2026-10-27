"""Structured reporting for expected workflow quality-control warnings."""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

from lutaslab_core.nidaq import RunningDataExcludedWarning

QC_WARNING_PREFIX = "QC_WARNING"


def _emit_warning(record: warnings.WarningMessage) -> None:
    warning = record.message
    if isinstance(warning, RunningDataExcludedWarning):
        payload = {
            "kind": "running_data_excluded",
            "reason": warning.reason,
            "path": warning.path or "",
        }
        print(f"{QC_WARNING_PREFIX}\t{json.dumps(payload, separators=(',', ':'))}")
        return
    stream = record.file if record.file is not None else sys.stderr
    stream.write(
        warnings.formatwarning(
            warning,
            record.category,
            record.filename,
            record.lineno,
            line=record.line,
        )
    )


@contextmanager
def concise_qc_warnings() -> Iterator[None]:
    """Encode expected QC warnings while preserving unexpected warning detail."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for record in caught:
                _emit_warning(record)


def parse_qc_warning_line(line: str) -> dict[str, str] | None:
    """Decode a structured QC line emitted by a maintained workflow CLI."""

    prefix = f"{QC_WARNING_PREFIX}\t"
    if not line.startswith(prefix):
        return None
    try:
        payload = json.loads(line[len(prefix) :])
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "running_data_excluded":
        return None
    return {
        "kind": "running_data_excluded",
        "reason": str(payload.get("reason", "Running data failed quality control")),
        "path": str(payload.get("path", "")),
    }
