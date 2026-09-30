"""Generic sequential batch execution with explicit result records."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


def run_batch(
    items: Iterable[Mapping[str, Any]],
    processor: Callable[[Mapping[str, Any]], str | Path],
    *,
    destination_for: Callable[[Mapping[str, Any]], str | Path] | None = None,
    identifier_for: Callable[[Mapping[str, Any]], str] | None = None,
    overwrite: bool = False,
    continue_on_error: bool = False,
    completed_status: str = "processed",
) -> list[dict[str, Any]]:
    """Process items consistently while recording skipped and failed outcomes."""

    results = []
    for item in items:
        info = dict(item)
        identifier = identifier_for(info) if identifier_for else str(info)
        destination = Path(destination_for(info)) if destination_for else None
        if destination is not None and destination.exists() and not overwrite:
            results.append({**info, "status": "skipped", "path": destination, "error": None})
            continue
        try:
            output = Path(processor(info))
        except Exception as error:
            if not continue_on_error:
                raise RuntimeError(f"Failed to process {identifier}.") from error
            results.append(
                {**info, "status": "failed", "path": destination, "error": str(error)}
            )
        else:
            results.append(
                {**info, "status": completed_status, "path": output, "error": None}
            )
    return results
