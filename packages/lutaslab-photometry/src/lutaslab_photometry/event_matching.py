"""Derived behavioral alignment events shared across acquisition systems."""

from __future__ import annotations

import numpy as np

DERIVED_ALIGNMENT_EVENTS = (
    "first_lick_after_ensure",
    "first_lick_bout_after_ensure",
)


def first_events_after_references(reference_times, event_times, window=(0.0, 20.0)):
    """Match one first event to each reference without crossing the next reference."""

    references = np.asarray(reference_times, dtype=float)
    events = np.asarray(event_times, dtype=float)
    if references.ndim != 1 or events.ndim != 1:
        raise ValueError("Reference and event times must be one-dimensional.")
    if np.any(~np.isfinite(references)) or np.any(~np.isfinite(events)):
        raise ValueError("Reference and event times must be finite.")
    if len(window) != 2 or not np.all(np.isfinite(window)) or window[0] < 0:
        raise ValueError("Derived-event window must contain finite nonnegative values.")
    start, stop = map(float, window)
    if start >= stop:
        raise ValueError("Derived-event window start must be less than its end.")
    references = np.sort(references)
    events = np.sort(events)
    matched = []
    matched_reference_indices = []
    for index, reference in enumerate(references):
        upper = reference + stop
        if index + 1 < references.size:
            upper = min(upper, references[index + 1])
        candidates = events[(events >= reference + start) & (events < upper)]
        if candidates.size:
            matched.append(float(candidates[0]))
            matched_reference_indices.append(index)
    return np.asarray(matched, dtype=float), np.asarray(
        matched_reference_indices, dtype=int
    )


def derived_alignment_times(event_name, ensure_times, lick_times, bout_onsets, window):
    """Return derived alignment times and the indices of matched Ensure events."""

    if event_name not in DERIVED_ALIGNMENT_EVENTS:
        raise ValueError(f"Unsupported derived alignment event: {event_name}")
    candidates = (
        lick_times if event_name == "first_lick_after_ensure" else bout_onsets
    )
    return first_events_after_references(ensure_times, candidates, window)
