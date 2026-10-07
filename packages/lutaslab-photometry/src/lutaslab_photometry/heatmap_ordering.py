"""Reusable behavioral ordering for peri-event heatmap rows."""

from __future__ import annotations

import numpy as np

HEATMAP_SORT_METHODS = (
    "event_order",
    "response_mean",
    "ensure_latency",
    "first_lick_latency",
    "first_bout_latency",
    "bout_size",
    "bout_duration",
    "post_event_lick_count",
    "pre_event_lick_rate",
    "cue_to_bout_latency",
)

HEATMAP_SORT_LABELS = {
    "event_order": "Recorded event order",
    "response_mean": "Photometry response magnitude",
    "ensure_latency": "Ensure delivery latency",
    "first_lick_latency": "First-lick latency",
    "first_bout_latency": "First lick-bout latency",
    "bout_size": "Lick-bout size",
    "bout_duration": "Lick-bout duration",
    "post_event_lick_count": "Post-event lick count",
    "pre_event_lick_rate": "Pre-event lick rate",
    "cue_to_bout_latency": "Cue-to-bout latency",
}

_DEFAULT_WINDOWS = {
    "ensure_latency": (0.0, 20.0),
    "first_lick_latency": (0.0, 20.0),
    "first_bout_latency": (0.0, 20.0),
    "bout_size": (0.0, 20.0),
    "bout_duration": (0.0, 20.0),
    "post_event_lick_count": (0.0, 20.0),
    "pre_event_lick_rate": (-5.0, 0.0),
    "cue_to_bout_latency": (-20.0, 0.0),
}

_AUTO_DESCENDING = {
    "response_mean",
    "bout_size",
    "bout_duration",
    "post_event_lick_count",
    "pre_event_lick_rate",
}


def default_heatmap_sort_window(method):
    """Return the documented default matching window for one ordering."""
    if method not in HEATMAP_SORT_METHODS:
        raise ValueError(f"Unknown heatmap sort method: {method}")
    return _DEFAULT_WINDOWS.get(method)


def _validate_window(method, window):
    if window is None:
        window = default_heatmap_sort_window(method)
    if window is None:
        return None
    if len(window) != 2 or not np.all(np.isfinite(window)) or window[0] >= window[1]:
        raise ValueError("heatmap sort window must contain increasing finite values.")
    return float(window[0]), float(window[1])


def _first_event(events, alignment, window):
    relative = np.asarray(events, dtype=float) - float(alignment)
    eligible = np.flatnonzero((relative >= window[0]) & (relative <= window[1]))
    if eligible.size == 0:
        return None, np.nan
    index = int(eligible[np.argmin(relative[eligible])])
    return index, float(relative[index])


def _previous_event(events, alignment, window):
    relative = np.asarray(events, dtype=float) - float(alignment)
    eligible = np.flatnonzero((relative >= window[0]) & (relative <= window[1]))
    if eligible.size == 0:
        return None, np.nan
    index = int(eligible[np.argmax(relative[eligible])])
    return index, float(relative[index])


def heatmap_sort_values(session_result, peri_time, method, sort_window=None):
    """Calculate one behavioral or signal-derived sort value per heatmap row."""
    if method not in HEATMAP_SORT_METHODS:
        raise ValueError(f"Unknown heatmap sort method: {method}")
    trials = np.asarray(session_result["trials"], dtype=float)
    alignments = np.asarray(session_result["alignment_times"], dtype=float)
    if trials.ndim != 2 or alignments.shape != (trials.shape[0],):
        raise ValueError("Heatmap trials and alignment times do not match.")
    events = session_result.get("behavioral_events", {})
    window = _validate_window(method, sort_window)
    values = np.full(trials.shape[0], np.nan, dtype=float)
    matched_times = np.full(trials.shape[0], np.nan, dtype=float)

    if method == "event_order":
        values = np.arange(trials.shape[0], dtype=float)
    elif method == "response_mean":
        response_mask = np.asarray(peri_time, dtype=float) >= 0
        if not np.any(response_mask):
            response_mask = np.ones(trials.shape[1], dtype=bool)
        values = np.nanmean(trials[:, response_mask], axis=1)
    elif method in {
        "ensure_latency",
        "first_lick_latency",
        "first_bout_latency",
        "bout_size",
        "bout_duration",
    }:
        event_key = {
            "ensure_latency": "solenoid_onset",
            "first_lick_latency": "lick_times",
            "first_bout_latency": "lick_bout_onset",
            "bout_size": "lick_bout_onset",
            "bout_duration": "lick_bout_onset",
        }[method]
        event_times = np.asarray(events.get(event_key, []), dtype=float)
        bout_measure = None
        if method == "bout_size":
            bout_measure = np.asarray(events.get("lick_bout_lick_count", []), dtype=float)
        elif method == "bout_duration":
            bout_measure = np.asarray(events.get("lick_bout_duration", []), dtype=float)
        if bout_measure is not None and bout_measure.shape != event_times.shape:
            raise ValueError(f"{method} values do not match lick_bout_onset.")
        for row, alignment in enumerate(alignments):
            index, latency = _first_event(event_times, alignment, window)
            if index is None:
                continue
            matched_times[row] = event_times[index]
            values[row] = latency if bout_measure is None else bout_measure[index]
    elif method in {"post_event_lick_count", "pre_event_lick_rate"}:
        lick_times = np.asarray(events.get("lick_times", []), dtype=float)
        duration = window[1] - window[0]
        for row, alignment in enumerate(alignments):
            relative = lick_times - alignment
            count = np.sum((relative >= window[0]) & (relative <= window[1]))
            values[row] = float(count) if method == "post_event_lick_count" else count / duration
    elif method == "cue_to_bout_latency":
        cue_times = np.asarray(events.get("cue_onset", []), dtype=float)
        for row, alignment in enumerate(alignments):
            index, _ = _previous_event(cue_times, alignment, window)
            if index is None:
                continue
            matched_times[row] = cue_times[index]
            values[row] = float(alignment - cue_times[index])

    return {
        "values": values,
        "matched_event_times": matched_times,
        "matched": np.isfinite(values),
        "sort_window": window,
    }


def order_heatmap_trials(
    session_result,
    peri_time,
    *,
    sort="event_order",
    sort_window=None,
    direction="auto",
    unmatched="bottom",
):
    """Return ordered trials and audit metadata for Python or notebook use."""
    if direction not in ("auto", "ascending", "descending"):
        raise ValueError("direction must be 'auto', 'ascending', or 'descending'.")
    if unmatched not in ("bottom", "exclude"):
        raise ValueError("unmatched must be 'bottom' or 'exclude'.")
    computed = heatmap_sort_values(session_result, peri_time, sort, sort_window)
    values = computed["values"]
    finite = np.flatnonzero(np.isfinite(values))
    missing = np.flatnonzero(~np.isfinite(values))
    descending = sort in _AUTO_DESCENDING if direction == "auto" else direction == "descending"
    order = finite[np.argsort(values[finite], kind="stable")]
    if descending:
        order = order[::-1]
    if unmatched == "bottom":
        order = np.concatenate((order, missing))
    trials = np.asarray(session_result["trials"], dtype=float)
    return {
        **computed,
        "trials": trials[order],
        "order": order,
        "ordered_values": values[order],
        "ordered_matched_event_times": computed["matched_event_times"][order],
        "direction": "descending" if descending else "ascending",
        "unmatched": unmatched,
        "sort": sort,
    }
