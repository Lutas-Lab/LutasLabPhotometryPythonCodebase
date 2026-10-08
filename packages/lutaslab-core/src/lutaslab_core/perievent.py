"""Sensor-independent peri-event extraction and normalization."""

from __future__ import annotations

import numpy as np


def validate_window(window: tuple[float, float]) -> tuple[float, float]:
    if len(window) != 2 or window[0] >= window[1]:
        raise ValueError("window must contain increasing start and end values")
    return float(window[0]), float(window[1])


def peri_time(window: tuple[float, float], dt: float) -> np.ndarray:
    start, end = validate_window(window)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    count = int(np.floor((end - start) / dt + 0.5))
    return start + np.arange(count + 1, dtype=float) * dt


def valid_event_indices(
    event_times: np.ndarray,
    recording_bounds: tuple[float, float],
    window: tuple[float, float],
) -> np.ndarray:
    """Return source-event indices whose complete window is recorded."""

    events = np.asarray(event_times, dtype=float)
    if events.ndim != 1:
        raise ValueError("event_times must be one-dimensional")
    if len(recording_bounds) != 2 or recording_bounds[0] >= recording_bounds[1]:
        raise ValueError("recording_bounds must contain increasing start and end values")
    start, end = validate_window(window)
    recording_start, recording_end = map(float, recording_bounds)
    return np.flatnonzero(
        np.isfinite(events)
        & (events + start >= recording_start)
        & (events + end <= recording_end)
    )


def extract_perievent_trials(
    signal_time: np.ndarray,
    signal: np.ndarray,
    event_times: np.ndarray,
    window: tuple[float, float] = (-5.0, 10.0),
    dt: float = 0.02,
    *,
    require_complete: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a continuous signal around events with complete windows."""

    signal_time = np.asarray(signal_time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if signal_time.ndim != 1 or signal.ndim != 1:
        raise ValueError("signal_time and signal must be one-dimensional")
    if signal_time.size != signal.size or signal_time.size < 2:
        raise ValueError("signal_time and signal must have equal nontrivial lengths")
    if not np.all(np.isfinite(signal_time)) or not np.all(np.diff(signal_time) > 0):
        raise ValueError("signal_time must be finite and strictly increasing")
    if np.any(np.isinf(signal)):
        raise ValueError("signal must not contain infinite values")

    relative_time = peri_time(window, dt)
    events = np.asarray(event_times, dtype=float)
    if require_complete:
        valid = valid_event_indices(
            events,
            (float(signal_time[0]), float(signal_time[-1])),
            window,
        )
    else:
        valid = np.flatnonzero(np.isfinite(events))
    trials = np.full((valid.size, relative_time.size), np.nan, dtype=float)
    for row, event_index in enumerate(valid):
        query_time = events[event_index] + relative_time
        inside = (query_time >= signal_time[0]) & (query_time <= signal_time[-1])
        trials[row, inside] = np.interp(query_time[inside], signal_time, signal)
    return relative_time, trials, valid


def extract_perievent_event_rate(
    event_times: np.ndarray,
    alignment_times: np.ndarray,
    recording_bounds: tuple[float, float] | None = None,
    window: tuple[float, float] = (-5.0, 10.0),
    dt: float = 0.1,
    *,
    require_complete: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin discrete events as rates around alignment events."""

    events = np.asarray(event_times, dtype=float)
    alignments = np.asarray(alignment_times, dtype=float)
    if events.ndim != 1 or alignments.ndim != 1:
        raise ValueError("event_times and alignment_times must be one-dimensional")
    start, end = validate_window(window)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    n_bins = int(np.floor((end - start) / dt + 0.5))
    if n_bins < 1 or not np.isclose(n_bins * dt, end - start):
        raise ValueError("The event-rate window duration must be divisible by dt")
    edges = start + np.arange(n_bins + 1, dtype=float) * dt
    relative_time = edges[:-1] + dt / 2
    if recording_bounds is None or not require_complete:
        valid = np.flatnonzero(np.isfinite(alignments))
    else:
        valid = valid_event_indices(alignments, recording_bounds, window)
    finite_events = events[np.isfinite(events)]
    trials = np.empty((valid.size, n_bins), dtype=float)
    for row, alignment_index in enumerate(valid):
        trials[row] = (
            np.histogram(finite_events - alignments[alignment_index], bins=edges)[0] / dt
        )
        if recording_bounds is not None and not require_complete:
            absolute_centers = alignments[alignment_index] + relative_time
            outside = (absolute_centers < recording_bounds[0]) | (
                absolute_centers > recording_bounds[1]
            )
            trials[row, outside] = np.nan
    return relative_time, trials, valid


def normalize_trials(
    peri_times: np.ndarray,
    trials: np.ndarray,
    normalization: str | None = "zscore",
    baseline: tuple[float, float] | None = (-5.0, 0.0),
) -> np.ndarray:
    """Apply trial-local baseline subtraction or z-scoring."""

    peri_times = np.asarray(peri_times, dtype=float)
    trials = np.asarray(trials, dtype=float)
    if peri_times.ndim != 1 or trials.ndim != 2:
        raise ValueError("peri_times and trials must be one- and two-dimensional")
    if trials.shape[1] != peri_times.size:
        raise ValueError("trials columns must match peri_times")
    if normalization in (None, "none"):
        return trials.copy()
    if normalization not in ("subtract", "zscore"):
        raise ValueError("normalization must be 'none', 'subtract', or 'zscore'")
    if baseline is None or len(baseline) != 2 or baseline[0] >= baseline[1]:
        raise ValueError("baseline must contain increasing start and end values")
    baseline_mask = (peri_times >= baseline[0]) & (peri_times < baseline[1])
    if not np.any(baseline_mask):
        raise ValueError("baseline does not overlap the peri-event time vector")
    baseline_values = trials[:, baseline_mask]
    baseline_mean = np.nanmean(baseline_values, axis=1, keepdims=True)
    output = trials - baseline_mean
    if normalization == "zscore":
        baseline_std = np.nanstd(baseline_values, axis=1, ddof=1, keepdims=True)
        baseline_std[(baseline_std <= 0) | ~np.isfinite(baseline_std)] = np.nan
        output = output / baseline_std
    return output


def summarize_trials(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-timepoint mean and SEM of a trial-by-time matrix."""

    trials = np.asarray(trials, dtype=float)
    if trials.ndim != 2:
        raise ValueError("trials must be a two-dimensional trial-by-time array")
    if trials.shape[0] == 0:
        empty = np.full(trials.shape[1], np.nan)
        return empty, empty.copy()
    count = np.sum(np.isfinite(trials), axis=0)
    mean = np.full(trials.shape[1], np.nan, dtype=float)
    np.divide(np.nansum(trials, axis=0), count, out=mean, where=count > 0)
    sem = np.full(trials.shape[1], np.nan, dtype=float)
    enough = count > 1
    if np.any(enough):
        sem[enough] = np.nanstd(trials[:, enough], axis=0, ddof=1) / np.sqrt(count[enough])
    return mean, sem


def generate_null_onsets(
    event_times: np.ndarray,
    onset_bounds: tuple[float, float],
    *,
    n_shuffles: int = 500,
    method: str = "random_onsets",
    exclusion: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate session-local random or circularly shifted event onsets."""

    events = np.asarray(event_times, dtype=float)
    events = events[np.isfinite(events)]
    if events.ndim != 1 or events.size == 0:
        raise ValueError("event_times must contain at least one finite event")
    if len(onset_bounds) != 2 or onset_bounds[0] >= onset_bounds[1]:
        raise ValueError("onset_bounds must contain increasing start and end values")
    if not isinstance(n_shuffles, int) or isinstance(n_shuffles, bool) or n_shuffles < 1:
        raise ValueError("n_shuffles must be a positive integer")
    if method not in ("random_onsets", "circular_shift"):
        raise ValueError("method must be 'random_onsets' or 'circular_shift'")
    if not np.isfinite(exclusion) or exclusion < 0:
        raise ValueError("exclusion must be finite and nonnegative")
    low, high = map(float, onset_bounds)
    rng = np.random.default_rng() if rng is None else rng
    output = np.empty((n_shuffles, events.size), dtype=float)

    def sufficiently_far(candidates):
        if exclusion == 0:
            return np.ones(len(candidates), dtype=bool)
        return np.all(np.abs(candidates[:, None] - events[None, :]) >= exclusion, axis=1)

    if method == "random_onsets":
        for shuffle_index in range(n_shuffles):
            selected = []
            for _ in range(1000):
                candidates = rng.uniform(low, high, size=max(64, 2 * events.size))
                selected.extend(candidates[sufficiently_far(candidates)].tolist())
                if len(selected) >= events.size:
                    break
            if len(selected) < events.size:
                raise ValueError("Could not sample enough random onsets. Reduce null exclusion")
            output[shuffle_index] = selected[: events.size]
        return output

    span = high - low
    wrapped = low + np.mod(events - low, span)
    for shuffle_index in range(n_shuffles):
        for _ in range(10000):
            candidates = low + np.mod(wrapped - low + rng.uniform(0.0, span), span)
            if np.all(sufficiently_far(candidates)):
                output[shuffle_index] = candidates
                break
        else:
            raise ValueError("Could not find an eligible circular shift. Reduce null exclusion")
    return output
