"""Canonical event representations and edge detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_vector(values: np.ndarray, name: str, *, increasing: bool = False) -> np.ndarray:
    result = np.atleast_1d(np.asarray(values, dtype=float).squeeze())
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if increasing and result.size > 1 and np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return result


@dataclass(frozen=True)
class TTLPulses:
    """Complete high pulses represented in sample and time coordinates."""

    onset_times: np.ndarray
    offset_times: np.ndarray
    rising_indices: np.ndarray
    falling_indices: np.ndarray

    def __post_init__(self) -> None:
        onset = _finite_vector(self.onset_times, "onset_times")
        offset = _finite_vector(self.offset_times, "offset_times")
        rising = np.atleast_1d(np.asarray(self.rising_indices, dtype=int).squeeze())
        falling = np.atleast_1d(np.asarray(self.falling_indices, dtype=int).squeeze())
        lengths = {onset.size, offset.size, rising.size, falling.size}
        if len(lengths) != 1:
            raise ValueError("TTL pulse arrays must have matching lengths")
        if np.any(offset < onset) or np.any(rising < 0) or np.any(falling < rising):
            raise ValueError("TTL pulse offsets must not precede onsets")
        object.__setattr__(self, "onset_times", onset)
        object.__setattr__(self, "offset_times", offset)
        object.__setattr__(self, "rising_indices", rising)
        object.__setattr__(self, "falling_indices", falling)

    @classmethod
    def empty(cls) -> TTLPulses:
        empty_time = np.array([], dtype=float)
        empty_index = np.array([], dtype=int)
        return cls(empty_time, empty_time.copy(), empty_index, empty_index.copy())

    @property
    def durations(self) -> np.ndarray:
        return self.offset_times - self.onset_times


@dataclass(frozen=True)
class LickBouts:
    """Groups of consecutive licks separated by no more than a configured gap."""

    onset_times: np.ndarray
    offset_times: np.ndarray
    lick_counts: np.ndarray
    start_indices: np.ndarray
    stop_indices: np.ndarray

    @property
    def durations(self) -> np.ndarray:
        return self.offset_times - self.onset_times


@dataclass(frozen=True)
class SampledBoutFeatures:
    """Lick-bout features represented on a regular sample grid."""

    onset_counts: np.ndarray
    duration_at_onset_seconds: np.ndarray
    occupancy: np.ndarray


def find_ttl_pulses(
    signal: np.ndarray,
    timestamps: np.ndarray | None = None,
    *,
    threshold: float | None = 1.5,
    min_width_seconds: float | None = None,
    max_width_seconds: float | None = None,
    include_boundary_pulses: bool = True,
) -> TTLPulses:
    """Detect complete TTL-high pulses, including pulses touching either boundary."""

    signal = _finite_vector(signal, "signal")
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples")
    if timestamps is None:
        timestamps = np.arange(signal.size, dtype=float)
    timestamps = _finite_vector(timestamps, "timestamps", increasing=True)
    if timestamps.size != signal.size:
        raise ValueError("signal and timestamps must have matching lengths")
    if threshold is None:
        threshold = 0.5 * (float(np.min(signal)) + float(np.max(signal)))
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if min_width_seconds is not None and min_width_seconds < 0:
        raise ValueError("min_width_seconds must be nonnegative")
    if max_width_seconds is not None and max_width_seconds < 0:
        raise ValueError("max_width_seconds must be nonnegative")
    if (
        min_width_seconds is not None
        and max_width_seconds is not None
        and min_width_seconds > max_width_seconds
    ):
        raise ValueError("min_width_seconds cannot exceed max_width_seconds")

    high = signal > threshold
    transitions = np.diff(high.astype(np.int8))
    rising = np.flatnonzero(transitions == 1) + 1
    falling = np.flatnonzero(transitions == -1) + 1
    if include_boundary_pulses and high[0]:
        rising = np.insert(rising, 0, 0)
    if include_boundary_pulses and high[-1]:
        falling = np.append(falling, signal.size - 1)

    pairs: list[tuple[int, int]] = []
    falling_position = 0
    for start in rising:
        while falling_position < falling.size and falling[falling_position] <= start:
            falling_position += 1
        if falling_position < falling.size:
            pairs.append((int(start), int(falling[falling_position])))
            falling_position += 1
    if not pairs:
        return TTLPulses.empty()

    rising = np.asarray([pair[0] for pair in pairs], dtype=int)
    falling = np.asarray([pair[1] for pair in pairs], dtype=int)
    onset = timestamps[rising]
    offset = timestamps[falling]
    widths = offset - onset
    keep = np.ones(widths.size, dtype=bool)
    if min_width_seconds is not None:
        keep &= widths >= min_width_seconds
    if max_width_seconds is not None:
        keep &= widths <= max_width_seconds
    return TTLPulses(onset[keep], offset[keep], rising[keep], falling[keep])


def find_lick_bouts(
    lick_times: np.ndarray,
    *,
    max_interlick_gap_seconds: float = 1.0,
    min_licks: int = 5,
) -> LickBouts:
    """Group ordered lick timestamps into bouts."""

    lick_times = _finite_vector(lick_times, "lick_times", increasing=True)
    if max_interlick_gap_seconds <= 0:
        raise ValueError("max_interlick_gap_seconds must be positive")
    if min_licks < 1:
        raise ValueError("min_licks must be at least one")
    if lick_times.size == 0:
        empty_float = np.array([], dtype=float)
        empty_int = np.array([], dtype=int)
        return LickBouts(
            empty_float,
            empty_float.copy(),
            empty_int,
            empty_int.copy(),
            empty_int.copy(),
        )
    starts = np.r_[0, np.flatnonzero(np.diff(lick_times) > max_interlick_gap_seconds) + 1]
    stops = np.r_[starts[1:], lick_times.size]
    counts = stops - starts
    keep = counts >= min_licks
    return LickBouts(
        onset_times=lick_times[starts[keep]],
        offset_times=lick_times[stops[keep] - 1],
        lick_counts=counts[keep],
        start_indices=starts[keep],
        stop_indices=stops[keep],
    )


def sample_lick_bout_features(
    lick_times: np.ndarray,
    sample_time: np.ndarray,
    *,
    max_interlick_gap_seconds: float = 1.0,
    min_licks: int = 3,
) -> SampledBoutFeatures:
    """Represent bout onsets, durations, and occupancy on a regular time grid."""

    sample_time = _finite_vector(sample_time, "sample_time", increasing=True)
    if sample_time.size < 2:
        raise ValueError("sample_time must contain at least two samples")
    dt = float(np.median(np.diff(sample_time)))
    if not np.allclose(np.diff(sample_time), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("sample_time must be regularly sampled")
    bouts = find_lick_bouts(
        lick_times,
        max_interlick_gap_seconds=max_interlick_gap_seconds,
        min_licks=min_licks,
    )
    onset_counts = np.zeros(sample_time.size, dtype=float)
    duration = np.zeros(sample_time.size, dtype=float)
    occupancy = np.zeros(sample_time.size, dtype=float)
    edges = np.r_[sample_time - dt / 2.0, sample_time[-1] + dt / 2.0]
    for onset, offset, bout_duration in zip(
        bouts.onset_times,
        bouts.offset_times,
        bouts.durations,
        strict=True,
    ):
        index = int(np.searchsorted(edges, onset, side="right") - 1)
        if 0 <= index < sample_time.size:
            onset_counts[index] += 1.0
            duration[index] += float(bout_duration)
        stop = int(np.searchsorted(edges, offset, side="right") - 1)
        first = max(0, index)
        last = min(sample_time.size - 1, stop)
        if first <= last:
            occupancy[first : last + 1] = 1.0
    return SampledBoutFeatures(onset_counts, duration, occupancy)
