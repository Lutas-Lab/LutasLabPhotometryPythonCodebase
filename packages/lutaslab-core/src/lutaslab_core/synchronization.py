"""Affine alignment between acquisition clocks using ordered shared pulses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ClockAlignment:
    intercept_seconds: float
    scale: float
    residuals_seconds: np.ndarray
    source_times: np.ndarray
    target_times: np.ndarray
    source_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    target_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    unmatched_source_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    unmatched_target_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )

    def source_to_target(self, times: np.ndarray) -> np.ndarray:
        return self.intercept_seconds + self.scale * np.asarray(times, dtype=float)

    def target_to_source(self, times: np.ndarray) -> np.ndarray:
        return (np.asarray(times, dtype=float) - self.intercept_seconds) / self.scale

    @property
    def matched_pulses(self) -> int:
        return self.source_times.size

    @property
    def drift_ppm(self) -> float:
        return (self.scale - 1.0) * 1e6

    @property
    def rms_residual_seconds(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_seconds**2)))

    @property
    def max_residual_seconds(self) -> float:
        return float(np.max(np.abs(self.residuals_seconds)))


def _pulse_ordinals(
    times: np.ndarray,
    period: float,
    *,
    interval_jitter_fraction: float,
    label: str,
) -> np.ndarray:
    """Assign pulse numbers while allowing quantization and missing pulses."""

    intervals = np.diff(times)
    ratios = intervals / period
    # Pulse times often come from decimal sample grids (for example 0.1 s),
    # whose binary representation can land several ulps to either side of an
    # exact half-period boundary. Use a small dimensionless tolerance here.
    epsilon = 1e-6
    steps = np.empty(intervals.size, dtype=int)
    for index, ratio in enumerate(ratios):
        if ratio < 0.5 - epsilon:
            # A second edge much sooner than one period is a duplicate candidate.
            step = 0
        elif ratio <= 1.0 + interval_jitter_fraction + epsilon:
            # Prefer one pulse at the upper boundary. This is important for a
            # 5 Hz marker quantized onto the 10 Hz iFLIP sample grid: 0.3 s is
            # ordinary +/- one-sample jitter around the 0.2 s period.
            step = 1
        else:
            step = int(np.floor(ratio + 0.5))
            if step < 1 or abs(ratio - step) > interval_jitter_fraction + epsilon:
                raise ValueError(
                    f"{label} pulse interval after index {index} is "
                    f"{intervals[index]:.9g} s ({ratio:.3f} nominal periods); "
                    "it cannot be explained by timing quantization or whole "
                    "missing pulses"
                )
        steps[index] = step
    return np.concatenate((np.array([0], dtype=int), np.cumsum(steps)))


def _match_by_ordinal(
    source: np.ndarray,
    target: np.ndarray,
    source_period: float,
    target_period: float,
    *,
    interval_jitter_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match ordered periodic pulses, skipping missing or duplicate entries."""

    source_ordinals = _pulse_ordinals(
        source,
        source_period,
        interval_jitter_fraction=interval_jitter_fraction,
        label="Source",
    )
    target_ordinals = _pulse_ordinals(
        target,
        target_period,
        interval_jitter_fraction=interval_jitter_fraction,
        label="Target",
    )
    common = np.intersect1d(source_ordinals, target_ordinals, assume_unique=False)
    scale_guess = target_period / source_period
    source_indices: list[int] = []
    target_indices: list[int] = []
    for ordinal in common:
        source_candidates = np.flatnonzero(source_ordinals == ordinal)
        target_candidates = np.flatnonzero(target_ordinals == ordinal)
        best = min(
            (
                abs(
                    (target[target_index] - target[0])
                    - scale_guess * (source[source_index] - source[0])
                ),
                int(source_index),
                int(target_index),
            )
            for source_index in source_candidates
            for target_index in target_candidates
        )
        source_indices.append(best[1])
        target_indices.append(best[2])
    source_indices_array = np.asarray(source_indices, dtype=int)
    target_indices_array = np.asarray(target_indices, dtype=int)
    unmatched_source = np.setdiff1d(
        np.arange(source.size), source_indices_array, assume_unique=True
    )
    unmatched_target = np.setdiff1d(
        np.arange(target.size), target_indices_array, assume_unique=True
    )
    return (
        source_indices_array,
        target_indices_array,
        unmatched_source,
        unmatched_target,
    )


def fit_clock_alignment(
    source_times: np.ndarray,
    target_times: np.ndarray,
    *,
    source_start_index: int = 0,
    target_start_index: int = 0,
    allow_unmatched_pulses: bool = False,
    interval_jitter_fraction: float = 0.25,
    max_rms_residual_fraction: float | None = None,
    max_residual_fraction: float | None = None,
    max_abs_drift_ppm: float | None = None,
) -> ClockAlignment:
    """Fit ``target = intercept + scale * source`` to ordered shared pulses."""

    source = np.atleast_1d(np.asarray(source_times, dtype=float).squeeze())
    target = np.atleast_1d(np.asarray(target_times, dtype=float).squeeze())
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Pulse times must be one-dimensional")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Pulse times must be finite")
    if source_start_index < 0 or target_start_index < 0:
        raise ValueError("Pulse start indices must be nonnegative")
    if not 0 <= interval_jitter_fraction < 1:
        raise ValueError("interval_jitter_fraction must be between 0 and 1")
    source = source[source_start_index:]
    target = target[target_start_index:]
    if min(source.size, target.size) < 3:
        raise ValueError("At least three matching pulses are required")
    if np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0):
        raise ValueError("Pulse times must be strictly increasing")
    source_interval = float(np.median(np.diff(source)))
    target_interval = float(np.median(np.diff(target)))
    tolerance = interval_jitter_fraction * max(source_interval, target_interval)
    if abs(source_interval - target_interval) > tolerance:
        raise ValueError(
            "Pulse intervals do not match: "
            f"source median={source_interval:.9g} s, "
            f"target median={target_interval:.9g} s"
        )
    if allow_unmatched_pulses:
        (
            source_indices,
            target_indices,
            unmatched_source,
            unmatched_target,
        ) = _match_by_ordinal(
            source,
            target,
            source_interval,
            target_interval,
            interval_jitter_fraction=interval_jitter_fraction,
        )
        matched_source = source[source_indices]
        matched_target = target[target_indices]
    else:
        count = min(source.size, target.size)
        matched_source = source[:count]
        matched_target = target[:count]
        source_indices = np.arange(count, dtype=int)
        target_indices = np.arange(count, dtype=int)
        unmatched_source = np.arange(count, source.size, dtype=int)
        unmatched_target = np.arange(count, target.size, dtype=int)
        epsilon = 32 * np.finfo(float).eps
        source_limit = 1.5 * source_interval
        target_limit = 1.5 * target_interval
        source_gaps = np.flatnonzero(np.diff(matched_source) > source_limit + epsilon)
        target_gaps = np.flatnonzero(np.diff(matched_target) > target_limit + epsilon)
        if source_gaps.size:
            index = int(source_gaps[0])
            raise ValueError(
                "Source pulse train contains an internal missing pulse after "
                f"index {index}: interval={np.diff(matched_source)[index]:.9g} s, "
                f"expected about {source_interval:.9g} s"
            )
        if target_gaps.size:
            index = int(target_gaps[0])
            raise ValueError(
                "Target pulse train contains an internal missing pulse after "
                f"index {index}: interval={np.diff(matched_target)[index]:.9g} s, "
                f"expected about {target_interval:.9g} s"
            )
    if matched_source.size < 3:
        raise ValueError(
            "At least three matching pulses are required after removing unmatched pulses"
        )
    scale, intercept = np.polyfit(matched_source, matched_target, 1)
    residuals = matched_target - (intercept + scale * matched_source)
    rms_residual = float(np.sqrt(np.mean(residuals**2)))
    max_residual = float(np.max(np.abs(residuals)))
    reference_period = max(source_interval, target_interval)
    drift_ppm = abs((float(scale) - 1.0) * 1e6)
    if (
        max_rms_residual_fraction is not None
        and rms_residual > max_rms_residual_fraction * reference_period
    ):
        raise ValueError(
            "Clock alignment residuals are too large: "
            f"RMS={rms_residual:.6g} s across {matched_source.size} matched pulses"
        )
    if (
        max_residual_fraction is not None
        and max_residual > max_residual_fraction * reference_period
    ):
        raise ValueError(
            "Clock alignment residuals are too large: "
            f"maximum={max_residual:.6g} s across {matched_source.size} matched pulses"
        )
    if max_abs_drift_ppm is not None and drift_ppm > max_abs_drift_ppm:
        raise ValueError(
            "Clock alignment drift is implausible: "
            f"{drift_ppm:.3f} ppm exceeds {max_abs_drift_ppm:.3f} ppm"
        )
    return ClockAlignment(
        float(intercept),
        float(scale),
        residuals,
        matched_source,
        matched_target,
        source_indices + source_start_index,
        target_indices + target_start_index,
        unmatched_source + source_start_index,
        unmatched_target + target_start_index,
    )
