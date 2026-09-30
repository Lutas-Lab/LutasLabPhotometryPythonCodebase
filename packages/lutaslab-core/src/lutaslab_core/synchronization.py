"""Affine alignment between acquisition clocks using ordered shared pulses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClockAlignment:
    intercept_seconds: float
    scale: float
    residuals_seconds: np.ndarray
    source_times: np.ndarray
    target_times: np.ndarray

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


def fit_clock_alignment(
    source_times: np.ndarray,
    target_times: np.ndarray,
    *,
    source_start_index: int = 0,
    target_start_index: int = 0,
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
    source = source[source_start_index:]
    target = target[target_start_index:]
    count = min(source.size, target.size)
    if count < 3:
        raise ValueError("At least three matching pulses are required")
    source = source[:count]
    target = target[:count]
    if np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0):
        raise ValueError("Pulse times must be strictly increasing")
    source_interval = float(np.median(np.diff(source)))
    target_interval = float(np.median(np.diff(target)))
    tolerance = 0.25 * max(source_interval, target_interval)
    if abs(source_interval - target_interval) > tolerance:
        raise ValueError("Pulse intervals do not match")
    if np.any(np.diff(source) > 1.5 * source_interval):
        raise ValueError("Source pulse train contains an internal missing pulse")
    if np.any(np.diff(target) > 1.5 * target_interval):
        raise ValueError("Target pulse train contains an internal missing pulse")
    scale, intercept = np.polyfit(source, target, 1)
    residuals = target - (intercept + scale * source)
    return ClockAlignment(float(intercept), float(scale), residuals, source, target)
