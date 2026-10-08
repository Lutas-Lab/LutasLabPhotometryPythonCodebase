"""iFLiP-to-NI-DAQ compatibility API backed by :mod:`lutaslab_core`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lutaslab_core.synchronization import fit_clock_alignment as _fit_clock_alignment


def external_marker_mask(marks: np.ndarray, marker: int = 1) -> np.ndarray:
    """Decode one external iFLiP marker bit (external markers 1 through 4)."""

    if marker not in {1, 2, 3, 4}:
        raise ValueError("marker must be between 1 and 4")
    words = np.asarray(marks, dtype=np.uint32)
    return (words & np.uint32(1 << (marker + 1))) != 0


@dataclass(frozen=True)
class ClockAlignment:
    """Legacy iFLiP3 names for one shared affine clock alignment."""

    intercept_seconds: float
    scale: float
    residuals_seconds: np.ndarray
    matched_pulses: int
    iflip_marker_times: np.ndarray
    nidaq_pulse_times: np.ndarray

    def iflip_to_nidaq(self, times: np.ndarray) -> np.ndarray:
        return self.intercept_seconds + self.scale * np.asarray(times, dtype=float)

    def nidaq_to_iflip(self, times: np.ndarray) -> np.ndarray:
        return (np.asarray(times, dtype=float) - self.intercept_seconds) / self.scale

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
    iflip_marker_times: np.ndarray,
    nidaq_pulse_times: np.ndarray,
    *,
    iflip_start_index: int = 0,
    nidaq_start_index: int = 0,
) -> ClockAlignment:
    """Fit the shared affine alignment and expose established iFLiP3 names."""

    alignment = _fit_clock_alignment(
        iflip_marker_times,
        nidaq_pulse_times,
        source_start_index=iflip_start_index,
        target_start_index=nidaq_start_index,
    )
    return ClockAlignment(
        intercept_seconds=alignment.intercept_seconds,
        scale=alignment.scale,
        residuals_seconds=alignment.residuals_seconds,
        matched_pulses=alignment.matched_pulses,
        iflip_marker_times=alignment.source_times,
        nidaq_pulse_times=alignment.target_times,
    )


__all__ = ["ClockAlignment", "external_marker_mask", "fit_clock_alignment"]
