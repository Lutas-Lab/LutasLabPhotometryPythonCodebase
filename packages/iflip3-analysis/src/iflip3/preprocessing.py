"""Background correction, temporal binning, and MPET calculation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .io import IFLiP3Header, IFLiP3Recording


def _as_cube(data: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(data)
    if array.ndim == 2:
        return array[:, :, None], True
    if array.ndim != 3:
        raise ValueError("Lifetime data must have shape (bins, samples[, channels])")
    return array, False


@dataclass(frozen=True)
class CorrectionResult:
    corrected: np.ndarray
    total_background: np.ndarray
    measured_background: np.ndarray
    afterpulse_background: np.ndarray
    count_efficiency: np.ndarray


def average_background(recordings: Sequence[IFLiP3Recording]) -> np.ndarray:
    """Average compatible background samples, returning ``(bins, channels)``.

    All supplied recordings must have been acquired under the same optical and
    detector conditions. In particular, backgrounds acquired at different
    laser powers must not be pooled.
    """

    if not recordings:
        raise ValueError("At least one background recording is required")
    shapes = {(r.data.shape[0], r.data.shape[2]) for r in recordings}
    if len(shapes) != 1:
        raise ValueError("Background recordings have incompatible bin/channel dimensions")
    return np.concatenate([r.data.astype(float) for r in recordings], axis=1).mean(axis=1)


def correct_lifetime_data(
    data: np.ndarray,
    measured_background: np.ndarray | None,
    *,
    dead_time_seconds: float,
    sampling_frequency: float,
    afterpulse_ratio: float = 0.0,
) -> CorrectionResult:
    """Apply the vendor dead-time-aware background and afterpulse correction."""

    cube, squeezed = _as_cube(data)
    cube = cube.astype(float, copy=False)
    n_bins, _, n_channels = cube.shape
    if afterpulse_ratio < 0:
        raise ValueError("afterpulse_ratio must be nonnegative")

    if measured_background is None:
        bg = np.zeros((n_bins, n_channels), dtype=float)
    else:
        bg = np.asarray(measured_background, dtype=float)
        if bg.ndim == 1:
            bg = bg[:, None]
        if bg.shape != (n_bins, n_channels):
            raise ValueError(
                f"Background shape {bg.shape} does not match {(n_bins, n_channels)}"
            )

    background_efficiency = 1.0 - bg.sum(axis=0) * sampling_frequency * dead_time_seconds
    if np.any(background_efficiency <= 0):
        raise ValueError("Background rate is too high for the dead-time correction")
    true_background = bg / background_efficiency[None, :]

    raw_rate = cube.sum(axis=0) * sampling_frequency
    count_efficiency = 1.0 - raw_rate * dead_time_seconds
    if np.any(count_efficiency <= 0):
        raise ValueError("Photon rate is too high for the dead-time correction")

    effective_bg = true_background[:, None, :] * count_efficiency[None, :, :]
    effective_bg_rate = true_background.sum(axis=0)[None, :] * count_efficiency * sampling_frequency
    afterpulse_per_bin = (
        (raw_rate - effective_bg_rate)
        * afterpulse_ratio
        / (sampling_frequency * n_bins)
    )
    afterpulse_bg = np.broadcast_to(afterpulse_per_bin[None, :, :], cube.shape).copy()
    total_bg = effective_bg + afterpulse_bg
    corrected = cube - total_bg

    if squeezed:
        return CorrectionResult(
            corrected[:, :, 0],
            total_bg[:, :, 0],
            effective_bg[:, :, 0],
            afterpulse_bg[:, :, 0],
            count_efficiency[:, 0],
        )
    return CorrectionResult(corrected, total_bg, effective_bg, afterpulse_bg, count_efficiency)


def mpet_from_corrected(
    corrected_data: np.ndarray,
    lifetime_time: np.ndarray,
    spc_range: tuple[float, float],
    t0: float,
) -> np.ndarray:
    """Calculate mean photon emission time from corrected lifetime curves."""

    cube, squeezed = _as_cube(corrected_data)
    lifetime_time = np.asarray(lifetime_time, dtype=float)
    if lifetime_time.shape != (cube.shape[0],):
        raise ValueError("lifetime_time length does not match the number of bins")
    use = (lifetime_time >= spc_range[0]) & (lifetime_time <= spc_range[1])
    if not np.any(use):
        raise ValueError("No lifetime bins fall inside spc_range")
    selected = cube[use]
    totals = selected.sum(axis=0)
    weighted = (selected * lifetime_time[use, None, None]).sum(axis=0)
    result = np.full(totals.shape, np.nan, dtype=float)
    valid = totals > 0
    result[valid] = weighted[valid] / totals[valid] - t0
    return result[:, 0] if squeezed else result


def calculate_mpet(
    recording: IFLiP3Recording,
    spc_range: tuple[float, float],
    t0: float,
    measured_background: np.ndarray | None = None,
    afterpulse_ratio: float = 0.0,
) -> tuple[np.ndarray, CorrectionResult]:
    correction = correct_lifetime_data(
        recording.data,
        measured_background,
        dead_time_seconds=recording.header.dead_time_seconds,
        sampling_frequency=recording.header.sampling_frequency,
        afterpulse_ratio=afterpulse_ratio,
    )
    mpet = mpet_from_corrected(correction.corrected, recording.lifetime_time, spc_range, t0)
    return mpet, correction


def bin_curves(data: np.ndarray, group_size: int, *, reducer: str = "sum") -> np.ndarray:
    """Combine adjacent photometry samples along axis 1."""

    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    array = np.asarray(data)
    usable = (array.shape[1] // group_size) * group_size
    if usable == 0:
        raise ValueError("group_size exceeds the number of samples")
    shape = (array.shape[0], usable // group_size, group_size) + array.shape[2:]
    grouped = array[:, :usable].reshape(shape)
    if reducer == "sum":
        return grouped.sum(axis=2)
    if reducer == "mean":
        return grouped.mean(axis=2)
    raise ValueError("reducer must be 'sum' or 'mean'")
