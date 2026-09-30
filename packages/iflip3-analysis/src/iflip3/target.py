"""Fixed-basis target analysis for time-resolved lifetime photometry."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy.optimize import nnls

from .models import design_matrix


@dataclass(frozen=True)
class TargetAnalysisResult:
    """Result of decomposing decay curves into fixed lifetime components."""

    lifetimes: np.ndarray
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    residual_background_per_bin: np.ndarray

    @property
    def component_counts(self) -> np.ndarray:
        """Area-normalized photon contribution for each curve and component."""

        return self.coefficients

    @property
    def total_component_counts(self) -> np.ndarray:
        return self.coefficients.sum(axis=1)

    @property
    def component_fractions(self) -> np.ndarray:
        totals = self.total_component_counts[:, None]
        return np.divide(
            self.coefficients,
            totals,
            out=np.full_like(self.coefficients, np.nan),
            where=totals > 0,
        )


def lifetime_window(
    time: np.ndarray,
    limits: tuple[float, float] = (0.4, 12.3),
) -> np.ndarray:
    """Return a mask for the artifact-free lifetime interval."""

    time = np.asarray(time, dtype=float)
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if limits[0] >= limits[1]:
        raise ValueError("lifetime window limits must be increasing")
    mask = (time >= limits[0]) & (time <= limits[1])
    if not np.any(mask):
        raise ValueError("No lifetime bins fall inside the requested window")
    return mask


def _background_array(
    value: float | np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    background = np.asarray(value, dtype=float)
    if background.ndim == 0:
        return np.full(shape, float(background))
    if background.shape == (shape[0],):
        return np.broadcast_to(background[:, None], shape)
    if background.shape == (shape[1],):
        return np.broadcast_to(background[None, :], shape)
    try:
        return np.broadcast_to(background, shape)
    except ValueError as exc:
        raise ValueError(
            "residual_background_per_bin cannot be broadcast to the decay matrix"
        ) from exc


def fit_target(
    time: np.ndarray,
    decays: np.ndarray,
    *,
    lifetimes: Sequence[float],
    t0: float,
    irf_sigma: float,
    pulse_interval: float = 12.5,
    residual_background_per_bin: float | np.ndarray = 0.0,
    weighting: str = "none",
) -> TargetAnalysisResult:
    """Solve component counts with lifetimes and instrument parameters fixed.

    Parameters are fixed across all curves. Only nonnegative component photon
    contributions vary with time. A known recording-level residual background
    can be subtracted, but no independent local background is fitted; this
    prevents the long component and background from becoming non-identifiable.
    """

    time = np.asarray(time, dtype=float)
    curves = np.asarray(decays, dtype=float)
    if curves.ndim == 1:
        curves = curves[:, None]
    if curves.ndim != 2 or curves.shape[0] != time.size:
        raise ValueError("decays must have shape (lifetime bins, curves)")
    if not np.all(np.isfinite(curves)):
        raise ValueError("decays must contain only finite values")

    lifetimes_array = np.sort(np.asarray(lifetimes, dtype=float))
    if lifetimes_array.ndim != 1 or lifetimes_array.size not in (1, 2):
        raise ValueError("lifetimes must contain one or two values")
    if np.any(lifetimes_array <= 0):
        raise ValueError("lifetimes must be positive")

    background = _background_array(residual_background_per_bin, curves.shape)
    adjusted = curves - background
    basis = design_matrix(
        time,
        lifetimes_array,
        t0,
        irf_sigma,
        pulse_interval,
        normalize="area",
        include_background=False,
    )

    if weighting == "none":
        weights = np.ones_like(adjusted)
    elif weighting == "poisson":
        if np.any(curves < 0):
            raise ValueError(
                "Poisson weighting is undefined for negative curves; use "
                "weighting='none' or an artifact-free nonnegative window"
            )
        weights = 1.0 / np.sqrt(np.maximum(curves, 1.0))
    else:
        raise ValueError("weighting must be 'none' or 'poisson'")

    coefficients = np.empty((curves.shape[1], lifetimes_array.size), dtype=float)
    for column in range(curves.shape[1]):
        weighted_basis = basis * weights[:, column, None]
        weighted_curve = adjusted[:, column] * weights[:, column]
        coefficients[column] = nnls(weighted_basis, weighted_curve)[0]

    fitted_adjusted = basis @ coefficients.T
    fitted = fitted_adjusted + background
    return TargetAnalysisResult(
        lifetimes=lifetimes_array,
        coefficients=coefficients,
        fitted=fitted,
        residuals=curves - fitted,
        residual_background_per_bin=background,
    )
