"""Lifetime models convolved with a Gaussian instrument response function."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

try:
    from scipy.special import log_ndtr as _log_ndtr
except ImportError:  # Keep model evaluation usable when only NumPy is installed.
    def _log_ndtr(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        output = np.empty_like(values)
        regular = values > -10.0
        erfc = np.frompyfunc(math.erfc, 1, 1)
        probabilities = 0.5 * np.asarray(
            erfc(-values[regular] / math.sqrt(2.0)), dtype=float
        )
        output[regular] = np.log(probabilities)
        x = values[~regular]
        inverse_square = 1.0 / (x * x)
        correction = 1.0 - inverse_square + 3.0 * inverse_square**2 - 15.0 * inverse_square**3
        output[~regular] = (
            -0.5 * x * x
            - np.log(-x)
            - 0.5 * math.log(2.0 * math.pi)
            + np.log(correction)
        )
        return output


def periodic_exgaussian_basis(
    time: np.ndarray,
    tau: float,
    t0: float,
    irf_sigma: float,
    pulse_interval: float,
    *,
    normalize: str | None = None,
    n_previous: int | None = None,
) -> np.ndarray:
    """Gaussian-convolved exponential including preceding excitation pulses.

    This is algebraically equivalent to the vendor MATLAB model but evaluates
    its Gaussian tail in log space for better numerical stability.
    """

    if tau <= 0 or irf_sigma <= 0 or pulse_interval <= 0:
        raise ValueError("tau, irf_sigma, and pulse_interval must be positive")
    time = np.asarray(time, dtype=float)
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if n_previous is None:
        n_previous = max(2, int(math.floor(tau / pulse_interval * 10.0 + 0.5)))
    offsets = -t0 + np.arange(n_previous + 1, dtype=float) * pulse_interval
    x = time[:, None] + offsets[None, :]
    log_terms = (
        irf_sigma**2 / (2.0 * tau**2)
        - x / tau
        + _log_ndtr(x / irf_sigma - irf_sigma / tau)
    )
    basis = np.exp(log_terms).sum(axis=1)
    if normalize is None or normalize == "none":
        return basis
    if normalize == "area":
        scale = basis.sum()
    elif normalize == "peak":
        scale = basis.max()
    else:
        raise ValueError("normalize must be None, 'area', or 'peak'")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot normalize a degenerate lifetime basis")
    return basis / scale


def design_matrix(
    time: np.ndarray,
    lifetimes: Sequence[float],
    t0: float,
    irf_sigma: float,
    pulse_interval: float,
    *,
    normalize: str | None = "area",
    include_background: bool = False,
) -> np.ndarray:
    columns = [
        periodic_exgaussian_basis(
            time,
            tau,
            t0,
            irf_sigma,
            pulse_interval,
            normalize=normalize,
        )
        for tau in lifetimes
    ]
    if include_background:
        background = np.ones_like(np.asarray(time, dtype=float))
        if normalize == "area":
            background /= background.sum()
        columns.append(background)
    return np.column_stack(columns)
