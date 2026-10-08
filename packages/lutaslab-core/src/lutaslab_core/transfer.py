"""Generic causal transfer functions for regularly sampled signals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import factorial

import numpy as np

from .glm import r2_score


@dataclass(frozen=True)
class GammaTransferFit:
    """Best nonnegative-gain gamma-cascade transfer from a candidate grid."""

    order: int
    tau_seconds: float
    delay_seconds: float
    gain: float
    prediction: np.ndarray
    r2: float
    sse: float


def gamma_cascade_kernel(
    lag_seconds: np.ndarray,
    tau_seconds: float,
    *,
    order: int = 1,
    delay_seconds: float = 0.0,
) -> np.ndarray:
    """Return a unit-area delayed gamma-cascade impulse response."""

    lags = np.asarray(lag_seconds, dtype=float)
    if lags.ndim != 1 or not np.all(np.isfinite(lags)) or np.any(np.diff(lags) < 0):
        raise ValueError("lag_seconds must be a finite nondecreasing vector")
    if tau_seconds <= 0 or not np.isfinite(tau_seconds):
        raise ValueError("tau_seconds must be finite and positive")
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("order must be a positive integer")
    if delay_seconds < 0 or not np.isfinite(delay_seconds):
        raise ValueError("delay_seconds must be finite and nonnegative")
    shifted = lags - delay_seconds
    kernel = np.zeros_like(shifted)
    causal = shifted >= 0
    x = shifted[causal]
    kernel[causal] = (
        x ** (order - 1)
        * np.exp(-x / tau_seconds)
        / (tau_seconds**order * factorial(order - 1))
    )
    return kernel


def apply_gamma_transfer(
    signal: np.ndarray,
    sample_interval_seconds: float,
    tau_seconds: float,
    *,
    order: int = 1,
    delay_seconds: float = 0.0,
) -> np.ndarray:
    """Convolve one finite signal with a causal gamma-cascade kernel."""

    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < 2 or not np.all(np.isfinite(signal)):
        raise ValueError("signal must be a finite one-dimensional array")
    if not np.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be finite and positive")
    lags = np.arange(signal.size, dtype=float) * sample_interval_seconds
    kernel = gamma_cascade_kernel(
        lags,
        tau_seconds,
        order=order,
        delay_seconds=delay_seconds,
    )
    return (
        np.convolve(signal, kernel, mode="full")[: signal.size]
        * sample_interval_seconds
    )


def fit_nonnegative_gamma_transfer(
    time: np.ndarray,
    drive: np.ndarray,
    response: np.ndarray,
    tau_candidates: Iterable[float],
    *,
    orders: Iterable[int] = (1,),
    delay_candidates: Iterable[float] = (0.0,),
    fit_window: tuple[float, float] | None = None,
    baseline_window: tuple[float, float] | None = None,
) -> GammaTransferFit:
    """Grid-fit a shared nonnegative gain to one or more paired traces."""

    time = np.asarray(time, dtype=float)
    drive = np.atleast_2d(np.asarray(drive, dtype=float))
    response_input = np.asarray(response, dtype=float)
    response_was_1d = response_input.ndim == 1
    response_rows = np.atleast_2d(response_input)
    if time.ndim != 1 or time.size < 2 or not np.all(np.diff(time) > 0):
        raise ValueError("time must be a strictly increasing vector")
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("time must be regularly sampled")
    if drive.shape[1] != time.size or response_rows.shape[1] != time.size:
        raise ValueError("drive and response columns must match time")
    if drive.shape[0] == 1 and response_rows.shape[0] > 1:
        drive = np.repeat(drive, response_rows.shape[0], axis=0)
    if drive.shape != response_rows.shape:
        raise ValueError("drive and response must have matching shapes")
    if not np.all(np.isfinite(drive)) or not np.all(np.isfinite(response_rows)):
        raise ValueError("drive and response must be finite")
    selected = np.ones(time.size, dtype=bool)
    if fit_window is not None:
        if len(fit_window) != 2 or fit_window[0] >= fit_window[1]:
            raise ValueError("fit_window must contain increasing bounds")
        selected = (time >= fit_window[0]) & (time <= fit_window[1])
        if not np.any(selected):
            raise ValueError("fit_window does not overlap time")
    baseline = None
    if baseline_window is not None:
        if len(baseline_window) != 2 or baseline_window[0] >= baseline_window[1]:
            raise ValueError("baseline_window must contain increasing bounds")
        baseline = (time >= baseline_window[0]) & (time < baseline_window[1])
        if not np.any(baseline):
            raise ValueError("baseline_window does not overlap time")
    taus = tuple(tau_candidates)
    order_values = tuple(orders)
    delays = tuple(delay_candidates)
    if not taus or not order_values or not delays:
        raise ValueError("candidate grids must not be empty")
    best: GammaTransferFit | None = None
    for order in order_values:
        for tau in taus:
            for delay in delays:
                filtered = np.vstack(
                    [
                        apply_gamma_transfer(
                            row,
                            dt,
                            float(tau),
                            order=int(order),
                            delay_seconds=float(delay),
                        )
                        for row in drive
                    ]
                )
                if baseline is not None:
                    filtered -= np.mean(filtered[:, baseline], axis=1, keepdims=True)
                x = filtered[:, selected].reshape(-1)
                y = response_rows[:, selected].reshape(-1)
                denominator = float(x @ x)
                gain = max(0.0, float(x @ y / denominator)) if denominator > 0 else 0.0
                prediction = gain * filtered
                residual = y - gain * x
                sse = float(residual @ residual)
                candidate = GammaTransferFit(
                    order=int(order),
                    tau_seconds=float(tau),
                    delay_seconds=float(delay),
                    gain=gain,
                    prediction=prediction[0] if response_was_1d else prediction,
                    r2=r2_score(y, gain * x),
                    sse=sse,
                )
                if best is None or candidate.sse < best.sse:
                    best = candidate
    assert best is not None
    return best
