"""Interpretable transfer models linking dopamine input to PKA biosensor output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from lutaslab_core.glm import r2_score


@dataclass(frozen=True)
class PKATransferFit:
    rise_tau_seconds: float
    decay_tau_seconds: float
    delay_seconds: float
    gain: float
    baseline: float
    prediction: np.ndarray
    r2: float
    cost: float
    success: bool


def causal_biexponential_kernel(
    time: np.ndarray,
    rise_tau_seconds: float,
    decay_tau_seconds: float,
    delay_seconds: float = 0.0,
) -> np.ndarray:
    """Return a unit-area causal rise-and-decay response kernel."""

    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("time must be a strictly increasing one-dimensional array")
    if rise_tau_seconds <= 0 or decay_tau_seconds <= rise_tau_seconds:
        raise ValueError("decay_tau_seconds must be greater than positive rise_tau_seconds")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be nonnegative")
    shifted = time - float(time[0]) - delay_seconds
    kernel = np.zeros_like(shifted)
    causal = shifted >= 0
    kernel[causal] = np.exp(-shifted[causal] / decay_tau_seconds) - np.exp(
        -shifted[causal] / rise_tau_seconds
    )
    dt = float(np.median(np.diff(time)))
    area = float(kernel.sum() * dt)
    if area <= 0 or not np.isfinite(area):
        raise ValueError("kernel has no finite support on the supplied time vector")
    return kernel / area


def predict_pka_from_dopamine(
    time: np.ndarray,
    dopamine: np.ndarray,
    *,
    rise_tau_seconds: float,
    decay_tau_seconds: float,
    delay_seconds: float = 0.0,
    gain: float = 1.0,
    baseline: float = 0.0,
    rectify: bool = True,
) -> np.ndarray:
    """Convolve dopamine with a causal biochemical response kernel."""

    time = np.asarray(time, dtype=float)
    dopamine = np.asarray(dopamine, dtype=float)
    if dopamine.shape != time.shape or not np.all(np.isfinite(dopamine)):
        raise ValueError("dopamine must be finite and have the same shape as time")
    drive = np.maximum(dopamine, 0.0) if rectify else dopamine
    kernel_time = np.arange(time.size, dtype=float) * np.median(np.diff(time))
    kernel = causal_biexponential_kernel(
        kernel_time,
        rise_tau_seconds,
        decay_tau_seconds,
        delay_seconds,
    )
    dt = float(np.median(np.diff(time)))
    filtered = np.convolve(drive, kernel, mode="full")[: time.size] * dt
    return baseline + gain * filtered


def fit_dopamine_to_pka(
    time: np.ndarray,
    dopamine: np.ndarray,
    pka: np.ndarray,
    *,
    rectify: bool = True,
    initial_rise_tau_seconds: float = 0.5,
    initial_decay_tau_seconds: float = 5.0,
    initial_delay_seconds: float = 0.2,
) -> PKATransferFit:
    """Fit a delayed biexponential dopamine-to-PKA transfer function."""

    time = np.asarray(time, dtype=float)
    dopamine = np.asarray(dopamine, dtype=float)
    pka = np.asarray(pka, dtype=float)
    if time.shape != dopamine.shape or time.shape != pka.shape:
        raise ValueError("time, dopamine, and pka must have identical shapes")
    finite = np.isfinite(time) & np.isfinite(dopamine) & np.isfinite(pka)
    if finite.sum() != time.size:
        raise ValueError("time, dopamine, and pka must be finite")
    if initial_decay_tau_seconds <= initial_rise_tau_seconds:
        raise ValueError("initial decay time must be greater than initial rise time")

    initial_gain = max(float(np.ptp(pka) / max(np.ptp(dopamine), 1e-6)), 1e-6)
    initial = np.array(
        [
            np.log(initial_rise_tau_seconds),
            np.log(initial_decay_tau_seconds - initial_rise_tau_seconds),
            np.log(max(initial_delay_seconds, 1e-4)),
            np.log(initial_gain),
            float(np.min(pka)),
        ]
    )

    def unpack(parameters: np.ndarray) -> tuple[float, float, float, float, float]:
        rise = float(np.exp(parameters[0]))
        decay = rise + float(np.exp(parameters[1]))
        delay = float(np.exp(parameters[2]))
        gain = float(np.exp(parameters[3]))
        baseline = float(parameters[4])
        return rise, decay, delay, gain, baseline

    def residual(parameters: np.ndarray) -> np.ndarray:
        rise, decay, delay, gain, baseline = unpack(parameters)
        return predict_pka_from_dopamine(
            time,
            dopamine,
            rise_tau_seconds=rise,
            decay_tau_seconds=decay,
            delay_seconds=delay,
            gain=gain,
            baseline=baseline,
            rectify=rectify,
        ) - pka

    result = least_squares(residual, initial, max_nfev=3000)
    rise, decay, delay, gain, baseline = unpack(result.x)
    prediction = predict_pka_from_dopamine(
        time,
        dopamine,
        rise_tau_seconds=rise,
        decay_tau_seconds=decay,
        delay_seconds=delay,
        gain=gain,
        baseline=baseline,
        rectify=rectify,
    )
    return PKATransferFit(
        rise_tau_seconds=rise,
        decay_tau_seconds=decay,
        delay_seconds=delay,
        gain=gain,
        baseline=baseline,
        prediction=prediction,
        r2=r2_score(pka, prediction),
        cost=float(result.cost),
        success=bool(result.success),
    )


def load_population_dopamine_input(
    path: str | Path, source: str = "legacy"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load time, population mean, and bootstrap interval from a template NPZ."""

    if source not in {"legacy", "raw465", "dff"}:
        raise ValueError("source must be 'legacy', 'raw465', or 'dff'")
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["time"], dtype=float),
            np.asarray(data[f"{source}_population_mean"], dtype=float),
            np.asarray(data[f"{source}_population_ci_low"], dtype=float),
            np.asarray(data[f"{source}_population_ci_high"], dtype=float),
        )
