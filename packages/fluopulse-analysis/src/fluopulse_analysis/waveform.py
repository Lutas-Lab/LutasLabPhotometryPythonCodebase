"""Independent exploratory analyses of raw FluoPulse waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve


def subtract_terminal_baseline(
    values: np.ndarray, *, fraction: float = 0.1
) -> np.ndarray:
    """Subtract the median of the terminal waveform segment."""

    values = np.asarray(values, dtype=float)
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie between zero and one")
    points = max(1, int(np.ceil(values.shape[-1] * fraction)))
    baseline = np.nanmedian(values[..., -points:], axis=-1, keepdims=True)
    return values - baseline


def moment_lifetime_ns(
    time_ns: np.ndarray,
    waveform: np.ndarray,
    irf: np.ndarray,
    *,
    window_ns: tuple[float, float] | None = None,
) -> float:
    """Estimate lifetime as the signal centroid minus the IRF centroid.

    This is an exploratory moment estimate for waveform-sampled data, not the
    vendor Tau01 algorithm.
    """

    time = np.asarray(time_ns, dtype=float)
    signal = subtract_terminal_baseline(np.asarray(waveform, dtype=float))
    response = subtract_terminal_baseline(np.asarray(irf, dtype=float))
    if signal.shape != time.shape or response.shape != time.shape:
        raise ValueError("time, waveform, and IRF must have matching shapes")
    use = np.ones(time.size, dtype=bool)
    if window_ns is not None:
        use = (time >= window_ns[0]) & (time <= window_ns[1])
    signal = np.clip(signal[use], 0, None)
    response = np.clip(response[use], 0, None)
    selected_time = time[use]
    if signal.sum() <= 0 or response.sum() <= 0:
        return float("nan")
    signal_centroid = np.sum(selected_time * signal) / np.sum(signal)
    irf_centroid = np.sum(selected_time * response) / np.sum(response)
    return float(signal_centroid - irf_centroid)


@dataclass(frozen=True)
class WaveformFitResult:
    tau_ns: float
    amplitude: float
    offset: float
    shift_ns: float
    fitted: np.ndarray
    residuals: np.ndarray
    r_square: float
    success: bool


@dataclass(frozen=True)
class WaveformDoubleFitResult:
    tau_short_ns: float
    tau_long_ns: float
    long_fraction: float
    amplitude: float
    offset: float
    shift_ns: float
    fitted: np.ndarray
    residuals: np.ndarray
    r_square: float
    success: bool


def _convolved_basis(
    time_ns: np.ndarray,
    irf: np.ndarray,
    tau_ns: float,
    shift_ns: float,
) -> np.ndarray:
    dt = float(np.median(np.diff(time_ns)))
    shifted_irf = np.interp(
        time_ns - shift_ns,
        time_ns,
        irf,
        left=0.0,
        right=0.0,
    )
    decay = np.exp(-(time_ns - time_ns[0]) / tau_ns)
    basis = fftconvolve(shifted_irf, decay, mode="full")[: time_ns.size] * dt
    maximum = float(np.max(np.abs(basis)))
    return basis / maximum if maximum > 0 else basis


def fit_irf_convolved_single_exponential(
    time_ns: np.ndarray,
    waveform: np.ndarray,
    irf: np.ndarray,
    *,
    fit_window_ns: tuple[float, float] | None = None,
    tau_bounds_ns: tuple[float, float] = (0.05, 20.0),
    shift_bounds_ns: tuple[float, float] = (-5.0, 5.0),
) -> WaveformFitResult:
    """Fit an IRF-convolved single exponential to one raw waveform.

    This independent fit is intended for method development and validation,
    not as an exact reproduction of Doric's proprietary deconvolution.
    """

    time = np.asarray(time_ns, dtype=float)
    observed = np.asarray(waveform, dtype=float)
    response = subtract_terminal_baseline(np.asarray(irf, dtype=float))
    if time.ndim != 1 or observed.shape != time.shape or response.shape != time.shape:
        raise ValueError("time, waveform, and IRF must be matching vectors")
    use = np.ones(time.size, dtype=bool)
    if fit_window_ns is not None:
        use = (time >= fit_window_ns[0]) & (time <= fit_window_ns[1])
    if use.sum() < 8:
        raise ValueError("Fit window must contain at least eight points")

    initial_offset = float(np.median(observed[-max(4, time.size // 10) :]))
    initial_amplitude = max(float(np.max(observed) - initial_offset), 1e-6)
    initial_tau = float(np.sqrt(tau_bounds_ns[0] * tau_bounds_ns[1]))

    def model(parameters: np.ndarray) -> np.ndarray:
        tau = float(np.exp(parameters[0]))
        amplitude = float(np.exp(parameters[1]))
        offset = float(parameters[2])
        shift = float(parameters[3])
        return offset + amplitude * _convolved_basis(time, response, tau, shift)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        return model(parameters)[use] - observed[use]

    result = least_squares(
        residuals,
        x0=np.array(
            [np.log(initial_tau), np.log(initial_amplitude), initial_offset, 0.0]
        ),
        bounds=(
            [np.log(tau_bounds_ns[0]), -30.0, -np.inf, shift_bounds_ns[0]],
            [np.log(tau_bounds_ns[1]), 30.0, np.inf, shift_bounds_ns[1]],
        ),
    )
    fitted = model(result.x)
    residual = observed - fitted
    centered = observed[use] - np.mean(observed[use])
    denominator = float(np.sum(centered**2))
    r_square = (
        1.0 - float(np.sum(residual[use] ** 2)) / denominator if denominator else np.nan
    )
    return WaveformFitResult(
        tau_ns=float(np.exp(result.x[0])),
        amplitude=float(np.exp(result.x[1])),
        offset=float(result.x[2]),
        shift_ns=float(result.x[3]),
        fitted=fitted,
        residuals=residual,
        r_square=r_square,
        success=bool(result.success),
    )


def fit_irf_convolved_double_exponential(
    time_ns: np.ndarray,
    waveform: np.ndarray,
    irf: np.ndarray,
    *,
    fit_window_ns: tuple[float, float] = (0.0, 40.0),
    tau_bounds_ns: tuple[float, float] = (0.05, 20.0),
    shift_bounds_ns: tuple[float, float] = (-5.0, 5.0),
) -> WaveformDoubleFitResult:
    """Fit an exploratory IRF-convolved two-exponential mixture."""

    time = np.asarray(time_ns, dtype=float)
    observed = np.asarray(waveform, dtype=float)
    response = subtract_terminal_baseline(np.asarray(irf, dtype=float))
    if time.ndim != 1 or observed.shape != time.shape or response.shape != time.shape:
        raise ValueError("time, waveform, and IRF must be matching vectors")
    use = (time >= fit_window_ns[0]) & (time <= fit_window_ns[1])
    if use.sum() < 10:
        raise ValueError("Fit window must contain at least ten points")
    initial_offset = float(np.median(observed[-max(4, time.size // 10) :]))
    initial_amplitude = max(float(np.max(observed) - initial_offset), 1e-6)

    def unpack(parameters: np.ndarray) -> tuple[float, float, float]:
        tau_a, tau_b = np.exp(parameters[:2])
        fraction_b = 1.0 / (1.0 + np.exp(-parameters[2]))
        if tau_a <= tau_b:
            return float(tau_a), float(tau_b), float(fraction_b)
        return float(tau_b), float(tau_a), float(1.0 - fraction_b)

    def model(parameters: np.ndarray) -> np.ndarray:
        tau_short, tau_long, long_fraction = unpack(parameters)
        amplitude = float(np.exp(parameters[3]))
        offset = float(parameters[4])
        shift = float(parameters[5])
        short_basis = _convolved_basis(time, response, tau_short, shift)
        long_basis = _convolved_basis(time, response, tau_long, shift)
        mixture = (1.0 - long_fraction) * short_basis + long_fraction * long_basis
        return offset + amplitude * mixture

    result = least_squares(
        lambda parameters: model(parameters)[use] - observed[use],
        x0=np.array(
            [
                np.log(0.6),
                np.log(2.5),
                0.0,
                np.log(initial_amplitude),
                initial_offset,
                0.0,
            ]
        ),
        bounds=(
            [
                np.log(tau_bounds_ns[0]),
                np.log(tau_bounds_ns[0]),
                -12.0,
                -30.0,
                -np.inf,
                shift_bounds_ns[0],
            ],
            [
                np.log(tau_bounds_ns[1]),
                np.log(tau_bounds_ns[1]),
                12.0,
                30.0,
                np.inf,
                shift_bounds_ns[1],
            ],
        ),
    )
    fitted = model(result.x)
    residuals = observed - fitted
    centered = observed[use] - np.mean(observed[use])
    denominator = float(np.sum(centered**2))
    r_square = (
        1.0 - float(np.sum(residuals[use] ** 2)) / denominator
        if denominator
        else np.nan
    )
    tau_short, tau_long, long_fraction = unpack(result.x)
    return WaveformDoubleFitResult(
        tau_short_ns=tau_short,
        tau_long_ns=tau_long,
        long_fraction=long_fraction,
        amplitude=float(np.exp(result.x[3])),
        offset=float(result.x[4]),
        shift_ns=float(result.x[5]),
        fitted=fitted,
        residuals=residuals,
        r_square=r_square,
        success=bool(result.success),
    )
