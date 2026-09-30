"""Backend-independent utilities for temporal Gaussian GLMs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalBasis:
    lag_times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class RidgeCVResult:
    coefficients: np.ndarray
    best_alpha: float
    alphas: np.ndarray
    mean_scores: np.ndarray
    fold_scores: np.ndarray
    column_scale: np.ndarray


def raised_cosine_basis(
    window: tuple[float, float],
    count: int,
    dt: float,
) -> TemporalBasis:
    """Create overlapping linear raised-cosine basis functions."""

    start, stop = map(float, window)
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError("window must contain increasing finite values")
    if count < 1:
        raise ValueError("count must be positive")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    lag_times = np.arange(start, stop + 0.5 * dt, dt)
    centers = np.linspace(start, stop, count)
    width = centers[1] - centers[0] if count > 1 else stop - start
    width = max(float(width), dt)
    values = np.zeros((lag_times.size, count), dtype=float)
    for column, center in enumerate(centers):
        distance = (lag_times - center) / width
        inside = np.abs(distance) <= 1
        values[inside, column] = 0.5 * (1 + np.cos(np.pi * distance[inside]))
    return TemporalBasis(lag_times, values)


def convolve_basis(events: np.ndarray, basis_values: np.ndarray) -> np.ndarray:
    """Causally convolve a binned event signal with each basis column."""

    events = np.asarray(events, dtype=float)
    basis_values = np.asarray(basis_values, dtype=float)
    if events.ndim != 1 or basis_values.ndim != 2:
        raise ValueError("events must be 1D and basis_values must be 2D")
    return np.column_stack(
        [
            np.convolve(events, basis_values[:, column], mode="full")[: events.size]
            for column in range(basis_values.shape[1])
        ]
    )


def apply_lag_basis(
    signal: np.ndarray,
    lag_samples: np.ndarray,
    basis_values: np.ndarray,
    *,
    edge_nan: bool = True,
) -> np.ndarray:
    """Apply basis weights over arbitrary signed sample lags."""

    signal = np.asarray(signal, dtype=float)
    lag_samples = np.asarray(lag_samples, dtype=int)
    basis_values = np.asarray(basis_values, dtype=float)
    if signal.ndim != 1 or lag_samples.ndim != 1 or basis_values.ndim != 2:
        raise ValueError("Invalid temporal-basis dimensions")
    if lag_samples.size != basis_values.shape[0]:
        raise ValueError("Each lag sample must have one row of basis weights")
    n_samples = signal.size
    design = np.zeros((n_samples, basis_values.shape[1]), dtype=float)
    for index, lag in enumerate(lag_samples):
        if lag < 0:
            response_slice = slice(-lag, n_samples)
            predictor_slice = slice(0, n_samples + lag)
        elif lag > 0:
            response_slice = slice(0, n_samples - lag)
            predictor_slice = slice(lag, n_samples)
        else:
            response_slice = slice(0, n_samples)
            predictor_slice = slice(0, n_samples)
        design[response_slice] += signal[predictor_slice, None] * basis_values[index]
    if edge_nan and lag_samples.size:
        left = max(0, -int(lag_samples.min(initial=0)))
        right = max(0, int(lag_samples.max(initial=0)))
        if left:
            design[:left] = np.nan
        if right:
            design[-right:] = np.nan
    return design


def lagged_signal_matrix(
    signal: np.ndarray,
    pre_samples: int,
    post_samples: int,
    *,
    stride: int = 1,
    edge_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a signal into one column per signed lag.

    Column ``j`` contains ``signal[t - lag[j]]``. Positive lags therefore
    describe responses after an event, matching the MATLAB design used for
    the Kerspern/Lutas Figure 5 GLMs. Samples beyond the recording boundary
    are filled with ``edge_value``.
    """

    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if pre_samples < 0 or post_samples < 0:
        raise ValueError("pre_samples and post_samples must be nonnegative")
    if stride < 1:
        raise ValueError("stride must be positive")
    if not np.isfinite(edge_value):
        raise ValueError("edge_value must be finite")
    lags = np.arange(-pre_samples, post_samples + 1, stride, dtype=int)
    design = np.full((signal.size, lags.size), edge_value, dtype=float)
    for column, lag in enumerate(lags):
        if lag < 0:
            design[: signal.size + lag, column] = signal[-lag:]
        elif lag > 0:
            design[lag:, column] = signal[: signal.size - lag]
        else:
            design[:, column] = signal
    return design, lags


def lagged_basis_matrix(
    signal: np.ndarray,
    lag_samples: np.ndarray,
    basis_values: np.ndarray,
) -> np.ndarray:
    """Apply a smooth basis using the raw-lag convention.

    This is equivalent to materializing the signed-lag matrix and multiplying
    it by ``basis_values``, but avoids the much larger intermediate array.
    """

    signal = np.asarray(signal, dtype=float)
    lag_samples = np.asarray(lag_samples, dtype=int)
    basis_values = np.asarray(basis_values, dtype=float)
    if signal.ndim != 1 or lag_samples.ndim != 1 or basis_values.ndim != 2:
        raise ValueError("Invalid lag-basis dimensions")
    if lag_samples.size != basis_values.shape[0]:
        raise ValueError("Each lag sample must have one row of basis weights")
    if lag_samples.size and np.all(np.diff(lag_samples) == 1):
        first_lag = int(lag_samples[0])
        start = -first_lag
        return np.column_stack(
            [
                np.convolve(signal, basis_values[:, column], mode="full")[
                    start : start + signal.size
                ]
                for column in range(basis_values.shape[1])
            ]
        )
    design = np.zeros((signal.size, basis_values.shape[1]), dtype=float)
    for row, lag in enumerate(lag_samples):
        if lag < 0:
            design[: signal.size + lag] += signal[-lag:, None] * basis_values[row]
        elif lag > 0:
            design[lag:] += signal[: signal.size - lag, None] * basis_values[row]
        else:
            design += signal[:, None] * basis_values[row]
    return design


def predict_lagged_signal(
    signal: np.ndarray,
    kernel: np.ndarray,
    pre_samples: int,
    *,
    stride: int = 1,
    intercept: float = 0.0,
) -> np.ndarray:
    """Predict from a raw signed-lag kernel without retaining a design matrix."""

    signal = np.asarray(signal, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if signal.ndim != 1 or kernel.ndim != 1:
        raise ValueError("signal and kernel must be one-dimensional")
    if pre_samples < 0 or stride < 1:
        raise ValueError("pre_samples must be nonnegative and stride must be positive")
    if not np.isfinite(intercept):
        raise ValueError("intercept must be finite")
    lags = -pre_samples + stride * np.arange(kernel.size)
    prediction = np.full(signal.size, float(intercept), dtype=float)
    for weight, lag in zip(kernel, lags, strict=True):
        if lag < 0:
            prediction[: signal.size + lag] += weight * signal[-lag:]
        elif lag > 0:
            prediction[lag:] += weight * signal[: signal.size - lag]
        else:
            prediction += weight * signal
    return prediction


def event_times_to_counts(event_times: np.ndarray, sample_time: np.ndarray) -> np.ndarray:
    """Bin timestamps onto a strictly increasing, approximately uniform timebase."""

    event_times = np.asarray(event_times, dtype=float)
    sample_time = np.asarray(sample_time, dtype=float)
    if sample_time.ndim != 1 or sample_time.size < 2:
        raise ValueError("sample_time must contain at least two samples")
    differences = np.diff(sample_time)
    if not np.all(np.isfinite(sample_time)) or np.any(differences <= 0):
        raise ValueError("sample_time must be finite and strictly increasing")
    dt = float(np.median(differences))
    if not np.allclose(differences, dt, rtol=1e-4, atol=1e-9):
        raise ValueError("sample_time must be approximately uniform")
    edges = np.concatenate(
        [sample_time - dt / 2, np.array([sample_time[-1] + dt / 2])]
    )
    return np.histogram(event_times[np.isfinite(event_times)], bins=edges)[0].astype(float)


def mse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if total == 0 else float(1 - residual / total)


def blocked_folds(
    n_samples: int,
    n_folds: int = 5,
    gap_samples: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create contiguous test folds with a training exclusion gap."""

    if n_samples < 2:
        raise ValueError("At least two samples are required for cross-validation")
    if n_folds < 2 or n_folds > n_samples:
        raise ValueError("n_folds must be between 2 and n_samples")
    if gap_samples < 0:
        raise ValueError("gap_samples must be nonnegative")
    indices = np.arange(n_samples)
    folds = []
    for test in np.array_split(indices, n_folds):
        lower = max(0, int(test[0]) - gap_samples)
        upper = min(n_samples, int(test[-1]) + gap_samples + 1)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[lower:upper] = False
        folds.append((indices[train_mask], test))
    return folds


def group_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create leave-one-group-out folds."""

    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size < 2:
        raise ValueError("groups must be a one-dimensional array")
    folds = []
    for held_out in np.unique(groups):
        test = np.flatnonzero(groups == held_out)
        train = np.flatnonzero(groups != held_out)
        if train.size and test.size:
            folds.append((train, test))
    if len(folds) < 2:
        raise ValueError("At least two nonempty groups are required")
    return folds


def fit_ridge(
    design: np.ndarray,
    response: np.ndarray,
    alpha: float,
    *,
    penalize: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a Gaussian ridge model by solving its normal equations."""

    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float)
    if design.ndim != 2 or response.ndim != 1 or design.shape[0] != response.size:
        raise ValueError("design and response dimensions do not match")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if penalize is None:
        penalize = np.ones(design.shape[1], dtype=float)
    penalize = np.asarray(penalize, dtype=float)
    if penalize.shape != (design.shape[1],):
        raise ValueError("penalize must have one value per design column")
    penalty = np.diag(penalize)
    system = design.T @ design + alpha * penalty
    target = design.T @ response
    try:
        return np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, target, rcond=None)[0]


def fit_grouped_ridge_cv(
    design: np.ndarray,
    response: np.ndarray,
    groups: np.ndarray,
    alphas: np.ndarray,
    *,
    penalize: np.ndarray | None = None,
) -> RidgeCVResult:
    """Select ridge strength by leave-one-group-out R² and refit all samples."""

    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    if design.ndim != 2 or response.ndim != 1 or design.shape[0] != response.size:
        raise ValueError("design and response dimensions do not match")
    if alphas.ndim != 1 or alphas.size == 0 or np.any(~np.isfinite(alphas)) or np.any(alphas < 0):
        raise ValueError("alphas must contain finite nonnegative values")
    if np.any(~np.isfinite(design)) or np.any(~np.isfinite(response)):
        raise ValueError("design and response must be finite")
    folds = group_folds(groups)
    scale = np.sqrt(np.mean(design**2, axis=0))
    scale[(scale == 0) | ~np.isfinite(scale)] = 1.0
    standardized = design / scale
    if penalize is None:
        penalize = np.ones(design.shape[1], dtype=float)
    fold_scores = np.full((alphas.size, len(folds)), np.nan, dtype=float)
    for fold_index, (train, test) in enumerate(folds):
        train_design = standardized[train]
        system_unpenalized = train_design.T @ train_design
        target = train_design.T @ response[train]
        penalty = np.diag(penalize)
        for alpha_index, alpha in enumerate(alphas):
            system = system_unpenalized + float(alpha) * penalty
            try:
                coefficients = np.linalg.solve(system, target)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(system, target, rcond=None)[0]
            fold_scores[alpha_index, fold_index] = r2_score(
                response[test], standardized[test] @ coefficients
            )
    mean_scores = np.nanmean(fold_scores, axis=1)
    best_index = int(np.nanargmax(mean_scores))
    coefficients_scaled = fit_ridge(
        standardized,
        response,
        float(alphas[best_index]),
        penalize=penalize,
    )
    return RidgeCVResult(
        coefficients=coefficients_scaled / scale,
        best_alpha=float(alphas[best_index]),
        alphas=alphas,
        mean_scores=mean_scores,
        fold_scores=fold_scores,
        column_scale=scale,
    )


def reconstruct_kernel(basis_values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Reconstruct a sampled temporal kernel from basis coefficients."""

    basis_values = np.asarray(basis_values, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float).squeeze()
    if basis_values.ndim != 2 or coefficients.ndim != 1:
        raise ValueError("basis_values must be 2D and coefficients must be 1D")
    if basis_values.shape[1] != coefficients.size:
        raise ValueError("coefficient count must match basis columns")
    return basis_values @ coefficients
