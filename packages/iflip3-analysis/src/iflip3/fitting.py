"""Bounded decay fits and semi-linear global lifetime analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares, nnls

from .models import design_matrix, periodic_exgaussian_basis


@dataclass(frozen=True)
class LifetimeFitResult:
    lifetimes: np.ndarray
    amplitudes: np.ndarray
    t0: float
    irf_sigma: float
    background: float
    fitted: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str
    cost: float
    parameter_standard_errors: dict[str, float]


def _default_decay_initial(
    time: np.ndarray, intensity: np.ndarray, n_components: int
) -> dict[str, float]:
    peak = int(np.nanargmax(intensity))
    t0 = float(time[peak] - 0.5)
    total = float(np.sum(np.clip(intensity, 0, None)))
    amplitude = max(float(np.nanmax(intensity)), 1.0) / n_components
    initial: dict[str, float] = {"t0": t0, "sigma": 0.13, "background": 0.0}
    guesses = (
        [0.6, 2.5]
        if n_components == 2
        else [max(total and np.sum(time * np.clip(intensity, 0, None)) / total - t0, 0.2)]
    )
    for index in range(n_components):
        initial[f"amplitude{index + 1}"] = amplitude
        initial[f"tau{index + 1}"] = float(guesses[index])
    return initial


def fit_decay(
    time: np.ndarray,
    intensity: np.ndarray,
    *,
    n_components: int = 1,
    pulse_interval: float = 12.5,
    initial: Mapping[str, float] | None = None,
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    weighting: str = "poisson",
    max_nfev: int = 1000,
) -> LifetimeFitResult:
    """Fit one decay with physical bounds and optional fixed parameters."""

    if n_components not in (1, 2):
        raise ValueError("n_components must be 1 or 2")
    time = np.asarray(time, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if time.shape != intensity.shape or time.ndim != 1:
        raise ValueError("time and intensity must be equal-length vectors")
    params = _default_decay_initial(time, intensity, n_components)
    if initial:
        params.update({key: float(value) for key, value in initial.items()})
    fixed = {key: float(value) for key, value in (fixed or {}).items()}
    params.update(fixed)

    names: list[str] = []
    for index in range(n_components):
        names.extend([f"amplitude{index + 1}", f"tau{index + 1}"])
    names.extend(["t0", "sigma", "background"])
    free = [name for name in names if name not in fixed]
    lower_map = {"t0": float(time.min() - 1.0), "sigma": 0.01, "background": -np.inf}
    upper_map = {"t0": float(time.max()), "sigma": 2.0, "background": np.inf}
    for index in range(n_components):
        lower_map[f"amplitude{index + 1}"] = 0.0
        upper_map[f"amplitude{index + 1}"] = np.inf
        lower_map[f"tau{index + 1}"] = 0.03
        upper_map[f"tau{index + 1}"] = pulse_interval * 4.0
    known_names = set(names)
    unknown_fixed = set(fixed) - known_names
    unknown_bounds = set(bounds or {}) - known_names
    if unknown_fixed or unknown_bounds:
        unknown = sorted(unknown_fixed | unknown_bounds)
        raise ValueError(f"Unknown fit parameter(s): {', '.join(unknown)}")
    for name, limits in (bounds or {}).items():
        if len(limits) != 2:
            raise ValueError(f"Bounds for {name} must contain a lower and upper value")
        lower_value, upper_value = map(float, limits)
        if not np.isfinite(lower_value) or not np.isfinite(upper_value):
            raise ValueError(f"Bounds for {name} must be finite")
        if lower_value >= upper_value:
            raise ValueError(f"Lower bound for {name} must be less than its upper bound")
        lower_map[name] = lower_value
        upper_map[name] = upper_value
    for name, value in fixed.items():
        if value < lower_map[name] or value > upper_map[name]:
            raise ValueError(f"Fixed {name} lies outside its allowed bounds")
    x0 = np.array([params[name] for name in free])
    lower = np.array([lower_map[name] for name in free])
    upper = np.array([upper_map[name] for name in free])
    x0 = np.maximum(
        np.minimum(x0, np.where(np.isfinite(upper), upper - 1e-9, x0)),
        np.where(np.isfinite(lower), lower + 1e-9, x0),
    )

    if weighting == "poisson":
        if np.any(intensity < 0):
            raise ValueError(
                "Poisson weighting is undefined for negative counts; restrict "
                "the lifetime window or use weighting='none'"
            )
        weights = 1.0 / np.sqrt(np.maximum(intensity, 1.0))
    elif weighting == "none":
        weights = np.ones_like(intensity)
    else:
        raise ValueError("weighting must be 'poisson' or 'none'")

    def unpack(values: np.ndarray) -> dict[str, float]:
        current = dict(params)
        current.update(zip(free, values, strict=True))
        return current

    def evaluate(current: Mapping[str, float]) -> np.ndarray:
        model = np.full_like(time, current["background"], dtype=float)
        for index in range(n_components):
            model += current[f"amplitude{index + 1}"] * periodic_exgaussian_basis(
                time,
                current[f"tau{index + 1}"],
                current["t0"],
                current["sigma"],
                pulse_interval,
            )
        return model

    result = least_squares(
        lambda values: (evaluate(unpack(values)) - intensity) * weights,
        x0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        x_scale="jac",
    )
    fitted_params = unpack(result.x)
    fitted = evaluate(fitted_params)

    standard_errors: dict[str, float] = {name: float("nan") for name in names}
    if result.jac.size and result.jac.shape[0] > result.jac.shape[1]:
        try:
            covariance = np.linalg.pinv(result.jac.T @ result.jac)
            variance = 2.0 * result.cost / (result.jac.shape[0] - result.jac.shape[1])
            errors = np.sqrt(np.maximum(np.diag(covariance) * variance, 0.0))
            standard_errors.update(zip(free, errors, strict=True))
        except np.linalg.LinAlgError:
            pass
    for name in fixed:
        standard_errors[name] = 0.0

    components = sorted(
        [
            (fitted_params[f"tau{i + 1}"], fitted_params[f"amplitude{i + 1}"])
            for i in range(n_components)
        ]
    )
    return LifetimeFitResult(
        lifetimes=np.array([item[0] for item in components]),
        amplitudes=np.array([item[1] for item in components]),
        t0=fitted_params["t0"],
        irf_sigma=fitted_params["sigma"],
        background=fitted_params["background"],
        fitted=fitted,
        residuals=intensity - fitted,
        success=bool(result.success),
        message=result.message,
        cost=float(result.cost),
        parameter_standard_errors=standard_errors,
    )


@dataclass(frozen=True)
class RecordingGlobalResult:
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    t0: float
    irf_sigma: float
    n_components: int

    @property
    def component_counts(self) -> np.ndarray:
        return self.coefficients[:, : self.n_components]

    @property
    def component_fractions(self) -> np.ndarray:
        counts = self.component_counts
        totals = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)


@dataclass(frozen=True)
class GlobalFitResult:
    lifetimes: np.ndarray
    recordings: tuple[RecordingGlobalResult, ...]
    success: bool
    message: str
    cost: float
    nfev: int
    shared_t0: bool
    shared_sigma: bool
    normalize: str | None


def _solve_local(
    basis: np.ndarray,
    curves: np.ndarray,
    *,
    nonnegative: bool,
    weighting: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_curves = curves.shape[1]
    coefficients = np.empty((n_curves, basis.shape[1]), dtype=float)
    weights = np.ones_like(curves, dtype=float)
    if weighting == "poisson":
        weights = 1.0 / np.sqrt(np.maximum(curves, 1.0))
    elif weighting != "none":
        raise ValueError("weighting must be 'poisson' or 'none'")

    if not nonnegative and weighting == "none":
        coefficients[:] = np.linalg.lstsq(basis, curves, rcond=None)[0].T
    else:
        for column in range(n_curves):
            weighted_basis = basis * weights[:, column, None]
            weighted_curve = curves[:, column] * weights[:, column]
            if nonnegative:
                coefficients[column] = nnls(weighted_basis, weighted_curve)[0]
            else:
                coefficients[column] = np.linalg.lstsq(weighted_basis, weighted_curve, rcond=None)[
                    0
                ]
    fitted = basis @ coefficients.T
    return coefficients, (fitted - curves) * weights


def fit_global(
    time: np.ndarray,
    decays: np.ndarray | Sequence[np.ndarray],
    *,
    n_components: int = 2,
    pulse_interval: float = 12.5,
    initial_lifetimes: Sequence[float] | None = None,
    initial_t0: float | Sequence[float] | None = None,
    initial_sigma: float | Sequence[float] = 0.15,
    shared_t0: bool = False,
    shared_sigma: bool = False,
    include_background: bool = False,
    nonnegative: bool = True,
    weighting: str = "poisson",
    normalize: str | None = "area",
    max_nfev: int = 100,
) -> GlobalFitResult:
    """Experimentally fit shared nonlinear parameters across decay matrices.

    Lifetimes are global. ``t0`` and IRF width may be global or recording-level.
    Component coefficients are solved linearly for every photometry time point
    at each nonlinear iteration (variable projection / semi-linear fitting).
    With area-normalized bases, component coefficients represent fitted counts.
    """

    if n_components not in (1, 2):
        raise ValueError("n_components must be 1 or 2")
    time = np.asarray(time, dtype=float)
    if isinstance(decays, np.ndarray):
        decay_list = [np.asarray(decays, dtype=float)]
    else:
        decay_list = [np.asarray(item, dtype=float) for item in decays]
    if not decay_list:
        raise ValueError("At least one decay matrix is required")
    for item in decay_list:
        if item.ndim == 1:
            item.shape = (item.size, 1)
        if item.ndim != 2 or item.shape[0] != time.size:
            raise ValueError("Each decay matrix must have shape (bins, curves)")
        if not np.all(np.isfinite(item)):
            raise ValueError("Decay matrices must contain only finite values")
        if weighting == "poisson" and np.any(item < 0):
            raise ValueError(
                "Poisson weighting is undefined for negative corrected bins; "
                "restrict the lifetime window or use weighting='none'"
            )
    n_recordings = len(decay_list)

    if initial_lifetimes is None:
        initial_lifetimes = [2.5] if n_components == 1 else [0.6, 2.5]
    lifetimes0 = np.sort(np.asarray(initial_lifetimes, dtype=float))
    if lifetimes0.shape != (n_components,) or np.any(lifetimes0 <= 0):
        raise ValueError("initial_lifetimes must contain one positive value per component")
    lifetime_params = np.r_[lifetimes0[0], np.diff(lifetimes0)]

    def expand(value: float | Sequence[float] | None, estimate: float) -> np.ndarray:
        if value is None:
            return np.full(n_recordings, estimate, dtype=float)
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            return np.full(n_recordings, float(values), dtype=float)
        if values.shape != (n_recordings,):
            raise ValueError("Recording-level initial values have the wrong length")
        return values

    aggregate = sum(item.sum(axis=1) for item in decay_list)
    estimated_t0 = float(time[int(np.argmax(aggregate))] - 0.5)
    t0_values = expand(initial_t0, estimated_t0)
    sigma_values = expand(initial_sigma, 0.15)
    t0_params = t0_values[:1] if shared_t0 else t0_values
    sigma_params = sigma_values[:1] if shared_sigma else sigma_values
    x0 = np.r_[lifetime_params, t0_params, sigma_params]
    lower = np.r_[
        np.full(n_components, 0.03),
        np.full(t0_params.size, time.min() - 1.0),
        np.full(sigma_params.size, 0.01),
    ]
    upper = np.r_[
        np.full(n_components, pulse_interval * 4.0),
        np.full(t0_params.size, time.max()),
        np.full(sigma_params.size, 2.0),
    ]

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lifetime_steps = values[:n_components]
        lifetimes = np.cumsum(lifetime_steps)
        position = n_components
        t0_raw = values[position : position + t0_params.size]
        position += t0_params.size
        sigma_raw = values[position : position + sigma_params.size]
        t0s = np.repeat(t0_raw, n_recordings) if shared_t0 else t0_raw
        sigmas = np.repeat(sigma_raw, n_recordings) if shared_sigma else sigma_raw
        return lifetimes, t0s, sigmas

    def project(values: np.ndarray, keep: bool = False) -> Any:
        lifetimes, t0s, sigmas = unpack(values)
        residual_parts: list[np.ndarray] = []
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for index, curves in enumerate(decay_list):
            basis = design_matrix(
                time,
                lifetimes,
                t0s[index],
                sigmas[index],
                pulse_interval,
                normalize=normalize,
                include_background=include_background,
            )
            coefficients, weighted_residual = _solve_local(
                basis,
                curves,
                nonnegative=nonnegative,
                weighting=weighting,
            )
            residual_parts.append(weighted_residual.ravel(order="F"))
            if keep:
                fitted = basis @ coefficients.T
                results.append((coefficients, fitted, curves - fitted))
        if keep:
            return lifetimes, t0s, sigmas, results
        return np.concatenate(residual_parts)

    result = least_squares(
        project,
        x0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        x_scale="jac",
        verbose=0,
    )
    lifetimes, t0s, sigmas, projected = project(result.x, keep=True)
    recording_results = tuple(
        RecordingGlobalResult(
            coefficients=values[0],
            fitted=values[1],
            residuals=values[2],
            t0=float(t0s[index]),
            irf_sigma=float(sigmas[index]),
            n_components=n_components,
        )
        for index, values in enumerate(projected)
    )
    return GlobalFitResult(
        lifetimes=lifetimes,
        recordings=recording_results,
        success=bool(result.success),
        message=result.message,
        cost=float(result.cost),
        nfev=int(result.nfev),
        shared_t0=shared_t0,
        shared_sigma=shared_sigma,
        normalize=normalize,
    )
