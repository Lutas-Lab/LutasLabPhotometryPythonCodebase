"""Trial-aware reanalysis of the paper's Figure 5 photometry data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lutaslab_core.glm import (
    RidgeCVResult,
    fit_grouped_ridge_cv,
    lagged_basis_matrix,
    r2_score,
    raised_cosine_basis,
    reconstruct_kernel,
)
from lutaslab_core.manifest import load_session_manifest
from lutaslab_core.perievent import extract_perievent_trials, normalize_trials
from scipy.io import loadmat
from scipy.signal import find_peaks

from .save_sessiondata import load_session
from .session_manifest import processed_session_path, resolve_session_channel


@dataclass(frozen=True)
class Figure5Trials:
    mouse_ids: np.ndarray
    trial_numbers: np.ndarray
    photometry: np.ndarray
    lick_events: np.ndarray
    cue_events: np.ndarray
    ensure_events: np.ndarray
    sample_rate_hz: float


@dataclass(frozen=True)
class Figure5Design:
    matrix: np.ndarray
    response: np.ndarray
    block_groups: np.ndarray
    trial_numbers: np.ndarray
    column_slices: dict[str, slice]
    penalty_weights: np.ndarray
    lick_basis_values: np.ndarray
    lick_lag_seconds: np.ndarray
    cue_basis_values: np.ndarray
    cue_lag_seconds: np.ndarray
    ensure_basis_values: np.ndarray
    ensure_lag_seconds: np.ndarray


@dataclass(frozen=True)
class NestedCVResult:
    prediction: np.ndarray
    heldout_r2: float
    fold_r2: np.ndarray
    outer_alphas: np.ndarray
    outer_coefficients: tuple[np.ndarray, ...]
    final_fit: RidgeCVResult


def _rising_edges(row: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    active = np.asarray(row, dtype=float) > threshold
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])


def load_figure5_trials(
    data_root: str | Path, data_file: str | Path | None = None
) -> Figure5Trials:
    """Load the full 20-second processed trials deposited for Figure 5."""

    path = Path(data_file) if data_file is not None else (
        Path(data_root)
        / "Figure 5 - Model GLM"
        / "Figure 5 - d- GLM total licking"
        / "FilteredGLM8sWithEnsureTotalLickingFinalsave.mat"
    )
    contents = loadmat(path, simplify_cells=True)
    data = contents["ConcatData"]
    model = contents["FilteredGLMData"]
    mouse_ids = np.asarray(data["mouseidnumlist"], dtype=int).reshape(-1)
    trial_numbers = np.asarray(data["concatmicetrialnum"], dtype=int).reshape(-1)
    photometry = np.asarray(data["concatmicephotom"], dtype=float)[:, :-1]
    raw_licks = np.asarray(data["concatmicelickmat"], dtype=float)
    sample_count = raw_licks.shape[1]
    trial_count = mouse_ids.size
    if photometry.shape != (trial_count, sample_count):
        raise ValueError("photometry and lick trials do not have matching shapes")

    lick_events = np.zeros_like(raw_licks)
    for row_index, row in enumerate(raw_licks):
        lick_events[row_index, find_peaks(row, height=2)[0]] = 1.0

    cue_analog = np.asarray(data["concatmiceLEDmat"], dtype=float)
    ensure_analog = np.asarray(data["concatmiceensuremat"], dtype=float)
    if cue_analog.shape[0] < trial_count or ensure_analog.shape[0] < trial_count:
        raise ValueError("deposited event matrices contain fewer rows than the trial table")
    cue_events = np.zeros_like(raw_licks)
    ensure_events = np.zeros_like(raw_licks)
    for row_index in range(trial_count):
        cue_onsets = _rising_edges(cue_analog[row_index, :sample_count])
        ensure_onsets = _rising_edges(ensure_analog[row_index, :sample_count])
        cue_events[row_index, cue_onsets] = 1.0
        ensure_events[row_index, ensure_onsets] = 1.0

    return Figure5Trials(
        mouse_ids=mouse_ids,
        trial_numbers=trial_numbers,
        photometry=photometry,
        lick_events=lick_events,
        cue_events=cue_events,
        ensure_events=ensure_events,
        sample_rate_hz=float(model["Fs"]),
    )


def _raw_processed_session_path(data_root: str | Path, session: dict) -> Path:
    """Resolve either the flat reprocessing layout or the standard nested layout."""

    root = Path(data_root)
    mouse = str(session["mouse"])
    date = str(session["date"])
    run = int(session["run"])
    flat_path = root / f"{mouse}-{date}-{run:03d}-processed.npz"
    if flat_path.is_file():
        return flat_path
    return processed_session_path(root, session)


def _event_grid(
    event_times: np.ndarray,
    alignment_times: np.ndarray,
    *,
    window_start: float,
    dt: float,
    sample_count: int,
) -> np.ndarray:
    """Place timestamped events in the nearest sample of each trial."""

    event_times = np.asarray(event_times, dtype=float)
    grid = np.zeros((alignment_times.size, sample_count), dtype=float)
    for row, alignment in enumerate(np.asarray(alignment_times, dtype=float)):
        relative = event_times - alignment
        indices = np.rint((relative - window_start) / dt).astype(int)
        indices = indices[(indices >= 0) & (indices < sample_count)]
        grid[row, np.unique(indices)] = 1.0
    return grid


def load_raw_figure5_trials(
    manifest_path: str | Path,
    processed_root: str | Path,
    *,
    photometry_source: str = "raw465",
    window: tuple[float, float] = (-5.0, 15.0),
    baseline: tuple[float, float] = (-5.0, 0.0),
    dt: float = 0.02,
    postcue_window: tuple[float, float] = (8.0, 10.0),
    boundary_tolerance: float = 0.01,
) -> Figure5Trials:
    """Build Figure 5 trials directly from newly processed raw sessions.

    The full-model trial set follows the paper's fixed programmed-cue rule: a
    trial is included when at least one lick occurs more than 8 and no later
    than 10 seconds after cue onset.  This deliberately does not use the
    detected cue offset, which can be shortened by an imperfect TTL pulse.
    """

    if photometry_source not in {"raw465", "dff"}:
        raise ValueError("photometry_source must be 'raw465' or 'dff'")
    if window[0] >= window[1] or baseline[0] >= baseline[1]:
        raise ValueError("window and baseline must contain increasing values")
    if postcue_window[0] >= postcue_window[1]:
        raise ValueError("postcue_window must contain increasing values")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be finite and nonnegative")

    mouse_ids = []
    trial_numbers = []
    photometry_trials = []
    lick_trials = []
    cue_trials = []
    ensure_trials = []

    sessions = load_session_manifest(manifest_path)
    expected_sample_count = int(np.floor((window[1] - window[0]) / dt + 0.5))
    for mouse_id, manifest_session in enumerate(sessions, start=1):
        channel = resolve_session_channel(manifest_session)
        session_path = _raw_processed_session_path(processed_root, manifest_session)
        session = load_session(session_path)
        time_key = f"photo_time_465_ch{channel}"
        signal_key = (
            f"photometry_465_ch{channel}"
            if photometry_source == "raw465"
            else f"dff_ch{channel}"
        )
        required = (time_key, signal_key, "cue_onset", "lick_times", "solenoid_onset")
        missing = {key for key in required if key not in session}
        if missing:
            raise ValueError(f"{session_path} is missing keys: {sorted(missing)}")

        cue_onsets = np.asarray(session["cue_onset"], dtype=float)
        lick_times = np.asarray(session["lick_times"], dtype=float)
        peri_time, extracted, valid = extract_perievent_trials(
            np.asarray(session[time_key], dtype=float),
            np.asarray(session[signal_key], dtype=float),
            cue_onsets,
            window=window,
            dt=dt,
        )
        extracted = normalize_trials(peri_time, extracted, "zscore", baseline)

        valid_cues = cue_onsets[valid]
        selected = np.asarray(
            [
                np.any(
                    (lick_times - cue > postcue_window[0])
                    & (lick_times - cue <= postcue_window[1] + boundary_tolerance)
                )
                for cue in valid_cues
            ],
            dtype=bool,
        )
        selected_cues = valid_cues[selected]
        selected_trial_numbers = valid[selected] + 1
        selected_photometry = extracted[selected, :expected_sample_count]
        if selected_photometry.shape[1] != expected_sample_count:
            raise ValueError(
                f"{session_path} produced {selected_photometry.shape[1]} samples per trial; "
                f"expected {expected_sample_count}"
            )
        if selected_cues.size == 0:
            raise ValueError(f"{session_path} has no trials satisfying the Figure 5 rule")

        mouse_ids.append(np.full(selected_cues.size, mouse_id, dtype=int))
        trial_numbers.append(selected_trial_numbers.astype(int))
        photometry_trials.append(selected_photometry)
        lick_trials.append(
            _event_grid(
                lick_times,
                selected_cues,
                window_start=window[0],
                dt=dt,
                sample_count=expected_sample_count,
            )
        )
        cue_trials.append(
            _event_grid(
                cue_onsets,
                selected_cues,
                window_start=window[0],
                dt=dt,
                sample_count=expected_sample_count,
            )
        )
        ensure_trials.append(
            _event_grid(
                np.asarray(session["solenoid_onset"], dtype=float),
                selected_cues,
                window_start=window[0],
                dt=dt,
                sample_count=expected_sample_count,
            )
        )

    return Figure5Trials(
        mouse_ids=np.concatenate(mouse_ids),
        trial_numbers=np.concatenate(trial_numbers),
        photometry=np.vstack(photometry_trials),
        lick_events=np.vstack(lick_trials),
        cue_events=np.vstack(cue_trials),
        ensure_events=np.vstack(ensure_trials),
        sample_rate_hz=1.0 / dt,
    )


def contiguous_trial_blocks(trial_count: int, block_count: int = 5) -> np.ndarray:
    """Assign ordered complete trials to approximately equal contiguous blocks."""

    if trial_count < 2:
        raise ValueError("at least two trials are required")
    if block_count < 2:
        raise ValueError("block_count must be at least two")
    block_count = min(block_count, trial_count)
    groups = np.empty(trial_count, dtype=int)
    for block, indices in enumerate(np.array_split(np.arange(trial_count), block_count)):
        groups[indices] = block
    return groups


def build_figure5_design(
    trials: Figure5Trials,
    mouse_id: int,
    *,
    lick_basis_count: int = 12,
    cue_basis_count: int = 8,
    ensure_basis_count: int = 8,
    block_count: int = 5,
) -> Figure5Design:
    """Build trial-reset lick, cue, and Ensure basis predictors for one mouse."""

    rows = np.flatnonzero(trials.mouse_ids == mouse_id)
    if rows.size < 2:
        raise ValueError(f"mouse_id {mouse_id} has fewer than two trials")
    dt = 1.0 / trials.sample_rate_hz
    lick_basis = raised_cosine_basis((-5.0, 8.0), lick_basis_count, dt)
    cue_basis = raised_cosine_basis((0.0, 8.0), cue_basis_count, dt)
    ensure_basis = raised_cosine_basis((0.0, 7.0), ensure_basis_count, dt)
    lick_lags = np.rint(lick_basis.lag_times / dt).astype(int)
    cue_lags = np.rint(cue_basis.lag_times / dt).astype(int)
    ensure_lags = np.rint(ensure_basis.lag_times / dt).astype(int)

    matrices = []
    for row in rows:
        matrices.append(
            np.column_stack(
                [
                    np.ones(trials.photometry.shape[1]),
                    lagged_basis_matrix(
                        trials.lick_events[row], lick_lags, lick_basis.values
                    ),
                    lagged_basis_matrix(
                        trials.cue_events[row], cue_lags, cue_basis.values
                    ),
                    lagged_basis_matrix(
                        trials.ensure_events[row], ensure_lags, ensure_basis.values
                    ),
                ]
            )
        )
    lick_start = 1
    cue_start = lick_start + lick_basis_count
    ensure_start = cue_start + cue_basis_count
    column_slices = {
        "intercept": slice(0, 1),
        "lick": slice(lick_start, cue_start),
        "cue": slice(cue_start, ensure_start),
        "ensure": slice(ensure_start, ensure_start + ensure_basis_count),
    }
    trial_blocks = contiguous_trial_blocks(rows.size, block_count)
    samples_per_trial = trials.photometry.shape[1]
    return Figure5Design(
        matrix=np.vstack(matrices),
        response=trials.photometry[rows].reshape(-1),
        block_groups=np.repeat(trial_blocks, samples_per_trial),
        trial_numbers=trials.trial_numbers[rows],
        column_slices=column_slices,
        penalty_weights=np.r_[0.0, np.ones(ensure_start + ensure_basis_count - 1)],
        lick_basis_values=lick_basis.values,
        lick_lag_seconds=lick_basis.lag_times,
        cue_basis_values=cue_basis.values,
        cue_lag_seconds=cue_basis.lag_times,
        ensure_basis_values=ensure_basis.values,
        ensure_lag_seconds=ensure_basis.lag_times,
    )


def model_columns(design: Figure5Design, terms: Iterable[str]) -> np.ndarray:
    names = ("intercept", *terms)
    columns = []
    for name in names:
        if name not in design.column_slices:
            raise ValueError(f"unknown model term: {name}")
        selection = design.column_slices[name]
        columns.extend(range(selection.start, selection.stop))
    return np.asarray(columns, dtype=int)


def fit_nested_blocked_ridge(
    matrix: np.ndarray,
    response: np.ndarray,
    block_groups: np.ndarray,
    alphas: np.ndarray,
    *,
    penalty_weights: np.ndarray,
) -> NestedCVResult:
    """Tune ridge strength inside each held-out contiguous trial block."""

    matrix = np.asarray(matrix, dtype=float)
    response = np.asarray(response, dtype=float)
    block_groups = np.asarray(block_groups)
    alphas = np.asarray(alphas, dtype=float)
    prediction = np.full(response.size, np.nan)
    unique_blocks = np.unique(block_groups)
    fold_r2 = np.full(unique_blocks.size, np.nan)
    outer_alphas = np.full(unique_blocks.size, np.nan)
    outer_coefficients = []
    for fold_index, held_out in enumerate(unique_blocks):
        test = block_groups == held_out
        train = ~test
        inner = fit_grouped_ridge_cv(
            matrix[train],
            response[train],
            block_groups[train],
            alphas,
            penalize=penalty_weights,
        )
        prediction[test] = matrix[test] @ inner.coefficients
        fold_r2[fold_index] = r2_score(response[test], prediction[test])
        outer_alphas[fold_index] = inner.best_alpha
        outer_coefficients.append(inner.coefficients)
    final_fit = fit_grouped_ridge_cv(
        matrix,
        response,
        block_groups,
        alphas,
        penalize=penalty_weights,
    )
    return NestedCVResult(
        prediction=prediction,
        heldout_r2=r2_score(response, prediction),
        fold_r2=fold_r2,
        outer_alphas=outer_alphas,
        outer_coefficients=tuple(outer_coefficients),
        final_fit=final_fit,
    )


def reconstruct_figure5_kernels(
    design: Figure5Design,
    coefficients: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return sampled lick, cue, and Ensure kernels from a fitted design."""

    coefficients = np.asarray(coefficients, dtype=float)
    return {
        "lick": (
            design.lick_lag_seconds,
            reconstruct_kernel(
                design.lick_basis_values, coefficients[design.column_slices["lick"]]
            ),
        ),
        "cue": (
            design.cue_lag_seconds,
            reconstruct_kernel(
                design.cue_basis_values, coefficients[design.column_slices["cue"]]
            ),
        ),
        "ensure": (
            design.ensure_lag_seconds,
            reconstruct_kernel(
                design.ensure_basis_values,
                coefficients[design.column_slices["ensure"]],
            ),
        ),
    }
