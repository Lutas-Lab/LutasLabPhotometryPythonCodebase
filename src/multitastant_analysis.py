"""Delivery-aligned multitastant models with behavioral nuisance terms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks

from lutaslab_core.glm import lagged_basis_matrix, raised_cosine_basis, reconstruct_kernel

from .figure5_bout_analysis import extract_bout_features
from .figure5_reanalysis import contiguous_trial_blocks


@dataclass(frozen=True)
class TastantTrials:
    mouse_names: np.ndarray
    photometry: np.ndarray
    lick_events: np.ndarray
    delivery_events: np.ndarray
    sample_rate_hz: float


@dataclass(frozen=True)
class TastantDesign:
    matrix: np.ndarray
    response: np.ndarray
    groups: np.ndarray
    slices: dict[str, slice]
    penalty_weights: np.ndarray
    delivery_basis_values: np.ndarray
    delivery_lag_seconds: np.ndarray
    trial_count: int


@dataclass(frozen=True)
class PairedTastantDesign:
    matrix: np.ndarray
    response: np.ndarray
    groups: np.ndarray
    slices: dict[str, slice]
    penalty_weights: np.ndarray
    delivery_basis_values: np.ndarray
    delivery_lag_seconds: np.ndarray


def _rising_edges(row: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    active = np.asarray(row, dtype=float) > threshold
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])


def load_tastant_trials(path: str | Path) -> TastantTrials:
    """Load one delivery-aligned ConcatData file and retain recorded TTLs."""

    data = loadmat(path, simplify_cells=True)["ConcatData"]
    numeric_ids = np.asarray(data["mouseidnumlist"], dtype=int).reshape(-1)
    names = np.atleast_1d(data["concatmiceIDname"]).astype(str)
    if np.min(numeric_ids) < 1 or np.max(numeric_ids) > names.size:
        raise ValueError("mouse IDs cannot be mapped to deposited mouse names")
    mouse_names = names[numeric_ids - 1]
    photometry = np.asarray(data["concatmicephotom"], dtype=float)[:, :-1]
    raw_licks = np.asarray(data["concatmicelickmat"], dtype=float)
    trial_count, sample_count = raw_licks.shape
    if photometry.shape != raw_licks.shape:
        raise ValueError("photometry and lick matrices do not match")
    lick_events = np.zeros_like(raw_licks)
    for row_index, row in enumerate(raw_licks):
        lick_events[row_index, find_peaks(row, height=2)[0]] = 1.0
    analog = np.asarray(data["concatmiceensuremat"], dtype=float)
    if analog.shape[0] < trial_count:
        raise ValueError("delivery matrix has fewer rows than the trial table")
    delivery_events = np.zeros_like(raw_licks)
    for row_index in range(trial_count):
        delivery_events[row_index, _rising_edges(analog[row_index, :sample_count])] = 1.0
    return TastantTrials(
        mouse_names=mouse_names,
        photometry=photometry,
        lick_events=lick_events,
        delivery_events=delivery_events,
        sample_rate_hz=float(data["freq"]),
    )


def build_tastant_design(
    trials: TastantTrials,
    mouse_name: str,
    *,
    block_count: int = 5,
    lick_basis_count: int = 12,
    bout_basis_count: int = 8,
    delivery_basis_count: int = 12,
    max_interlick_gap_seconds: float = 1.0,
    min_licks: int = 3,
) -> TastantDesign:
    """Build lick, bout, and delivery terms for one mouse and tastant."""

    rows = np.flatnonzero(trials.mouse_names == mouse_name)
    if rows.size < 3:
        raise ValueError(f"{mouse_name} has fewer than three trials")
    dt = 1.0 / trials.sample_rate_hz
    lick_basis = raised_cosine_basis((-5.0, 8.0), lick_basis_count, dt)
    bout_basis = raised_cosine_basis((-5.0, 8.0), bout_basis_count, dt)
    delivery_basis = raised_cosine_basis((0.0, 10.0), delivery_basis_count, dt)
    lick_lags = np.rint(lick_basis.lag_times / dt).astype(int)
    bout_lags = np.rint(bout_basis.lag_times / dt).astype(int)
    delivery_lags = np.rint(delivery_basis.lag_times / dt).astype(int)
    matrices = []
    for row in rows:
        bouts = extract_bout_features(
            trials.lick_events[row],
            trials.sample_rate_hz,
            max_interlick_gap_seconds=max_interlick_gap_seconds,
            min_licks=min_licks,
        )
        matrices.append(
            np.column_stack(
                [
                    np.ones(trials.photometry.shape[1]),
                    lagged_basis_matrix(trials.lick_events[row], lick_lags, lick_basis.values),
                    lagged_basis_matrix(bouts.onset_events, bout_lags, bout_basis.values),
                    lagged_basis_matrix(bouts.size_events, bout_lags, bout_basis.values),
                    bouts.occupancy,
                    bouts.rate,
                    lagged_basis_matrix(
                        trials.delivery_events[row], delivery_lags, delivery_basis.values
                    ),
                ]
            )
        )
    lick_start = 1
    bout_start = lick_start + lick_basis_count
    bout_stop = bout_start + 2 * bout_basis_count + 2
    delivery_stop = bout_stop + delivery_basis_count
    slices = {
        "intercept": slice(0, 1),
        "lick": slice(lick_start, bout_start),
        "bout": slice(bout_start, bout_stop),
        "delivery": slice(bout_stop, delivery_stop),
    }
    groups = contiguous_trial_blocks(rows.size, block_count=min(block_count, rows.size))
    return TastantDesign(
        matrix=np.vstack(matrices),
        response=trials.photometry[rows].reshape(-1),
        groups=np.repeat(groups, trials.photometry.shape[1]),
        slices=slices,
        penalty_weights=np.r_[0.0, np.ones(delivery_stop - 1)],
        delivery_basis_values=delivery_basis.values,
        delivery_lag_seconds=delivery_basis.lag_times,
        trial_count=rows.size,
    )


def build_paired_tastant_design(first: TastantDesign, second: TastantDesign) -> PairedTastantDesign:
    """Combine two tastants with condition-specific behavior and delivery terms."""

    if first.matrix.shape[1] != second.matrix.shape[1]:
        raise ValueError("tastant designs must have matching columns")
    behavior_columns = np.r_[np.arange(first.slices["lick"].start, first.slices["lick"].stop), np.arange(first.slices["bout"].start, first.slices["bout"].stop)]
    delivery_columns = np.arange(first.slices["delivery"].start, first.slices["delivery"].stop)
    behavior = np.vstack([first.matrix[:, behavior_columns], second.matrix[:, behavior_columns]])
    delivery = np.vstack([first.matrix[:, delivery_columns], second.matrix[:, delivery_columns]])
    condition = np.concatenate(
        [np.full(first.response.size, -0.5), np.full(second.response.size, 0.5)]
    )
    matrix = np.column_stack(
        [
            np.ones(condition.size),
            condition,
            behavior,
            behavior * condition[:, None],
            delivery,
            delivery * condition[:, None],
        ]
    )
    behavior_start = 2
    behavior_stop = behavior_start + behavior.shape[1]
    behavior_interaction_stop = behavior_stop + behavior.shape[1]
    delivery_stop = behavior_interaction_stop + delivery.shape[1]
    delivery_interaction_stop = delivery_stop + delivery.shape[1]
    slices = {
        "intercept": slice(0, 1),
        "condition": slice(1, 2),
        "behavior": slice(behavior_start, behavior_stop),
        "behavior_interaction": slice(behavior_stop, behavior_interaction_stop),
        "delivery": slice(behavior_interaction_stop, delivery_stop),
        "delivery_interaction": slice(delivery_stop, delivery_interaction_stop),
    }
    return PairedTastantDesign(
        matrix=matrix,
        response=np.concatenate([first.response, second.response]),
        groups=np.concatenate([first.groups, second.groups]),
        slices=slices,
        penalty_weights=np.r_[0.0, 0.0, np.ones(matrix.shape[1] - 2)],
        delivery_basis_values=first.delivery_basis_values,
        delivery_lag_seconds=first.delivery_lag_seconds,
    )


def paired_model_columns(design: PairedTastantDesign, terms: tuple[str, ...]) -> np.ndarray:
    columns = []
    for term in terms:
        selection = design.slices[term]
        columns.extend(range(selection.start, selection.stop))
    return np.asarray(columns, dtype=int)


def condition_delivery_kernels(
    design: PairedTastantDesign,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = coefficients[design.slices["delivery"]]
    difference = coefficients[design.slices["delivery_interaction"]]
    first = reconstruct_kernel(design.delivery_basis_values, base - 0.5 * difference)
    second = reconstruct_kernel(design.delivery_basis_values, base + 0.5 * difference)
    return first, second, second - first
