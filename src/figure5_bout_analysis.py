"""Bout-structure models for the processed Figure 5 trials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lutaslab_core.glm import lagged_basis_matrix, raised_cosine_basis

from .figure5_reanalysis import Figure5Trials, build_figure5_design


@dataclass(frozen=True)
class BoutFeatures:
    onset_events: np.ndarray
    size_events: np.ndarray
    occupancy: np.ndarray
    rate: np.ndarray
    bout_count: int


@dataclass(frozen=True)
class Figure5BoutDesign:
    matrix: np.ndarray
    response: np.ndarray
    block_groups: np.ndarray
    trial_numbers: np.ndarray
    column_slices: dict[str, slice]
    penalty_weights: np.ndarray
    bout_counts: np.ndarray


def extract_bout_features(
    lick_events: np.ndarray,
    sample_rate_hz: float,
    *,
    max_interlick_gap_seconds: float = 1.0,
    min_licks: int = 3,
) -> BoutFeatures:
    """Represent lick bouts by onset, size, occupancy, and within-bout rate."""

    lick_events = np.asarray(lick_events, dtype=float)
    if lick_events.ndim != 1:
        raise ValueError("lick_events must be one-dimensional")
    if sample_rate_hz <= 0 or max_interlick_gap_seconds <= 0 or min_licks < 1:
        raise ValueError("invalid bout parameters")
    lick_indices = np.flatnonzero(lick_events > 0)
    onset = np.zeros(lick_events.size)
    size = np.zeros(lick_events.size)
    occupancy = np.zeros(lick_events.size)
    rate = np.zeros(lick_events.size)
    if lick_indices.size == 0:
        return BoutFeatures(onset, size, occupancy, rate, 0)
    max_gap = max_interlick_gap_seconds * sample_rate_hz
    starts = np.r_[0, np.flatnonzero(np.diff(lick_indices) > max_gap) + 1]
    stops = np.r_[starts[1:], lick_indices.size]
    kept = 0
    for start, stop in zip(starts, stops, strict=True):
        count = int(stop - start)
        if count < min_licks:
            continue
        first = int(lick_indices[start])
        last = int(lick_indices[stop - 1])
        duration_seconds = max((last - first + 1) / sample_rate_hz, 1 / sample_rate_hz)
        onset[first] = 1.0
        size[first] = count
        occupancy[first : last + 1] = 1.0
        rate[first : last + 1] = count / duration_seconds
        kept += 1
    return BoutFeatures(onset, size, occupancy, rate, kept)


def build_figure5_bout_design(
    trials: Figure5Trials,
    mouse_id: int,
    *,
    max_interlick_gap_seconds: float = 1.0,
    min_licks: int = 3,
    lick_basis_count: int = 12,
    bout_basis_count: int = 8,
    block_count: int = 5,
) -> Figure5BoutDesign:
    """Augment the primary Figure 5 design with explicit bout structure."""

    base = build_figure5_design(
        trials,
        mouse_id,
        lick_basis_count=lick_basis_count,
        block_count=block_count,
    )
    rows = np.flatnonzero(trials.mouse_ids == mouse_id)
    dt = 1.0 / trials.sample_rate_hz
    bout_basis = raised_cosine_basis((-5.0, 8.0), bout_basis_count, dt)
    bout_lags = np.rint(bout_basis.lag_times / dt).astype(int)
    bout_matrices = []
    bout_counts = []
    for row in rows:
        features = extract_bout_features(
            trials.lick_events[row],
            trials.sample_rate_hz,
            max_interlick_gap_seconds=max_interlick_gap_seconds,
            min_licks=min_licks,
        )
        bout_matrices.append(
            np.column_stack(
                [
                    lagged_basis_matrix(
                        features.onset_events, bout_lags, bout_basis.values
                    ),
                    lagged_basis_matrix(
                        features.size_events, bout_lags, bout_basis.values
                    ),
                    features.occupancy,
                    features.rate,
                ]
            )
        )
        bout_counts.append(features.bout_count)

    matrix = np.column_stack([base.matrix, np.vstack(bout_matrices)])
    start = base.matrix.shape[1]
    slices = dict(base.column_slices)
    slices["bout_onset"] = slice(start, start + bout_basis_count)
    slices["bout_size"] = slice(start + bout_basis_count, start + 2 * bout_basis_count)
    slices["bout_occupancy"] = slice(start + 2 * bout_basis_count, start + 2 * bout_basis_count + 1)
    slices["bout_rate"] = slice(start + 2 * bout_basis_count + 1, start + 2 * bout_basis_count + 2)
    return Figure5BoutDesign(
        matrix=matrix,
        response=base.response,
        block_groups=base.block_groups,
        trial_numbers=base.trial_numbers,
        column_slices=slices,
        penalty_weights=np.r_[base.penalty_weights, np.ones(matrix.shape[1] - start)],
        bout_counts=np.asarray(bout_counts, dtype=int),
    )


def bout_model_columns(design: Figure5BoutDesign, terms: tuple[str, ...]) -> np.ndarray:
    columns = list(range(design.column_slices["intercept"].start, design.column_slices["intercept"].stop))
    for term in terms:
        selection = design.column_slices[term]
        columns.extend(range(selection.start, selection.stop))
    return np.asarray(columns, dtype=int)
