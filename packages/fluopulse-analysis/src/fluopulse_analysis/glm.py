"""Adaptive Ensure convolutional GLM built on shared modeling primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from lutaslab_core.glm import (
    RidgeCVResult,
    TemporalBasis,
    convolve_basis,
    fit_grouped_ridge_cv,
    raised_cosine_basis,
    reconstruct_kernel,
)


@dataclass(frozen=True)
class AdaptiveEnsureDesign:
    time: np.ndarray
    matrix: np.ndarray
    lick_basis: TemporalBasis
    ensure_basis: TemporalBasis
    column_slices: dict[str, slice]
    penalty_weights: np.ndarray


def build_adaptive_ensure_design(
    time: np.ndarray,
    lick_counts: np.ndarray,
    ensure_counts: np.ndarray,
    *,
    lick_kernel_seconds: float = 10.0,
    ensure_kernel_seconds: float = 45.0,
    lick_basis_count: int = 8,
    ensure_basis_count: int = 14,
) -> AdaptiveEnsureDesign:
    """Construct the drift, lick, and adapting Ensure design used by FluoPulse."""

    time = np.asarray(time, dtype=float)
    licks = np.asarray(lick_counts, dtype=float)
    ensures = np.asarray(ensure_counts, dtype=float)
    if time.ndim != 1 or licks.ndim != 1 or ensures.ndim != 1:
        raise ValueError("time and event counts must be one-dimensional")
    if not (time.size == licks.size == ensures.size) or time.size < 2:
        raise ValueError("time and event counts must have matching nontrivial lengths")
    differences = np.diff(time)
    if np.any(~np.isfinite(time)) or np.any(differences <= 0):
        raise ValueError("time must be finite and strictly increasing")
    dt = float(np.median(differences))
    if not np.allclose(differences, dt, rtol=1e-4, atol=1e-9):
        raise ValueError("time must be approximately uniform")
    if np.any(~np.isfinite(licks)) or np.any(~np.isfinite(ensures)):
        raise ValueError("event counts must be finite")

    lick_basis = raised_cosine_basis((0.0, lick_kernel_seconds), lick_basis_count, dt)
    ensure_basis = raised_cosine_basis((0.0, ensure_kernel_seconds), ensure_basis_count, dt)
    event_indices = np.flatnonzero(ensures > 0)
    first_ensure = np.zeros_like(ensures)
    progress = np.zeros_like(ensures)
    if event_indices.size:
        first_ensure[event_indices[0]] = ensures[event_indices[0]]
    if event_indices.size > 1:
        progress[event_indices[1:]] = np.linspace(0.0, 1.0, event_indices.size - 1)

    components = {
        "lick": convolve_basis(licks, lick_basis.values),
        "ensure": convolve_basis(ensures, ensure_basis.values),
        "first_ensure": convolve_basis(first_ensure, ensure_basis.values),
        "ensure_linear": convolve_basis(ensures * progress, ensure_basis.values),
        "ensure_quadratic": convolve_basis(
            ensures * progress**2,
            ensure_basis.values,
        ),
    }
    normalized_time = (time - time[0]) / max(float(time[-1] - time[0]), dt)
    matrices = [
        np.ones((time.size, 1)),
        normalized_time[:, None],
        (normalized_time**2)[:, None],
    ]
    column_slices = {"drift": slice(0, 3)}
    offset = 3
    for name, component in components.items():
        matrices.append(component)
        column_slices[name] = slice(offset, offset + component.shape[1])
        offset += component.shape[1]
    matrix = np.column_stack(matrices)
    penalty = np.ones(matrix.shape[1], dtype=float)
    penalty[0] = 0.0
    return AdaptiveEnsureDesign(
        time=time,
        matrix=matrix,
        lick_basis=lick_basis,
        ensure_basis=ensure_basis,
        column_slices=column_slices,
        penalty_weights=penalty,
    )


def fit_adaptive_ensure_glm(
    designs: Sequence[AdaptiveEnsureDesign],
    outcomes: Sequence[np.ndarray],
    group_labels: Sequence[str],
    alphas: np.ndarray,
) -> RidgeCVResult:
    """Fit several recordings with leave-group-out ridge selection."""

    if not designs or len(designs) != len(outcomes) or len(designs) != len(group_labels):
        raise ValueError("designs, outcomes, and group_labels must have equal nonzero length")
    column_count = designs[0].matrix.shape[1]
    if any(design.matrix.shape[1] != column_count for design in designs):
        raise ValueError("all designs must have the same columns")
    response_rows = []
    group_rows = []
    for design, outcome, label in zip(designs, outcomes, group_labels, strict=True):
        outcome = np.asarray(outcome, dtype=float)
        if outcome.shape != (design.matrix.shape[0],):
            raise ValueError("each outcome must match its design rows")
        response_rows.append(outcome)
        group_rows.append(np.full(outcome.size, label, dtype=object))
    return fit_grouped_ridge_cv(
        np.vstack([design.matrix for design in designs]),
        np.concatenate(response_rows),
        np.concatenate(group_rows),
        alphas,
        penalize=designs[0].penalty_weights,
    )


def reconstruct_adaptive_kernels(
    design: AdaptiveEnsureDesign,
    coefficients: np.ndarray,
    *,
    progress_levels: Sequence[float] = (0.0, 0.5, 1.0),
) -> dict[str, np.ndarray]:
    """Reconstruct lick, first-Ensure, and subsequent adaptation kernels."""

    coefficients = np.asarray(coefficients, dtype=float)
    slices = design.column_slices
    ensure_base = coefficients[slices["ensure"]]
    first_delta = coefficients[slices["first_ensure"]]
    linear = coefficients[slices["ensure_linear"]]
    quadratic = coefficients[slices["ensure_quadratic"]]
    kernels = {
        "lick": reconstruct_kernel(
            design.lick_basis.values,
            coefficients[slices["lick"]],
        ),
        "ensure_first": reconstruct_kernel(
            design.ensure_basis.values,
            ensure_base + first_delta,
        ),
    }
    for level in progress_levels:
        kernels[f"ensure_progress_{float(level):g}"] = reconstruct_kernel(
            design.ensure_basis.values,
            ensure_base + float(level) * linear + float(level) ** 2 * quadratic,
        )
    return kernels
