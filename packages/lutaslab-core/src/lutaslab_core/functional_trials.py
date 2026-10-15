"""Validated trial-by-time datasets for functional outcome models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _one_dimensional(values: Any, name: str, size: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size != size:
        raise ValueError(f"{name} must contain one value per trial")
    return result


@dataclass(frozen=True)
class FunctionalTrialDataset:
    """A complete, common-grid trial matrix plus its repeated-measures identity."""

    time: np.ndarray
    outcome: np.ndarray
    mouse: np.ndarray
    session: np.ndarray
    trial_index: np.ndarray
    alignment_time: np.ndarray
    scalar_covariates: Mapping[str, np.ndarray] = field(default_factory=dict)
    exclusions: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        outcome = np.asarray(self.outcome, dtype=float)
        if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
            raise ValueError("time must be a finite one-dimensional grid")
        if np.any(np.diff(time) <= 0):
            raise ValueError("time must be strictly increasing")
        if outcome.ndim != 2 or outcome.shape[1] != time.size:
            raise ValueError("outcome must have shape (trials, timepoints)")
        if outcome.shape[0] == 0:
            raise ValueError("outcome must contain at least one trial")
        if not np.all(np.isfinite(outcome)):
            raise ValueError("FLMM outcomes must be complete and finite")

        n_trials = outcome.shape[0]
        mouse = _one_dimensional(self.mouse, "mouse", n_trials).astype(str)
        session = _one_dimensional(self.session, "session", n_trials).astype(str)
        trial_index = _one_dimensional(
            self.trial_index, "trial_index", n_trials
        ).astype(int)
        alignment_time = _one_dimensional(
            self.alignment_time, "alignment_time", n_trials
        ).astype(float)
        if np.any(trial_index < 1):
            raise ValueError("trial_index values must be positive")
        if not np.all(np.isfinite(alignment_time)):
            raise ValueError("alignment_time values must be finite")
        if np.any(mouse == "") or np.any(session == ""):
            raise ValueError("mouse and session identities cannot be blank")

        covariates: dict[str, np.ndarray] = {}
        reserved = {"mouse", "session", "trial_index", "alignment_time", "outcome"}
        for raw_name, values in self.scalar_covariates.items():
            name = str(raw_name)
            if not name.isidentifier() or name in reserved:
                raise ValueError(f"Invalid scalar covariate name: {name!r}")
            covariates[name] = _one_dimensional(values, name, n_trials)

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "mouse", mouse)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "trial_index", trial_index)
        object.__setattr__(self, "alignment_time", alignment_time)
        object.__setattr__(self, "scalar_covariates", covariates)
        object.__setattr__(self, "exclusions", tuple(dict(row) for row in self.exclusions))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_trials(self) -> int:
        return int(self.outcome.shape[0])

    @property
    def n_timepoints(self) -> int:
        return int(self.outcome.shape[1])

    @property
    def n_mice(self) -> int:
        return int(np.unique(self.mouse).size)

    @property
    def n_sessions(self) -> int:
        return int(np.unique(self.session).size)

    def summary(self) -> dict[str, Any]:
        """Return JSON-ready dimensions, levels, and preflight warnings."""

        levels = {
            name: sorted({str(value) for value in values})
            for name, values in self.scalar_covariates.items()
            if values.dtype.kind in "OUSb"
        }
        level_counts = {
            name: {
                level: {
                    "trials": int(np.sum(values.astype(str) == level)),
                    "mice": int(
                        np.unique(self.mouse[values.astype(str) == level]).size
                    ),
                    "sessions": int(
                        np.unique(self.session[values.astype(str) == level]).size
                    ),
                }
                for level in category_levels
            }
            for name, category_levels in levels.items()
            for values in (self.scalar_covariates[name],)
        }
        warnings = []
        if self.n_mice < 5:
            warnings.append(
                "Fewer than five mice are present; random-effect and population "
                "inference may be unstable."
            )
        if self.n_sessions < 3:
            warnings.append("Fewer than three sessions are present.")
        return {
            "n_trials": self.n_trials,
            "n_timepoints": self.n_timepoints,
            "n_mice": self.n_mice,
            "n_sessions": self.n_sessions,
            "excluded_trials": len(self.exclusions),
            "levels": levels,
            "level_counts": level_counts,
            "warnings": warnings,
        }


__all__ = ["FunctionalTrialDataset"]
