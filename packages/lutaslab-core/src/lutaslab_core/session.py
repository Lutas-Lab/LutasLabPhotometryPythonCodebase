"""Sensor-independent containers for aligned continuous and event data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _vector(values: np.ndarray, name: str, *, allow_nan: bool = False) -> np.ndarray:
    result = np.atleast_1d(np.asarray(values, dtype=float).squeeze())
    invalid = np.isinf(result) if allow_nan else ~np.isfinite(result)
    if result.ndim != 1 or np.any(invalid):
        qualifier = "non-infinite" if allow_nan else "finite"
        raise ValueError(f"{name} must be a {qualifier} one-dimensional array")
    return result


@dataclass(frozen=True)
class ContinuousSignal:
    timestamps: np.ndarray
    values: np.ndarray
    units: str = ""

    def __post_init__(self) -> None:
        timestamps = _vector(self.timestamps, "timestamps")
        values = _vector(self.values, "values", allow_nan=True)
        if timestamps.size != values.size:
            raise ValueError("timestamps and values must have matching lengths")
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class EventSeries:
    timestamps: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        timestamps = _vector(self.timestamps, "timestamps")
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps", timestamps)


@dataclass(frozen=True)
class IntervalSeries:
    starts: np.ndarray
    ends: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        starts = _vector(self.starts, "starts")
        ends = _vector(self.ends, "ends")
        if starts.size != ends.size or np.any(ends < starts):
            raise ValueError("interval starts and ends must be ordered and equal length")
        if starts.size > 1 and np.any(np.diff(starts) <= 0):
            raise ValueError("interval starts must be strictly increasing")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "ends", ends)


@dataclass
class AlignedSession:
    """Common analysis-level session produced by sensor-specific preprocessors."""

    session_id: str
    continuous: dict[str, ContinuousSignal] = field(default_factory=dict)
    events: dict[str, EventSeries] = field(default_factory=dict)
    intervals: dict[str, IntervalSeries] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_pynapple(self) -> dict[str, Any]:
        try:
            import pynapple as nap
        except ImportError as error:
            raise ImportError(
                "Pynapple conversion requires the 'events' optional dependency"
            ) from error
        result: dict[str, Any] = {}
        for name, signal in self.continuous.items():
            result[name] = nap.Tsd(t=signal.timestamps, d=signal.values, time_units="s")
        for name, events in self.events.items():
            result[name] = nap.Ts(t=events.timestamps, time_units="s")
        for name, intervals in self.intervals.items():
            result[name] = nap.IntervalSet(
                start=intervals.starts,
                end=intervals.ends,
                time_units="s",
            )
        return result
