"""iFLiP3 NI-DAQ compatibility API backed by :mod:`lutaslab_core`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lutaslab_core.events import TTLPulses
from lutaslab_core.events import find_ttl_pulses as _find_ttl_pulses
from lutaslab_core.nidaq import (
    DEFAULT_CHANNEL_ROWS,
    NIDAQRecording,
    nidaq_from_mapping,
    running_from_mapping,
)
from scipy.io import loadmat


@dataclass(frozen=True)
class RunningData:
    """Legacy iFLiP3 view of shared running data."""

    path: Path
    speed: np.ndarray
    position: np.ndarray | None


def read_nidaq(path: str | Path, *, channel_rows=None) -> NIDAQRecording:
    """Load NI-DAQ arrays locally and validate them with the shared core."""

    values = loadmat(path, variable_names=["data", "timestamps", "Fs"], squeeze_me=True)
    return nidaq_from_mapping(path, values, channel_rows=channel_rows)


def read_running(path: str | Path) -> RunningData:
    """Load running arrays locally and retain the established iFLiP3 API."""

    values = loadmat(path, variable_names=["speed", "position"], squeeze_me=True)
    running = running_from_mapping(path, values)
    return RunningData(running.path, running.speed, running.position)


def find_ttl_pulses(
    signal: np.ndarray,
    timestamps: np.ndarray,
    *,
    threshold: float = 1.5,
    min_width_seconds: float | None = None,
    max_width_seconds: float | None = None,
) -> TTLPulses:
    """Detect internal complete pulses using the shared implementation.

    Historically iFLiP3 ignored a signal already high at the first sample or
    still high at the final sample. The explicit boundary option preserves
    that behavior during migration.
    """

    return _find_ttl_pulses(
        signal,
        timestamps,
        threshold=threshold,
        min_width_seconds=min_width_seconds,
        max_width_seconds=max_width_seconds,
        include_boundary_pulses=False,
    )


__all__ = [
    "DEFAULT_CHANNEL_ROWS",
    "NIDAQRecording",
    "RunningData",
    "TTLPulses",
    "find_ttl_pulses",
    "read_nidaq",
    "read_running",
]
