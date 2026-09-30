"""FluoPulse NI-DAQ compatibility API backed by :mod:`lutaslab_core`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lutaslab_core.nidaq import (
    DEFAULT_CHANNEL_ROWS,
    NIDAQRecording,
    read_nidaq,
    read_running as _read_running,
)


@dataclass(frozen=True)
class RunningData:
    """Legacy FluoPulse view of shared running data."""

    speed: np.ndarray
    position: np.ndarray | None
    timestamps: np.ndarray | None


def read_running(path: str | Path) -> RunningData:
    """Read running data and retain the established FluoPulse return shape."""

    running = _read_running(path)
    return RunningData(running.speed, running.position, running.timestamps)


__all__ = [
    "DEFAULT_CHANNEL_ROWS",
    "NIDAQRecording",
    "RunningData",
    "read_nidaq",
    "read_running",
]
