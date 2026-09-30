"""Load laboratory NI-DAQ and running files using explicit hardware mappings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .events import TTLPulses, find_ttl_pulses


DEFAULT_CHANNEL_ROWS: dict[str, int] = {
    "photoreceiver_1": 0,
    "sync": 1,
    "photoreceiver_2": 2,
    "licking": 3,
    "visual_cue": 4,
    "ttl_465": 5,
    "ttl_405": 6,
    "ensure": 7,
}


@dataclass(frozen=True)
class NIDAQRecording:
    path: Path
    data: np.ndarray
    timestamps: np.ndarray
    sampling_frequency: float
    channel_rows: Mapping[str, int]

    def signal(self, name: str) -> np.ndarray:
        if name not in self.channel_rows:
            raise KeyError(f"Unknown NI-DAQ channel: {name!r}")
        return self.data[self.channel_rows[name]]

    def pulses(
        self,
        name: str,
        *,
        threshold: float | None = 1.5,
        min_width_seconds: float | None = None,
        max_width_seconds: float | None = None,
    ) -> TTLPulses:
        return find_ttl_pulses(
            self.signal(name),
            self.timestamps,
            threshold=threshold,
            min_width_seconds=min_width_seconds,
            max_width_seconds=max_width_seconds,
        )


@dataclass(frozen=True)
class RunningData:
    path: Path
    speed: np.ndarray
    position: np.ndarray | None = None
    timestamps: np.ndarray | None = None


def read_nidaq(
    path: str | Path,
    *,
    channel_rows: Mapping[str, int] | None = None,
) -> NIDAQRecording:
    """Read a channel-by-sample NI-DAQ MAT file without trusting name metadata."""

    path = Path(path)
    values = loadmat(path, variable_names=["data", "timestamps", "Fs"], squeeze_me=True)
    return nidaq_from_mapping(path, values, channel_rows=channel_rows)


def nidaq_from_mapping(
    path: str | Path,
    values: Mapping[str, object],
    *,
    channel_rows: Mapping[str, int] | None = None,
) -> NIDAQRecording:
    """Validate NI-DAQ arrays already loaded from a MAT-like mapping."""

    path = Path(path)
    missing = {"data", "timestamps", "Fs"}.difference(values)
    if missing:
        raise ValueError(f"NI-DAQ file is missing variables: {sorted(missing)}")
    data = np.asarray(values["data"], dtype=float)
    timestamps = np.atleast_1d(np.asarray(values["timestamps"], dtype=float).squeeze())
    sampling_frequency = float(np.asarray(values["Fs"], dtype=float).squeeze())
    rows = dict(DEFAULT_CHANNEL_ROWS if channel_rows is None else channel_rows)
    if data.ndim != 2:
        raise ValueError("NI-DAQ data must be a two-dimensional channel-by-sample array")
    if timestamps.ndim != 1 or data.shape[1] != timestamps.size:
        raise ValueError("NI-DAQ timestamps must match the data sample dimension")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("NI-DAQ sampling frequency must be finite and positive")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("NI-DAQ timestamps must be finite and strictly increasing")
    if not rows or min(rows.values()) < 0 or max(rows.values()) >= data.shape[0]:
        raise ValueError("NI-DAQ channel mapping refers to unavailable data rows")
    return NIDAQRecording(path, data, timestamps, sampling_frequency, rows)


def read_running(path: str | Path) -> RunningData:
    """Read running speed and optional position/timestamps from a MAT file."""

    path = Path(path)
    values = loadmat(
        path,
        variable_names=["speed", "position", "timestamps", "time"],
        squeeze_me=True,
    )
    return running_from_mapping(path, values)


def running_from_mapping(
    path: str | Path,
    values: Mapping[str, object],
) -> RunningData:
    """Validate running arrays already loaded from a MAT-like mapping."""

    path = Path(path)
    if "speed" not in values:
        raise ValueError("Running MAT file is missing 'speed'")
    speed = np.atleast_1d(np.asarray(values["speed"], dtype=float).squeeze())
    if speed.ndim != 1 or not np.all(np.isfinite(speed)):
        raise ValueError("Running speed must be a finite one-dimensional array")
    position = None
    if "position" in values:
        position = np.atleast_1d(np.asarray(values["position"], dtype=float).squeeze())
        if position.shape != speed.shape or not np.all(np.isfinite(position)):
            raise ValueError("Running position must be finite and match speed")
    timestamps = None
    for name in ("timestamps", "time"):
        if name in values:
            timestamps = np.atleast_1d(np.asarray(values[name], dtype=float).squeeze())
            if timestamps.shape != speed.shape or np.any(np.diff(timestamps) <= 0):
                raise ValueError("Running timestamps must increase and match speed")
            break
    return RunningData(path, speed, position, timestamps)
