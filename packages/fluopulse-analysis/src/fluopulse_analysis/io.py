"""Read Doric FluoPulse HDF5 recordings without loading all waveforms eagerly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np

from .events import TTLPulses, find_ttl_pulses

SERIES = "/DataAcquisition/FluoPulse/Signals/Series0001"
CALCULATION = f"{SERIES}/Calculation01"
DIGITAL = f"{SERIES}/DigitalIO"
FLUORESCENCE = f"{SERIES}/Fluorescence01"
IRF = f"{SERIES}/IRF01"
CONFIGURATION = "/Configurations/FluoPulse"


def _first_existing(file: h5py.File, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in file:
            return candidate
    raise ValueError(f"Doric file is missing {label}; checked {list(candidates)}")


def _decode_attribute(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _semantic_channel(username: str, dataset_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", username.lower())
    aliases = {
        "campulse": "sync",
        "camera": "sync",
        "synch": "sync",
        "sync": "sync",
        "lickevent": "licking",
        "lick": "licking",
        "licking": "licking",
        "ensuredelivery": "ensure",
        "ensure": "ensure",
        "solenoid": "ensure",
    }
    return aliases.get(normalized, username.strip() or dataset_name)


def _digital_channel_map(file: h5py.File) -> dict[str, str]:
    available = {
        name
        for name in file[DIGITAL].keys()
        if name != "Time" and isinstance(file[f"{DIGITAL}/{name}"], h5py.Dataset)
    }
    mapping: dict[str, str] = {}
    for dataset_name in sorted(available):
        settings_path = f"{CONFIGURATION}/{dataset_name}/Settings"
        username = dataset_name
        if settings_path in file:
            username = _decode_attribute(
                file[settings_path].attrs.get("Username", dataset_name)
            )
        semantic = _semantic_channel(username, dataset_name)
        if semantic in mapping:
            semantic = dataset_name
        mapping[semantic] = dataset_name
    return mapping


@dataclass(frozen=True)
class WaveformBatch:
    sample_indices: np.ndarray
    sample_time: np.ndarray
    waveform_time_ns: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class FluoPulseRecording:
    path: Path
    time: np.ndarray
    tau_ns: np.ndarray
    amplitude: np.ndarray
    chi_square: np.ndarray
    r_square: np.ndarray
    irf_time_ns: np.ndarray
    irf_raw: np.ndarray
    irf_values: np.ndarray
    digital_channels: dict[str, str]
    digital_pulses: dict[str, TTLPulses]
    waveform_points: int
    waveform_values_path: str = f"{FLUORESCENCE}/Values"
    waveform_time_path: str = f"{FLUORESCENCE}/Time"

    @property
    def sampling_frequency(self) -> float:
        return float(1.0 / np.median(np.diff(self.time)))

    def load_waveforms(
        self, indices: np.ndarray | list[int] | None = None
    ) -> WaveformBatch:
        """Load selected raw waveforms; all waveforms are loaded only if requested."""

        if indices is None:
            selected = np.arange(self.time.size, dtype=int)
        else:
            selected = np.asarray(indices, dtype=int).reshape(-1)
        if selected.size == 0:
            raise ValueError("At least one waveform index is required")
        if np.any(selected < 0) or np.any(selected >= self.time.size):
            raise IndexError("Waveform index is outside the recording")

        values = np.empty((selected.size, self.waveform_points), dtype=float)
        relative_time = None
        with h5py.File(self.path, "r") as file:
            value_dataset = file[self.waveform_values_path]
            time_dataset = file[self.waveform_time_path]
            for output_index, sample_index in enumerate(selected):
                start = int(sample_index) * self.waveform_points
                stop = start + self.waveform_points
                values[output_index] = value_dataset[start:stop]
                if relative_time is None:
                    absolute_time = np.asarray(time_dataset[start:stop], dtype=float)
                    relative_time = (absolute_time - absolute_time[0]) * 1e9
        assert relative_time is not None
        return WaveformBatch(selected, self.time[selected], relative_time, values)

    def load_digital_signal(
        self,
        channel: str,
        *,
        start_seconds: float | None = None,
        stop_seconds: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a raw DIO signal, optionally restricted to a time interval."""

        dataset_name = self.digital_channels.get(channel, channel)
        with h5py.File(self.path, "r") as file:
            dataset_path = f"{DIGITAL}/{dataset_name}"
            if dataset_path not in file:
                raise KeyError(f"Unknown digital channel: {channel!r}")
            time = np.asarray(file[f"{DIGITAL}/Time"], dtype=float)
            start = (
                0
                if start_seconds is None
                else int(np.searchsorted(time, start_seconds))
            )
            stop = (
                time.size
                if stop_seconds is None
                else int(np.searchsorted(time, stop_seconds))
            )
            values = np.asarray(file[dataset_path][start:stop], dtype=float)
        return time[start:stop], values


def read_doric(
    path: str | Path,
    *,
    extract_digital: bool = True,
    digital_threshold: float = 0.5,
) -> FluoPulseRecording:
    """Load vendor lifetime outputs, IRF, metadata, and optional TTL events."""

    path = Path(path)
    with h5py.File(path, "r") as file:
        calculation = _first_existing(
            file,
            (f"{SERIES}/Calculation01", f"{SERIES}/Calculation"),
            "calculation group",
        )
        amplitude_path = _first_existing(
            file,
            (f"{calculation}/Amplitude", f"{calculation}/Amplitude01"),
            "amplitude dataset",
        )
        waveform_values_path = _first_existing(
            file,
            (f"{FLUORESCENCE}/Values", f"{SERIES}/AnalogIn/Detector01"),
            "raw waveform dataset",
        )
        waveform_time_path = _first_existing(
            file,
            (f"{FLUORESCENCE}/Time", f"{SERIES}/AnalogIn/Time"),
            "raw waveform time dataset",
        )
        irf_group = _first_existing(
            file,
            (IRF, f"{CONFIGURATION}/IRF"),
            "IRF group",
        )
        required = (
            f"{calculation}/Time",
            f"{calculation}/Tau01",
            f"{calculation}/Chisquare",
            f"{calculation}/Rsquare",
            f"{irf_group}/Time",
            f"{irf_group}/Values",
        )
        missing = [name for name in required if name not in file]
        if missing:
            raise ValueError(f"Doric file is missing datasets: {missing}")

        time = np.asarray(file[f"{calculation}/Time"], dtype=float)
        tau = np.asarray(file[f"{calculation}/Tau01"], dtype=float)
        amplitude = np.asarray(file[amplitude_path], dtype=float)
        chi_square = np.asarray(file[f"{calculation}/Chisquare"], dtype=float)
        r_square = np.asarray(file[f"{calculation}/Rsquare"], dtype=float)
        series = [time, tau, amplitude, chi_square, r_square]
        if any(values.shape != time.shape for values in series):
            raise ValueError("Doric calculation datasets do not have matching shapes")
        if time.ndim != 1 or np.any(np.diff(time) <= 0):
            raise ValueError("Doric calculation time must be strictly increasing")

        waveform_size = int(file[waveform_values_path].size)
        if waveform_size % time.size:
            raise ValueError("Raw waveform count is not divisible by lifetime samples")
        waveform_points = waveform_size // time.size

        irf_time = np.asarray(file[f"{irf_group}/Time"], dtype=float)
        irf_time_ns = (irf_time - irf_time[0]) * 1e9
        irf_values = np.asarray(file[f"{irf_group}/Values"], dtype=float)
        raw_path = f"{irf_group}/Raw"
        irf_raw = (
            np.asarray(file[raw_path], dtype=float)
            if raw_path in file
            else irf_values.copy()
        )
        if not (irf_time.shape == irf_raw.shape == irf_values.shape):
            raise ValueError("Doric IRF datasets do not have matching shapes")

        digital_channels = _digital_channel_map(file) if DIGITAL in file else {}
        pulses: dict[str, TTLPulses] = {}
        if extract_digital and digital_channels:
            digital_time = np.asarray(file[f"{DIGITAL}/Time"], dtype=float)
            for semantic, dataset_name in digital_channels.items():
                signal = np.asarray(file[f"{DIGITAL}/{dataset_name}"], dtype=float)
                pulses[semantic] = find_ttl_pulses(
                    signal,
                    digital_time,
                    threshold=digital_threshold,
                )

    return FluoPulseRecording(
        path=path,
        time=time,
        tau_ns=tau,
        amplitude=amplitude,
        chi_square=chi_square,
        r_square=r_square,
        irf_time_ns=irf_time_ns,
        irf_raw=irf_raw,
        irf_values=irf_values,
        digital_channels=digital_channels,
        digital_pulses=pulses,
        waveform_points=waveform_points,
        waveform_values_path=waveform_values_path,
        waveform_time_path=waveform_time_path,
    )
