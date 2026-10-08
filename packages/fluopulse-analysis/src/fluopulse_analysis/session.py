"""End-to-end Doric FluoPulse and NI-DAQ alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lutaslab_core.session import (
    AlignedSession as CoreAlignedSession,
)
from lutaslab_core.session import (
    ContinuousSignal,
    EventSeries,
    IntervalSeries,
)

from .events import TTLPulses
from .io import FluoPulseRecording, read_doric
from .nidaq import NIDAQRecording, read_nidaq, read_running
from .synchronization import ClockAlignment, fit_clock_alignment


def _resample_uniform(timestamps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate samples onto a strictly uniform grid for Pynapple."""

    timestamps = np.asarray(timestamps, dtype=float)
    values = np.asarray(values, dtype=float)
    if timestamps.ndim != 1 or values.ndim != 1 or timestamps.size != values.size:
        raise ValueError("Timestamps and values must be matching one-dimensional arrays")
    if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("At least two strictly increasing timestamps are required")
    uniform_time = np.linspace(timestamps[0], timestamps[-1], timestamps.size)
    return uniform_time, np.interp(uniform_time, timestamps, values)


@dataclass(frozen=True)
class AlignedSession:
    doric: FluoPulseRecording
    nidaq: NIDAQRecording
    alignment: ClockAlignment
    lifetime_time_nidaq: np.ndarray
    doric_licks: TTLPulses
    nidaq_licks: TTLPulses
    ensure_pulses: TTLPulses
    visual_cue_pulses: TTLPulses
    doric_ensure_pulses: TTLPulses | None
    running_time_nidaq: np.ndarray | None = None
    running_speed: np.ndarray | None = None

    @property
    def nidaq_coverage_mask(self) -> np.ndarray:
        """Lifetime samples that fall inside the recorded NI-DAQ interval."""

        start = float(self.nidaq.timestamps[0])
        stop = float(self.nidaq.timestamps[-1])
        return (self.lifetime_time_nidaq >= start) & (self.lifetime_time_nidaq <= stop)

    @property
    def nidaq_coverage_fraction(self) -> float:
        """Fraction of the Doric lifetime series covered by NI-DAQ behavior."""

        return float(np.mean(self.nidaq_coverage_mask))

    def to_core_session(self, session_id: str | None = None) -> CoreAlignedSession:
        """Return the sensor-independent representation used by shared analyses."""

        continuous = {
            "tau": ContinuousSignal(
                self.lifetime_time_nidaq,
                self.doric.tau_ns,
                "ns",
            ),
            "amplitude": ContinuousSignal(
                self.lifetime_time_nidaq,
                self.doric.amplitude,
                "a.u.",
            ),
            "fit_r_square": ContinuousSignal(
                self.lifetime_time_nidaq,
                self.doric.r_square,
                "",
            ),
        }
        if self.running_time_nidaq is not None and self.running_speed is not None:
            continuous["running_speed"] = ContinuousSignal(
                self.running_time_nidaq,
                self.running_speed,
                "a.u.",
            )
        events = {
            "licks": EventSeries(self.doric_licks.onset_times, "licks"),
            "nidaq_licks": EventSeries(self.nidaq_licks.onset_times, "nidaq_licks"),
            "ensure": EventSeries(self.ensure_pulses.onset_times, "ensure"),
            "visual_cue": EventSeries(
                self.visual_cue_pulses.onset_times,
                "visual_cue",
            ),
        }
        intervals = {
            "ensure": IntervalSeries(
                self.ensure_pulses.onset_times,
                self.ensure_pulses.offset_times,
                "ensure",
            ),
            "visual_cue": IntervalSeries(
                self.visual_cue_pulses.onset_times,
                self.visual_cue_pulses.offset_times,
                "visual_cue",
            ),
        }
        if self.doric_ensure_pulses is not None:
            intervals["doric_ensure"] = IntervalSeries(
                self.doric_ensure_pulses.onset_times,
                self.doric_ensure_pulses.offset_times,
                "doric_ensure",
            )
        return CoreAlignedSession(
            session_id=session_id or self.doric.path.stem,
            continuous=continuous,
            events=events,
            intervals=intervals,
            metadata={
                "sensor": "fluopulse",
                "source_path": str(self.doric.path),
                "clock_drift_ppm": self.alignment.drift_ppm,
                "clock_rms_residual_seconds": self.alignment.rms_residual_seconds,
                "nidaq_coverage_fraction": self.nidaq_coverage_fraction,
            },
        )

    def to_pynapple(self) -> dict[str, Any]:
        try:
            import pynapple as nap
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Install the event-analysis extra with: pip install -e ".[events]"'
            ) from exc

        lifetime_time, tau = _resample_uniform(self.lifetime_time_nidaq, self.doric.tau_ns)
        _, amplitude = _resample_uniform(self.lifetime_time_nidaq, self.doric.amplitude)
        _, fit_r_square = _resample_uniform(self.lifetime_time_nidaq, self.doric.r_square)
        support = nap.IntervalSet(
            start=float(lifetime_time[0]),
            end=float(lifetime_time[-1]),
            time_units="s",
        )
        objects: dict[str, Any] = {
            "tau": nap.Tsd(
                t=lifetime_time,
                d=tau,
                time_units="s",
                time_support=support,
            ),
            "amplitude": nap.Tsd(
                t=lifetime_time,
                d=amplitude,
                time_units="s",
                time_support=support,
            ),
            "fit_r_square": nap.Tsd(
                t=lifetime_time,
                d=fit_r_square,
                time_units="s",
                time_support=support,
            ),
            "licks": nap.Ts(
                t=self.doric_licks.onset_times,
                time_units="s",
                time_support=support,
            ),
            "nidaq_licks": nap.Ts(t=self.nidaq_licks.onset_times, time_units="s"),
            "ensure": nap.Ts(t=self.ensure_pulses.onset_times, time_units="s"),
            "ensure_intervals": nap.IntervalSet(
                start=self.ensure_pulses.onset_times,
                end=self.ensure_pulses.offset_times,
                time_units="s",
            ),
            "visual_cue": nap.Ts(t=self.visual_cue_pulses.onset_times, time_units="s"),
        }
        if self.running_time_nidaq is not None and self.running_speed is not None:
            running_time, running_speed = _resample_uniform(
                self.running_time_nidaq, self.running_speed
            )
            objects["running_speed"] = nap.Tsd(
                t=running_time,
                d=running_speed,
                time_units="s",
            )
        return objects


def process_aligned_session(
    doric_path: str | Path,
    nidaq_path: str | Path,
    *,
    running_path: str | Path | None = None,
    nidaq_threshold: float = 1.5,
    doric_sync_start_index: int = 0,
    nidaq_sync_start_index: int = 0,
) -> AlignedSession:
    """Load one paired session and align Doric time to the NI-DAQ clock."""

    doric = read_doric(doric_path, extract_digital=True)
    nidaq = read_nidaq(nidaq_path)
    if "sync" not in doric.digital_pulses:
        raise ValueError("Doric recording has no configured CAM/sync pulse channel")
    doric_sync = doric.digital_pulses["sync"]
    nidaq_sync = nidaq.pulses("sync", threshold=nidaq_threshold)
    alignment = fit_clock_alignment(
        doric_sync.onset_times,
        nidaq_sync.onset_times,
        source_start_index=doric_sync_start_index,
        target_start_index=nidaq_sync_start_index,
    )

    empty = TTLPulses(
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=int),
        np.array([], dtype=int),
    )
    doric_licks_native = doric.digital_pulses.get("licking", empty)
    doric_licks = TTLPulses(
        alignment.source_to_target(doric_licks_native.onset_times),
        alignment.source_to_target(doric_licks_native.offset_times),
        doric_licks_native.rising_indices,
        doric_licks_native.falling_indices,
    )
    doric_ensure_native = doric.digital_pulses.get("ensure")
    doric_ensure = None
    if doric_ensure_native is not None:
        doric_ensure = TTLPulses(
            alignment.source_to_target(doric_ensure_native.onset_times),
            alignment.source_to_target(doric_ensure_native.offset_times),
            doric_ensure_native.rising_indices,
            doric_ensure_native.falling_indices,
        )

    running_time = None
    running_speed = None
    if running_path is not None:
        running = read_running(running_path)
        if running.timestamps is not None:
            running_time = running.timestamps.copy()
        elif running.speed.size == nidaq_sync.onset_times.size:
            running_time = nidaq_sync.onset_times.copy()
        else:
            duration = nidaq.timestamps[-1] - nidaq.timestamps[0]
            inferred_rate = running.speed.size / duration
            running_time = nidaq.timestamps[0] + np.arange(running.speed.size) / inferred_rate
        running_speed = running.speed.copy()

    return AlignedSession(
        doric=doric,
        nidaq=nidaq,
        alignment=alignment,
        lifetime_time_nidaq=alignment.source_to_target(doric.time),
        doric_licks=doric_licks,
        nidaq_licks=nidaq.pulses("licking", threshold=nidaq_threshold),
        ensure_pulses=nidaq.pulses("ensure", threshold=nidaq_threshold),
        visual_cue_pulses=nidaq.pulses("visual_cue", threshold=nidaq_threshold),
        doric_ensure_pulses=doric_ensure,
        running_time_nidaq=running_time,
        running_speed=running_speed,
    )
