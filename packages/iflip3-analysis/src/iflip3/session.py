"""End-to-end alignment of iFLiP lifetime data with NI-DAQ behavior."""

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

from .io import read_iflip3
from .nidaq import (
    NIDAQRecording,
    RunningData,
    TTLPulses,
    find_ttl_pulses,
    read_nidaq,
    read_running,
)
from .preprocessing import average_background, calculate_mpet
from .synchronization import ClockAlignment, external_marker_mask, fit_clock_alignment


@dataclass(frozen=True)
class AlignedSession:
    """Lifetime and behavioral data represented on the NI-DAQ clock."""

    lifetime_time_nidaq: np.ndarray
    lifetime_time_iflip: np.ndarray
    mpet_ns: np.ndarray
    raw_intensity_counts: np.ndarray
    lick_times_nidaq: np.ndarray
    visual_cue_pulses: TTLPulses
    ensure_pulses: TTLPulses
    sync_pulses: TTLPulses
    alignment: ClockAlignment
    running_time_nidaq: np.ndarray | None = None
    running_speed: np.ndarray | None = None

    def to_core_session(self, session_id: str = "iflip3-session") -> CoreAlignedSession:
        """Return the sensor-independent representation used by shared analyses."""

        continuous = {
            "mpet": ContinuousSignal(
                self.lifetime_time_nidaq,
                self.mpet_ns,
                "ns",
            ),
            "raw_intensity": ContinuousSignal(
                self.lifetime_time_nidaq,
                self.raw_intensity_counts,
                "counts",
            ),
        }
        if self.running_time_nidaq is not None and self.running_speed is not None:
            continuous["running_speed"] = ContinuousSignal(
                self.running_time_nidaq,
                self.running_speed,
                "a.u.",
            )
        return CoreAlignedSession(
            session_id=session_id,
            continuous=continuous,
            events={
                "licks": EventSeries(self.lick_times_nidaq, "licks"),
                "visual_cue": EventSeries(
                    self.visual_cue_pulses.onset_times,
                    "visual_cue",
                ),
                "ensure": EventSeries(self.ensure_pulses.onset_times, "ensure"),
                "sync": EventSeries(self.sync_pulses.onset_times, "sync"),
            },
            intervals={
                "visual_cue": IntervalSeries(
                    self.visual_cue_pulses.onset_times,
                    self.visual_cue_pulses.offset_times,
                    "visual_cue",
                ),
                "ensure": IntervalSeries(
                    self.ensure_pulses.onset_times,
                    self.ensure_pulses.offset_times,
                    "ensure",
                ),
                "sync": IntervalSeries(
                    self.sync_pulses.onset_times,
                    self.sync_pulses.offset_times,
                    "sync",
                ),
            },
            metadata={
                "sensor": "iflip3",
                "clock_drift_ppm": self.alignment.drift_ppm,
                "clock_rms_residual_seconds": self.alignment.rms_residual_seconds,
            },
        )

    def to_pynapple(self) -> dict[str, Any]:
        """Return Pynapple objects for lifetime, QC, behavior, and running."""

        try:
            import pynapple as nap
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                'Pynapple is optional; install it with: pip install -e ".[events]"'
            ) from exc

        support = nap.IntervalSet(
            start=float(self.lifetime_time_nidaq[0]),
            end=float(self.lifetime_time_nidaq[-1]),
            time_units="s",
        )
        objects: dict[str, Any] = {
            "mpet": nap.Tsd(
                t=self.lifetime_time_nidaq,
                d=self.mpet_ns,
                time_units="s",
                time_support=support,
            ),
            "raw_intensity": nap.Tsd(
                t=self.lifetime_time_nidaq,
                d=self.raw_intensity_counts,
                time_units="s",
                time_support=support,
            ),
            "licks": nap.Ts(
                t=self.lick_times_nidaq,
                time_units="s",
                time_support=support,
            ),
            "visual_cue": nap.Ts(
                t=self.visual_cue_pulses.onset_times,
                time_units="s",
                time_support=support,
            ),
            "visual_cue_intervals": nap.IntervalSet(
                start=self.visual_cue_pulses.onset_times,
                end=self.visual_cue_pulses.offset_times,
                time_units="s",
            ),
            "ensure": nap.Ts(
                t=self.ensure_pulses.onset_times,
                time_units="s",
                time_support=support,
            ),
            "ensure_intervals": nap.IntervalSet(
                start=self.ensure_pulses.onset_times,
                end=self.ensure_pulses.offset_times,
                time_units="s",
            ),
        }
        if self.running_time_nidaq is not None and self.running_speed is not None:
            objects["running_speed"] = nap.Tsd(
                t=self.running_time_nidaq,
                d=self.running_speed,
                time_units="s",
            )
        return objects


def process_aligned_session(
    iflip_path: str | Path,
    nidaq_path: str | Path,
    background_path: str | Path,
    *,
    running_path: str | Path | None = None,
    spc_range: tuple[float, float] = (0.4, 12.3),
    afterpulse_ratio: float = 0.03,
    sync_threshold: float = 1.5,
    behavior_threshold: float = 1.5,
    iflip_sync_start_index: int = 0,
    nidaq_sync_start_index: int = 0,
) -> AlignedSession:
    """Process MPET, align clocks, and extract NI-DAQ behavior events."""

    iflip = read_iflip3(iflip_path)
    background = read_iflip3(background_path)
    nidaq: NIDAQRecording = read_nidaq(nidaq_path)

    mpet, _ = calculate_mpet(
        iflip,
        spc_range,
        t0=float(iflip.header.get_path("state.t0.Value")),
        measured_background=average_background([background]),
        afterpulse_ratio=afterpulse_ratio,
    )
    mpet_values = np.asarray(mpet[:, 0] if mpet.ndim == 2 else mpet, dtype=float)
    raw_intensity = np.asarray(iflip.data[:, :, 0].sum(axis=0), dtype=float)

    marker_mask = external_marker_mask(iflip.marks, marker=1)
    iflip_marker_times = iflip.sample_time[marker_mask]
    sync_pulses = find_ttl_pulses(
        nidaq.signal("sync"),
        nidaq.timestamps,
        threshold=sync_threshold,
    )
    alignment = fit_clock_alignment(
        iflip_marker_times,
        sync_pulses.onset_times,
        iflip_start_index=iflip_sync_start_index,
        nidaq_start_index=nidaq_sync_start_index,
    )

    lick_pulses = find_ttl_pulses(
        nidaq.signal("licking"),
        nidaq.timestamps,
        threshold=behavior_threshold,
    )
    visual_cue_pulses = find_ttl_pulses(
        nidaq.signal("visual_cue"),
        nidaq.timestamps,
        threshold=behavior_threshold,
    )
    ensure_pulses = find_ttl_pulses(
        nidaq.signal("ensure"),
        nidaq.timestamps,
        threshold=behavior_threshold,
    )

    running_time = None
    running_speed = None
    if running_path is not None:
        running: RunningData = read_running(running_path)
        if running.speed.size != sync_pulses.onset_times.size:
            raise ValueError("Running speed length does not match the number of NI-DAQ sync pulses")
        running_time = sync_pulses.onset_times.copy()
        running_speed = running.speed.copy()

    return AlignedSession(
        lifetime_time_nidaq=alignment.iflip_to_nidaq(iflip.sample_time),
        lifetime_time_iflip=iflip.sample_time.copy(),
        mpet_ns=mpet_values,
        raw_intensity_counts=raw_intensity,
        lick_times_nidaq=lick_pulses.onset_times,
        visual_cue_pulses=visual_cue_pulses,
        ensure_pulses=ensure_pulses,
        sync_pulses=sync_pulses,
        alignment=alignment,
        running_time_nidaq=running_time,
        running_speed=running_speed,
    )
