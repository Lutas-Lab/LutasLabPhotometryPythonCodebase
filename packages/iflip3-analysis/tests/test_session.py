import sys
import types
from pathlib import Path

import numpy as np
import pytest
from iflip3.nidaq import RunningData, TTLPulses
from iflip3.session import AlignedSession, process_aligned_session
from iflip3.synchronization import ClockAlignment
from lutaslab_core.nidaq import RunningDataExcludedWarning


class _NapObject:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _pulses(onsets, offsets):
    onsets = np.asarray(onsets, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    return TTLPulses(
        rising_indices=np.arange(onsets.size),
        falling_indices=np.arange(offsets.size),
        onset_times=onsets,
        offset_times=offsets,
    )


def test_aligned_session_builds_pynapple_objects(monkeypatch):
    fake_pynapple = types.SimpleNamespace(
        IntervalSet=_NapObject,
        Tsd=_NapObject,
        Ts=_NapObject,
    )
    monkeypatch.setitem(sys.modules, "pynapple", fake_pynapple)
    alignment = ClockAlignment(
        intercept_seconds=0.2,
        scale=1.0,
        residuals_seconds=np.zeros(3),
        matched_pulses=3,
        iflip_marker_times=np.array([0.0, 0.2, 0.4]),
        nidaq_pulse_times=np.array([0.2, 0.4, 0.6]),
    )
    session = AlignedSession(
        lifetime_time_nidaq=np.array([0.2, 0.3, 0.4]),
        lifetime_time_iflip=np.array([0.0, 0.1, 0.2]),
        mpet_ns=np.array([1.1, 1.2, 1.3]),
        raw_intensity_counts=np.array([100.0, 101.0, 102.0]),
        lick_times_nidaq=np.array([0.25]),
        visual_cue_pulses=_pulses([0.3], [0.31]),
        ensure_pulses=_pulses([0.35], [0.37]),
        sync_pulses=_pulses([0.2, 0.4], [0.21, 0.41]),
        alignment=alignment,
        running_time_nidaq=np.array([0.2, 0.4]),
        running_speed=np.array([2.0, 3.0]),
    )

    objects = session.to_pynapple()

    assert set(objects) == {
        "mpet",
        "raw_intensity",
        "licks",
        "visual_cue",
        "visual_cue_intervals",
        "ensure",
        "ensure_intervals",
        "running_speed",
    }
    np.testing.assert_allclose(objects["mpet"].kwargs["d"], session.mpet_ns)
    np.testing.assert_allclose(objects["ensure"].kwargs["t"], session.ensure_pulses.onset_times)
    np.testing.assert_allclose(objects["running_speed"].kwargs["d"], session.running_speed)
    common = session.to_core_session("mouse-date-run")
    assert common.session_id == "mouse-date-run"
    assert set(common.continuous) == {"mpet", "raw_intensity", "running_speed"}
    assert set(common.events) == {"licks", "visual_cue", "ensure", "sync"}
    assert common.metadata["sensor"] == "iflip3"


def test_process_session_without_nidaq_uses_iflip_clock(monkeypatch):
    recording = types.SimpleNamespace(
        sample_time=np.array([0.1, 0.2, 0.3]),
        marks=np.array([0, 4, 0], dtype=np.uint32),
        data=np.ones((2, 3, 1), dtype=float),
        header=types.SimpleNamespace(get_path=lambda path: 1.0),
    )
    monkeypatch.setattr("iflip3.session.read_iflip3", lambda *a, **k: recording)
    monkeypatch.setattr(
        "iflip3.session.average_background",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("background loaded")),
    )
    observed = {}

    def fake_calculate_mpet(*args, **kwargs):
        observed["measured_background"] = kwargs["measured_background"]
        return np.array([1.1, 1.2, 1.3]), None

    monkeypatch.setattr(
        "iflip3.session.calculate_mpet",
        fake_calculate_mpet,
    )

    session = process_aligned_session("recording.iFLiP3", None, None)
    common = session.to_core_session()

    assert session.alignment is None
    np.testing.assert_allclose(common.continuous["mpet"].timestamps, recording.sample_time)
    assert common.events["licks"].timestamps.size == 0
    assert common.events["ensure"].timestamps.size == 0
    np.testing.assert_allclose(common.events["sync"].timestamps, [0.2])
    assert common.metadata["timebase"] == "iflip_native"
    assert common.metadata["nidaq_aligned"] is False
    assert common.metadata["measured_background_applied"] is False
    assert observed["measured_background"] is None


def test_flat_zero_running_is_flagged_and_omitted(monkeypatch):
    sample_time = np.array([0.0, 0.2, 0.4])
    recording = types.SimpleNamespace(
        sample_time=sample_time,
        marks=np.array([4, 4, 4], dtype=np.uint32),
        data=np.ones((2, 3, 1), dtype=float),
        header=types.SimpleNamespace(get_path=lambda path: 1.0),
    )
    nidaq = types.SimpleNamespace(
        timestamps=np.linspace(0.0, 1.0, 1001),
        signal=lambda name: name,
    )
    pulse_map = {
        "sync": _pulses([0.1, 0.3, 0.5], [0.11, 0.31, 0.51]),
        "licking": TTLPulses.empty(),
        "visual_cue": TTLPulses.empty(),
        "ensure": TTLPulses.empty(),
    }
    alignment = ClockAlignment(
        intercept_seconds=0.1,
        scale=1.0,
        residuals_seconds=np.zeros(3),
        matched_pulses=3,
        iflip_marker_times=sample_time,
        nidaq_pulse_times=np.array([0.1, 0.3, 0.5]),
    )
    monkeypatch.setattr("iflip3.session.read_iflip3", lambda *a, **k: recording)
    monkeypatch.setattr(
        "iflip3.session.calculate_mpet",
        lambda *a, **k: (np.array([1.1, 1.2, 1.3]), None),
    )
    monkeypatch.setattr("iflip3.session.read_nidaq", lambda *a, **k: nidaq)
    monkeypatch.setattr(
        "iflip3.session.find_ttl_pulses", lambda signal, *a, **k: pulse_map[signal]
    )
    monkeypatch.setattr("iflip3.session.fit_clock_alignment", lambda *a, **k: alignment)
    monkeypatch.setattr(
        "iflip3.session.read_running",
        lambda *a, **k: RunningData(Path("running.mat"), np.zeros(3), None),
    )

    with pytest.warns(RunningDataExcludedWarning, match="flat line at zero"):
        session = process_aligned_session(
            "recording.iFLiP3", "nidaq.mat", running_path="running.mat"
        )

    assert session.running_speed is None
    assert session.running_time_nidaq is None
    common = session.to_core_session()
    assert "running_speed" not in common.continuous
    assert common.metadata["running_data_valid"] is False
    assert common.metadata["running_data_status"] == "invalid_flat_zero"
