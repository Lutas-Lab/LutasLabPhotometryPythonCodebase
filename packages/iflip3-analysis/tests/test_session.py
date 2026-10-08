import sys
import types

import numpy as np
from iflip3.nidaq import TTLPulses
from iflip3.session import AlignedSession
from iflip3.synchronization import ClockAlignment


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
