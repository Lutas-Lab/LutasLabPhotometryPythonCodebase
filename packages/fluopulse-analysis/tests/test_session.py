from pathlib import Path

import numpy as np
from fluopulse_analysis.events import TTLPulses
from fluopulse_analysis.io import FluoPulseRecording
from fluopulse_analysis.nidaq import NIDAQRecording, RunningData
from fluopulse_analysis.session import _resample_uniform, process_aligned_session


def _pulses(times, width=0.002):
    times = np.asarray(times, dtype=float)
    indices = np.arange(times.size)
    return TTLPulses(times, times + width, indices, indices)


def test_resample_uniform_interpolates_timestamp_jitter():
    time = np.array([0.0, 0.099, 0.201, 0.3])
    values = 2.0 * time
    regular_time, regular_values = _resample_uniform(time, values)
    assert np.allclose(np.diff(regular_time), 0.1)
    assert np.allclose(regular_values, 2.0 * regular_time)


def test_process_aligned_session(monkeypatch):
    doric_sync = np.arange(0.01, 2.01, 1 / 30)
    doric = FluoPulseRecording(
        path=Path("recording.doric"),
        time=np.arange(0.1, 2.0, 0.1),
        tau_ns=np.ones(19) * 2.5,
        amplitude=np.ones(19) * 80,
        chi_square=np.ones(19) * 0.01,
        r_square=np.ones(19) * 0.999,
        irf_time_ns=np.arange(4),
        irf_raw=np.ones(4),
        irf_values=np.ones(4),
        digital_channels={"sync": "DIO02", "licking": "DIO04"},
        digital_pulses={"sync": _pulses(doric_sync), "licking": _pulses([0.5])},
        waveform_points=4,
    )
    nidaq_sync = 0.2 + 1.00001 * doric_sync
    nidaq_time = np.arange(0, 2.5, 0.001)
    data = np.zeros((8, nidaq_time.size))
    for time in nidaq_sync:
        index = np.searchsorted(nidaq_time, time)
        if 0 <= index < nidaq_time.size - 2:
            data[1, index : index + 2] = 3
    data[3, 1000:1010] = 3
    data[7, 1500:1550] = 3
    nidaq = NIDAQRecording(
        Path("nidaq.mat"),
        data,
        nidaq_time,
        1000.0,
        {
            "photoreceiver_1": 0,
            "sync": 1,
            "photoreceiver_2": 2,
            "licking": 3,
            "visual_cue": 4,
            "ttl_465": 5,
            "ttl_405": 6,
            "ensure": 7,
        },
    )
    monkeypatch.setattr("fluopulse_analysis.session.read_doric", lambda *a, **k: doric)
    monkeypatch.setattr("fluopulse_analysis.session.read_nidaq", lambda *a, **k: nidaq)
    monkeypatch.setattr(
        "fluopulse_analysis.session.read_running",
        lambda *a, **k: RunningData(np.ones(doric_sync.size), None, None),
    )

    session = process_aligned_session(
        "recording.doric",
        "nidaq.mat",
        running_path="running.mat",
    )
    assert session.alignment.matched_pulses == doric_sync.size
    assert session.doric_licks.onset_times.size == 1
    assert session.nidaq_licks.onset_times.size == 1
    assert session.ensure_pulses.onset_times.size == 1
    assert session.nidaq_coverage_fraction == 1.0
    assert np.all(session.nidaq_coverage_mask)
    common = session.to_core_session()
    assert common.session_id == "recording"
    assert set(common.continuous) == {
        "tau",
        "amplitude",
        "fit_r_square",
        "running_speed",
    }
    assert set(common.events) == {"licks", "nidaq_licks", "ensure", "visual_cue"}
    assert common.metadata["sensor"] == "fluopulse"


def test_process_session_without_nidaq_uses_doric_clock_and_events(monkeypatch):
    doric = FluoPulseRecording(
        path=Path("baseline.doric"),
        time=np.array([0.1, 0.2, 0.3]),
        tau_ns=np.array([2.5, 2.6, 2.7]),
        amplitude=np.array([80.0, 81.0, 82.0]),
        chi_square=np.array([0.01, 0.01, 0.01]),
        r_square=np.array([0.99, 0.99, 0.99]),
        irf_time_ns=np.arange(4),
        irf_raw=np.ones(4),
        irf_values=np.ones(4),
        digital_channels={"licking": "DIO04", "ensure": "DIO05"},
        digital_pulses={"licking": _pulses([0.2]), "ensure": _pulses([0.25])},
        waveform_points=4,
    )
    monkeypatch.setattr("fluopulse_analysis.session.read_doric", lambda *a, **k: doric)

    session = process_aligned_session("baseline.doric", None)
    common = session.to_core_session()

    assert session.nidaq is None
    assert session.alignment is None
    assert session.nidaq_coverage_fraction is None
    np.testing.assert_allclose(common.continuous["tau"].timestamps, doric.time)
    np.testing.assert_allclose(common.events["licks"].timestamps, [0.2])
    np.testing.assert_allclose(common.events["ensure"].timestamps, [0.25])
    assert common.metadata["timebase"] == "doric_native"
    assert common.metadata["nidaq_aligned"] is False
