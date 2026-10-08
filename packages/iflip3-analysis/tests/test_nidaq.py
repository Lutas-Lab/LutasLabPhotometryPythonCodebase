import numpy as np
from iflip3.nidaq import find_ttl_pulses, read_nidaq, read_running


def test_read_nidaq_uses_explicit_hardware_rows(monkeypatch):
    timestamps = np.arange(10, dtype=float) / 10
    data = np.zeros((8, 10), dtype=float)
    data[1, 2:4] = 3.0
    data[3, 5:7] = 4.0
    monkeypatch.setattr(
        "iflip3.nidaq.loadmat",
        lambda *args, **kwargs: {
            "data": data,
            "timestamps": timestamps,
            "Fs": 10.0,
            "channelnames": np.array(["incorrect metadata"], dtype=object),
        },
    )
    recording = read_nidaq("session-nidaq.mat")
    np.testing.assert_array_equal(recording.signal("sync"), data[1])
    np.testing.assert_array_equal(recording.signal("licking"), data[3])


def test_find_ttl_pulses_returns_complete_filtered_pulses():
    timestamps = np.arange(12, dtype=float) / 10
    signal = np.array([0, 3, 3, 0, 0, 3, 0, 0, 3, 3, 3, 0], dtype=float)
    pulses = find_ttl_pulses(
        signal,
        timestamps,
        threshold=1.5,
        min_width_seconds=0.15,
        max_width_seconds=0.25,
    )
    np.testing.assert_array_equal(pulses.rising_indices, [1])
    np.testing.assert_array_equal(pulses.falling_indices, [3])
    np.testing.assert_allclose(pulses.durations, [0.2])


def test_read_running_requires_and_loads_speed(monkeypatch):
    monkeypatch.setattr(
        "iflip3.nidaq.loadmat",
        lambda *args, **kwargs: {
            "speed": [[1.0, 2.0, 3.0]],
            "position": [[4.0, 5.0, 6.0]],
        },
    )
    running = read_running("running.mat")
    np.testing.assert_allclose(running.speed, [1, 2, 3])
    np.testing.assert_allclose(running.position, [4, 5, 6])
