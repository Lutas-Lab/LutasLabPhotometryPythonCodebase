from pathlib import Path

import numpy as np

from lutaslab_core.nidaq import nidaq_from_mapping, running_from_mapping


def test_nidaq_from_mapping_uses_explicit_hardware_rows():
    timestamps = np.arange(10, dtype=float) / 10
    data = np.zeros((8, 10), dtype=float)
    data[1, 2:4] = 3.0
    recording = nidaq_from_mapping(
        "session-nidaq.mat",
        {"data": data, "timestamps": timestamps, "Fs": 10.0},
    )
    assert recording.path == Path("session-nidaq.mat")
    np.testing.assert_array_equal(recording.signal("sync"), data[1])


def test_running_from_mapping_preserves_optional_fields():
    running = running_from_mapping(
        "running.mat",
        {
            "speed": [1.0, 2.0],
            "position": [3.0, 4.0],
            "timestamps": [0.0, 0.1],
        },
    )
    np.testing.assert_allclose(running.speed, [1.0, 2.0])
    np.testing.assert_allclose(running.position, [3.0, 4.0])
    np.testing.assert_allclose(running.timestamps, [0.0, 0.1])
