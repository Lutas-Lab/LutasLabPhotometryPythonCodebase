import io

import h5py
import numpy as np

from fluopulse_analysis.io import read_doric


def _synthetic_doric_bytes() -> bytes:
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as file:
        base = "/DataAcquisition/FluoPulse/Signals/Series0001"
        calculation = file.create_group(f"{base}/Calculation01")
        time = np.array([0.1, 0.2, 0.3])
        for name, values in {
            "Time": time,
            "Tau01": [2.4, 2.5, 2.6],
            "Amplitude": [80, 81, 82],
            "Chisquare": [0.1, 0.1, 0.1],
            "Rsquare": [0.99, 0.98, 0.97],
        }.items():
            calculation.create_dataset(name, data=values)
        fluorescence = file.create_group(f"{base}/Fluorescence01")
        relative = np.arange(4) * 0.1e-9
        fluorescence.create_dataset(
            "Time", data=np.concatenate([sample + relative for sample in time])
        )
        fluorescence.create_dataset("Values", data=np.arange(12, dtype=float))
        irf = file.create_group(f"{base}/IRF01")
        irf.create_dataset("Time", data=relative)
        irf.create_dataset("Raw", data=[0, 1, 0.5, 0])
        irf.create_dataset("Values", data=[0, 1, 0.5, 0])
        digital = file.create_group(f"{base}/DigitalIO")
        digital.create_dataset("Time", data=np.arange(20) / 1000)
        digital.create_dataset("DIO02", data=np.tile([0, 1], 10))
        digital.create_dataset(
            "DIO04", data=np.r_[np.zeros(5), np.ones(2), np.zeros(13)]
        )
        for name, username in {"DIO02": "CAMpulse", "DIO04": "lick_event"}.items():
            settings = file.create_group(f"/Configurations/FluoPulse/{name}/Settings")
            settings.attrs["Username"] = username
    return buffer.getvalue()


def _synthetic_legacy_doric_bytes() -> bytes:
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as file:
        base = "/DataAcquisition/FluoPulse/Signals/Series0001"
        calculation = file.create_group(f"{base}/Calculation")
        time = np.array([0.0, 0.1, 0.2])
        for name, values in {
            "Time": time,
            "Tau01": [2.8, 2.9, 3.0],
            "Amplitude01": [70, 71, 72],
            "Chisquare": [0.2, 0.2, 0.2],
            "Rsquare": [0.96, 0.97, 0.98],
        }.items():
            calculation.create_dataset(name, data=values)
        analog = file.create_group(f"{base}/AnalogIn")
        relative = np.arange(4) * 0.1e-9
        analog.create_dataset(
            "Time", data=np.concatenate([sample + relative for sample in time])
        )
        analog.create_dataset("Detector01", data=np.arange(12, dtype=float))
        irf = file.create_group("/Configurations/FluoPulse/IRF")
        irf.create_dataset("Time", data=relative)
        irf.create_dataset("Values", data=[0, 1, 0.5, 0])
        digital = file.create_group(f"{base}/DigitalIO")
        digital.create_dataset("Time", data=np.arange(20) / 1000)
        digital.create_dataset(
            "DIO04", data=np.r_[np.zeros(5), np.ones(2), np.zeros(13)]
        )
        settings = file.create_group("/Configurations/FluoPulse/DIO04/Settings")
        settings.attrs["Username"] = "Lick"
    return buffer.getvalue()


def test_read_doric_and_lazy_waveform_loading(monkeypatch):
    payload = _synthetic_doric_bytes()
    real_file = h5py.File

    def open_memory(*args, **kwargs):
        return real_file(io.BytesIO(payload), "r")

    monkeypatch.setattr("fluopulse_analysis.io.h5py.File", open_memory)
    recording = read_doric("synthetic.doric")
    np.testing.assert_allclose(recording.tau_ns, [2.4, 2.5, 2.6])
    assert recording.waveform_points == 4
    assert recording.digital_channels == {"sync": "DIO02", "licking": "DIO04"}
    assert recording.digital_pulses["licking"].onset_times.size == 1
    batch = recording.load_waveforms([1])
    np.testing.assert_allclose(batch.values[0], [4, 5, 6, 7])
    np.testing.assert_allclose(batch.waveform_time_ns, [0, 0.1, 0.2, 0.3])


def test_read_legacy_doric_layout(monkeypatch):
    payload = _synthetic_legacy_doric_bytes()
    real_file = h5py.File

    def open_memory(*args, **kwargs):
        return real_file(io.BytesIO(payload), "r")

    monkeypatch.setattr("fluopulse_analysis.io.h5py.File", open_memory)
    recording = read_doric("legacy.doric")
    np.testing.assert_allclose(recording.tau_ns, [2.8, 2.9, 3.0])
    np.testing.assert_allclose(recording.amplitude, [70, 71, 72])
    np.testing.assert_allclose(recording.irf_raw, recording.irf_values)
    assert recording.waveform_points == 4
    assert recording.digital_pulses["licking"].onset_times.size == 1
    np.testing.assert_allclose(recording.load_waveforms([2]).values[0], [8, 9, 10, 11])
