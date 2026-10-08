from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from fluopulse_analysis.events import TTLPulses
from fluopulse_analysis.io import FluoPulseRecording
from fluopulse_analysis.plotting import (
    plot_aligned_session,
    plot_recording_qc,
    plot_waveform_fit,
)
from fluopulse_analysis.waveform import WaveformFitResult


def _pulses(onsets):
    onsets = np.asarray(onsets, dtype=float)
    return TTLPulses(
        onsets,
        onsets + 0.1,
        np.arange(onsets.size),
        np.arange(onsets.size) + 1,
    )


def _recording():
    time = np.linspace(0.0, 2.0, 21)
    return FluoPulseRecording(
        path=Path("synthetic.doric"),
        time=time,
        tau_ns=2.5 + 0.05 * np.sin(time),
        amplitude=100.0 + np.cos(time),
        chi_square=np.full(time.size, 0.02),
        r_square=np.full(time.size, 0.99),
        irf_time_ns=np.linspace(0.0, 10.0, 8),
        irf_raw=np.ones(8),
        irf_values=np.ones(8),
        digital_channels={},
        digital_pulses={},
        waveform_points=8,
    )


def _assert_figure_saves(figure):
    stream = BytesIO()
    figure.savefig(stream, format="png")
    assert stream.tell() > 0
    plt.close(figure)


def test_plot_recording_qc_renders_four_panels():
    figure = plot_recording_qc(_recording())
    assert len(figure.axes) == 4
    assert figure.axes[-1].get_xlabel() == "Doric time (s)"
    _assert_figure_saves(figure)


def test_plot_aligned_session_renders_events_and_running():
    recording = _recording()
    session = SimpleNamespace(
        doric=recording,
        lifetime_time_nidaq=recording.time + 0.01,
        doric_licks=_pulses([0.5, 1.5]),
        ensure_pulses=_pulses([1.0]),
        running_time_nidaq=recording.time,
        running_speed=np.linspace(0.0, 3.0, recording.time.size),
    )

    figure = plot_aligned_session(session)
    assert len(figure.axes) == 3
    assert figure.axes[-1].lines
    assert all(axis.patches for axis in figure.axes)
    _assert_figure_saves(figure)


def test_plot_waveform_fit_renders_fit_and_residuals():
    time = np.linspace(0.0, 8.0, 40)
    waveform = np.exp(-time / 2.0)
    fitted = 0.98 * waveform
    result = WaveformFitResult(
        tau_ns=2.0,
        amplitude=1.0,
        offset=0.0,
        shift_ns=0.0,
        fitted=fitted,
        residuals=fitted - waveform,
        r_square=0.99,
        success=True,
    )

    figure = plot_waveform_fit(time, waveform, result)
    assert len(figure.axes) == 2
    assert len(figure.axes[0].lines) == 2
    assert len(figure.axes[1].lines) == 2
    _assert_figure_saves(figure)
