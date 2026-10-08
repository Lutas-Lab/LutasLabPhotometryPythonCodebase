"""Standard quality-control plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .io import FluoPulseRecording
from .session import AlignedSession
from .waveform import WaveformFitResult


def plot_recording_qc(recording: FluoPulseRecording):
    figure, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    axes[0].plot(recording.time, recording.tau_ns, color="tab:blue")
    axes[0].set_ylabel("Vendor tau (ns)")
    axes[1].plot(recording.time, recording.amplitude, color="0.35")
    axes[1].set_ylabel("Amplitude")
    axes[2].plot(recording.time, recording.r_square, color="tab:green")
    axes[2].set_ylabel("Fit R-squared")
    axes[3].plot(recording.time, recording.chi_square, color="tab:red")
    axes[3].set(xlabel="Doric time (s)", ylabel="Chi-square")
    return figure


def plot_aligned_session(session: AlignedSession):
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(session.lifetime_time_nidaq, session.doric.tau_ns, color="tab:blue")
    axes[0].set_ylabel("Vendor tau (ns)")
    axes[1].eventplot(session.doric_licks.onset_times, colors="black")
    axes[1].set_ylabel("Doric licks")
    if session.running_time_nidaq is not None:
        axes[2].plot(session.running_time_nidaq, session.running_speed, color="tab:green")
    axes[2].set(xlabel="NI-DAQ time (s)", ylabel="Running speed")
    for onset, offset in zip(
        session.ensure_pulses.onset_times,
        session.ensure_pulses.offset_times,
        strict=True,
    ):
        for axis in axes:
            axis.axvspan(onset, offset, color="tab:orange", alpha=0.2)
    return figure


def plot_waveform_fit(
    time_ns: np.ndarray,
    waveform: np.ndarray,
    result: WaveformFitResult,
):
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time_ns, waveform, ".", ms=2, label="Raw waveform")
    axes[0].plot(time_ns, result.fitted, color="tab:red", label="Independent fit")
    axes[0].legend()
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(time_ns, result.residuals, color="tab:blue")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(xlabel="Waveform time (ns)", ylabel="Residual")
    return figure
