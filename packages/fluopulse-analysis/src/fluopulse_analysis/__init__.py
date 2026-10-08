"""Doric FluoPulse lifetime photometry analysis."""

from .events import (
    LickBouts,
    TTLPulses,
    extract_perievent,
    find_lick_bouts,
    find_ttl_pulses,
)
from .glm import (
    AdaptiveEnsureDesign,
    build_adaptive_ensure_design,
    fit_adaptive_ensure_glm,
    reconstruct_adaptive_kernels,
)
from .io import FluoPulseRecording, WaveformBatch, read_doric
from .nidaq import DEFAULT_CHANNEL_ROWS, NIDAQRecording, read_nidaq, read_running
from .paths import (
    DEFAULT_DORIC_ROOT,
    DEFAULT_PHOTOMETRY_ROOT,
    NIDAQPaths,
    SessionIdentity,
    find_doric_files,
    infer_session_identity,
    nidaq_paths,
    nidaq_paths_for_doric,
)
from .plotting import plot_aligned_session, plot_recording_qc, plot_waveform_fit
from .session import AlignedSession, process_aligned_session
from .synchronization import ClockAlignment, fit_clock_alignment
from .waveform import (
    WaveformDoubleFitResult,
    WaveformFitResult,
    fit_irf_convolved_double_exponential,
    fit_irf_convolved_single_exponential,
    moment_lifetime_ns,
    subtract_terminal_baseline,
)

__all__ = [
    "AlignedSession",
    "AdaptiveEnsureDesign",
    "ClockAlignment",
    "DEFAULT_CHANNEL_ROWS",
    "DEFAULT_DORIC_ROOT",
    "DEFAULT_PHOTOMETRY_ROOT",
    "FluoPulseRecording",
    "LickBouts",
    "NIDAQPaths",
    "NIDAQRecording",
    "SessionIdentity",
    "TTLPulses",
    "WaveformBatch",
    "WaveformDoubleFitResult",
    "WaveformFitResult",
    "extract_perievent",
    "build_adaptive_ensure_design",
    "find_doric_files",
    "find_lick_bouts",
    "find_ttl_pulses",
    "fit_clock_alignment",
    "fit_adaptive_ensure_glm",
    "fit_irf_convolved_double_exponential",
    "fit_irf_convolved_single_exponential",
    "infer_session_identity",
    "moment_lifetime_ns",
    "nidaq_paths",
    "nidaq_paths_for_doric",
    "plot_aligned_session",
    "plot_recording_qc",
    "plot_waveform_fit",
    "process_aligned_session",
    "read_doric",
    "read_nidaq",
    "read_running",
    "reconstruct_adaptive_kernels",
    "subtract_terminal_baseline",
]
