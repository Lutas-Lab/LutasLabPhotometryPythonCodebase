"""Tools for fluorescence-lifetime photometry recorded by iFLiP3.

Numerical fitting symbols are imported lazily so file reading and preprocessing
remain usable in lightweight environments that have NumPy but not SciPy.
"""

from typing import Any

from .io import IFLiP3Header, IFLiP3Recording, read_iflip3, read_iflip3_bytes
from .nidaq import (
    DEFAULT_CHANNEL_ROWS,
    NIDAQRecording,
    RunningData,
    TTLPulses,
    find_ttl_pulses,
    read_nidaq,
    read_running,
)
from .paths import DEFAULT_DATA_ROOT, SessionPaths, session_paths
from .preprocessing import (
    CorrectionResult,
    average_background,
    bin_curves,
    calculate_mpet,
    correct_lifetime_data,
    mpet_from_corrected,
)
from .session import AlignedSession, process_aligned_session
from .synchronization import ClockAlignment, external_marker_mask, fit_clock_alignment

__all__ = [
    "CorrectionResult",
    "ClockAlignment",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_CHANNEL_ROWS",
    "GlobalFitResult",
    "AlignedSession",
    "IFLiP3Header",
    "IFLiP3Recording",
    "LifetimeFitResult",
    "NIDAQRecording",
    "RunningData",
    "SessionPaths",
    "TTLPulses",
    "TargetAnalysisResult",
    "average_background",
    "bin_curves",
    "calculate_mpet",
    "correct_lifetime_data",
    "fit_decay",
    "fit_clock_alignment",
    "fit_global",
    "fit_target",
    "find_ttl_pulses",
    "external_marker_mask",
    "lifetime_window",
    "mpet_from_corrected",
    "periodic_exgaussian_basis",
    "process_aligned_session",
    "read_iflip3",
    "read_iflip3_bytes",
    "read_nidaq",
    "read_running",
    "session_paths",
]


def __getattr__(name: str) -> Any:
    if name in {"GlobalFitResult", "LifetimeFitResult", "fit_decay", "fit_global"}:
        from . import fitting

        return getattr(fitting, name)
    if name == "periodic_exgaussian_basis":
        from .models import periodic_exgaussian_basis

        return periodic_exgaussian_basis
    if name in {"TargetAnalysisResult", "fit_target", "lifetime_window"}:
        from . import target

        return getattr(target, name)
    raise AttributeError(name)
