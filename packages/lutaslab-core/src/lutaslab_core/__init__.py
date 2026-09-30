"""Shared time-series infrastructure for Lutas Lab analysis packages."""

from .events import LickBouts, TTLPulses, find_lick_bouts, find_ttl_pulses
from .batch import run_batch
from .manifest import load_session_manifest
from .glm import (
    RidgeCVResult,
    TemporalBasis,
    apply_lag_basis,
    blocked_folds,
    convolve_basis,
    event_times_to_counts,
    fit_grouped_ridge_cv,
    fit_ridge,
    group_folds,
    lagged_basis_matrix,
    lagged_signal_matrix,
    predict_lagged_signal,
    raised_cosine_basis,
    reconstruct_kernel,
)
from .nidaq import (
    DEFAULT_CHANNEL_ROWS,
    NIDAQRecording,
    RunningData,
    nidaq_from_mapping,
    read_nidaq,
    read_running,
    running_from_mapping,
)
from .session import AlignedSession, ContinuousSignal, EventSeries, IntervalSeries
from .perievent import (
    extract_perievent_event_rate,
    extract_perievent_trials,
    generate_null_onsets,
    normalize_trials,
    summarize_trials,
)
from .provenance import build_run_record, describe_path, utc_now, write_run_record
from .synchronization import ClockAlignment, fit_clock_alignment

__all__ = [
    "AlignedSession",
    "ClockAlignment",
    "ContinuousSignal",
    "DEFAULT_CHANNEL_ROWS",
    "EventSeries",
    "IntervalSeries",
    "LickBouts",
    "NIDAQRecording",
    "RunningData",
    "RidgeCVResult",
    "TTLPulses",
    "TemporalBasis",
    "apply_lag_basis",
    "blocked_folds",
    "build_run_record",
    "convolve_basis",
    "describe_path",
    "event_times_to_counts",
    "find_lick_bouts",
    "find_ttl_pulses",
    "fit_clock_alignment",
    "fit_grouped_ridge_cv",
    "fit_ridge",
    "group_folds",
    "lagged_basis_matrix",
    "lagged_signal_matrix",
    "extract_perievent_event_rate",
    "extract_perievent_trials",
    "generate_null_onsets",
    "load_session_manifest",
    "nidaq_from_mapping",
    "normalize_trials",
    "predict_lagged_signal",
    "read_nidaq",
    "read_running",
    "raised_cosine_basis",
    "reconstruct_kernel",
    "running_from_mapping",
    "run_batch",
    "summarize_trials",
    "utc_now",
    "write_run_record",
]

__version__ = "0.1.0"
