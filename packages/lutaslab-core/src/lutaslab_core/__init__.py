"""Shared time-series infrastructure for Lutas Lab analysis packages."""

from .batch import run_batch
from .events import (
    LickBouts,
    SampledBoutFeatures,
    TTLPulses,
    find_lick_bouts,
    find_ttl_pulses,
    sample_lick_bout_features,
)
from .glm import (
    RidgeCVResult,
    TemporalBasis,
    TrialwiseDesign,
    apply_lag_basis,
    blocked_folds,
    build_trialwise_basis_design,
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
from .manifest import load_session_manifest
from .nidaq import (
    DEFAULT_CHANNEL_ROWS,
    NIDAQRecording,
    RunningData,
    nidaq_from_mapping,
    read_nidaq,
    read_running,
    running_from_mapping,
)
from .perievent import (
    extract_perievent_event_rate,
    extract_perievent_trials,
    generate_null_onsets,
    normalize_trials,
    summarize_trials,
)
from .provenance import build_run_record, describe_path, utc_now, write_run_record
from .publication import BundleValidation, PublicationBundle, validate_publication_bundle
from .session import AlignedSession, ContinuousSignal, EventSeries, IntervalSeries
from .synchronization import ClockAlignment, fit_clock_alignment
from .transfer import (
    GammaTransferFit,
    apply_gamma_transfer,
    fit_nonnegative_gamma_transfer,
    gamma_cascade_kernel,
)

__all__ = [
    "AlignedSession",
    "ClockAlignment",
    "BundleValidation",
    "ContinuousSignal",
    "DEFAULT_CHANNEL_ROWS",
    "EventSeries",
    "GammaTransferFit",
    "IntervalSeries",
    "LickBouts",
    "SampledBoutFeatures",
    "NIDAQRecording",
    "PublicationBundle",
    "RunningData",
    "RidgeCVResult",
    "TTLPulses",
    "TemporalBasis",
    "TrialwiseDesign",
    "apply_gamma_transfer",
    "apply_lag_basis",
    "blocked_folds",
    "build_trialwise_basis_design",
    "build_run_record",
    "convolve_basis",
    "describe_path",
    "event_times_to_counts",
    "find_lick_bouts",
    "find_ttl_pulses",
    "fit_clock_alignment",
    "fit_grouped_ridge_cv",
    "fit_nonnegative_gamma_transfer",
    "fit_ridge",
    "group_folds",
    "lagged_basis_matrix",
    "lagged_signal_matrix",
    "extract_perievent_event_rate",
    "extract_perievent_trials",
    "generate_null_onsets",
    "gamma_cascade_kernel",
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
    "sample_lick_bout_features",
    "summarize_trials",
    "utc_now",
    "validate_publication_bundle",
    "write_run_record",
]

__version__ = "0.1.0"
