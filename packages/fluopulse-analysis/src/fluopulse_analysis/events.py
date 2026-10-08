"""Event-analysis compatibility API backed by :mod:`lutaslab_core`."""

from __future__ import annotations

import numpy as np
from lutaslab_core.events import (
    LickBouts,
    TTLPulses,
    find_lick_bouts,
    find_ttl_pulses,
)
from lutaslab_core.perievent import extract_perievent_trials


def extract_perievent(
    timestamps: np.ndarray,
    values: np.ndarray,
    event_times: np.ndarray,
    *,
    window_seconds: tuple[float, float] = (-10.0, 30.0),
    sample_interval_seconds: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a continuous trace onto one common grid around each event."""
    relative_time, trials, _ = extract_perievent_trials(
        timestamps,
        values,
        event_times,
        window_seconds,
        sample_interval_seconds,
        require_complete=False,
    )
    return relative_time, trials


__all__ = [
    "LickBouts",
    "TTLPulses",
    "extract_perievent",
    "find_lick_bouts",
    "find_ttl_pulses",
]
