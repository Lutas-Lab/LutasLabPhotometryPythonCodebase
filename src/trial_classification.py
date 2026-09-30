"""Configurable cue-trial classifications derived from saved event timestamps."""

from __future__ import annotations

import numpy as np


TRIAL_CLASS_KEYS = (
    "all",
    "cue_lick",
    "post_cue_lick",
    "cue_only",
    "post_only",
    "cue_and_post",
    "cue_miss",
)


def classify_cue_licking(cue_onset, cue_offset, lick_times, post_cue_window=2.0):
    """Classify cue trials from timestamps and a configurable post-cue window."""
    cue_onset = np.atleast_1d(np.asarray(cue_onset, dtype=float).squeeze())
    cue_offset = np.atleast_1d(np.asarray(cue_offset, dtype=float).squeeze())
    lick_times = np.atleast_1d(np.asarray(lick_times, dtype=float).squeeze())
    if cue_onset.ndim != 1 or cue_offset.ndim != 1 or lick_times.ndim != 1:
        raise ValueError("Cue and lick timestamps must be one-dimensional.")
    if len(cue_onset) != len(cue_offset):
        raise ValueError("cue_onset and cue_offset must have the same length.")
    if not np.isfinite(post_cue_window) or post_cue_window < 0:
        raise ValueError("post_cue_window must be finite and nonnegative.")
    if not (
        np.all(np.isfinite(cue_onset))
        and np.all(np.isfinite(cue_offset))
        and np.all(np.isfinite(lick_times))
    ):
        raise ValueError("Cue and lick timestamps must be finite.")
    if np.any(cue_offset < cue_onset):
        raise ValueError("Every cue offset must be at or after its cue onset.")

    cue_lick = np.zeros(len(cue_onset), dtype=bool)
    post_cue_lick = np.zeros(len(cue_onset), dtype=bool)
    for index, (onset, offset) in enumerate(zip(cue_onset, cue_offset)):
        cue_lick[index] = np.any((lick_times >= onset) & (lick_times <= offset))
        post_cue_lick[index] = np.any(
            (lick_times > offset) & (lick_times <= offset + post_cue_window)
        )

    cue_only = cue_lick & ~post_cue_lick
    post_only = ~cue_lick & post_cue_lick
    cue_and_post = cue_lick & post_cue_lick
    cue_miss = ~cue_lick & ~post_cue_lick
    return cue_lick, post_cue_lick, cue_only, post_only, cue_and_post, cue_miss


def classify_session_cue_licking(session, post_cue_window=2.0):
    """Return named masks recalculated from a processed session's timestamps."""
    required = {"cue_onset", "cue_offset", "lick_times"}
    missing = required.difference(session)
    if missing:
        raise ValueError(
            "Dynamic cue-trial classification requires saved timestamps: "
            f"{sorted(missing)} are missing."
        )
    values = classify_cue_licking(
        session["cue_onset"],
        session["cue_offset"],
        session["lick_times"],
        post_cue_window=post_cue_window,
    )
    keys = TRIAL_CLASS_KEYS[1:]
    masks = dict(zip(keys, values))
    masks["all"] = np.ones(
        len(np.atleast_1d(np.asarray(session["cue_onset"]))), dtype=bool
    )
    return masks


def cue_trial_mask(session, trial_class="all", post_cue_window=2.0):
    """Return one analysis-time cue-trial mask."""
    if trial_class not in TRIAL_CLASS_KEYS:
        raise ValueError(f"trial_class must be one of {TRIAL_CLASS_KEYS}.")
    return classify_session_cue_licking(session, post_cue_window)[trial_class]
