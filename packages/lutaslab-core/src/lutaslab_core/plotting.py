"""Reusable plots for peri-event trial matrices."""

from __future__ import annotations

import numpy as np

from .perievent import normalize_trials, summarize_trials


def plot_perievent_mean(
    peri_time,
    trials,
    *,
    ax=None,
    ylabel="Signal",
    title=None,
    event_label="Event",
    normalization=None,
    baseline=None,
):
    """Plot mean and SEM for a trial-by-time peri-event matrix."""

    import matplotlib.pyplot as plt

    peri_time = np.asarray(peri_time, dtype=float)
    normalized = normalize_trials(peri_time, trials, normalization, baseline)
    if normalized.shape[0] == 0:
        raise ValueError("Cannot plot a peri-event mean with no valid trials")
    mean, sem = summarize_trials(normalized)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(peri_time, mean)
    ax.fill_between(peri_time, mean - sem, mean + sem, alpha=0.3)
    ax.axvline(0, linestyle="--")
    ax.set_xlabel(f"Time from {event_label.lower()} (s)")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def plot_perievent_heatmap(
    peri_time,
    trials,
    *,
    ax=None,
    title=None,
    ylabel="Trial",
    colorbar_label="Signal",
    normalization=None,
    baseline=None,
    cmap="bwr",
    center_zero=True,
    interpolation="nearest",
):
    """Plot normalized peri-event trials as a heatmap."""

    import matplotlib.pyplot as plt

    peri_time = np.asarray(peri_time, dtype=float)
    normalized = normalize_trials(peri_time, trials, normalization, baseline)
    if normalized.shape[0] == 0:
        raise ValueError("Cannot plot a peri-event heatmap with no valid trials")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    limits = {}
    if center_zero:
        maximum = float(np.nanmax(np.abs(normalized)))
        if np.isfinite(maximum) and maximum > 0:
            limits = {"vmin": -maximum, "vmax": maximum}
    image = ax.imshow(
        normalized,
        aspect="auto",
        extent=(peri_time[0], peri_time[-1], normalized.shape[0], 0),
        cmap=cmap,
        interpolation=interpolation,
        **limits,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Time from event (s)", ylabel=ylabel)
    if title is not None:
        ax.set_title(title)
    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    return ax
