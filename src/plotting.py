import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt


# ============================================================
# Peri-event continuous signal extraction
# ============================================================

def get_perievent_trials(
    signal,
    event_times,
    window=(-5, 10),
    dt=0.02
):
    """
    Extract a continuous signal around a set of event times.

    Events that do not have the full requested window inside
    the signal are excluded.

    Returns
    -------
    peri_time : np.ndarray
        Time relative to event.

    trials : np.ndarray
        Trial x time matrix.

    valid_event_times : np.ndarray
        Events included in the analysis.
    """

    event_times = np.asarray(
        event_times,
        dtype=float
    )

    peri_time = np.arange(
        window[0],
        window[1] + dt,
        dt
    )

    signal_start = float(signal.t[0])
    signal_end = float(signal.t[-1])

    valid_event_mask = (
        (event_times + window[0] >= signal_start)
        &
        (event_times + window[1] <= signal_end)
    )

    valid_event_times = event_times[
        valid_event_mask
    ]

    trials = []

    for event_time in valid_event_times:

        sample_times = (
            event_time + peri_time
        )

        sample_ts = nap.Ts(
            t=sample_times,
            time_units="s"
        )

        trial = signal.interpolate(
            sample_ts
        )

        trials.append(
            np.asarray(trial.d)
        )

    if len(trials) == 0:

        return (
            peri_time,
            np.empty(
                (0, len(peri_time))
            ),
            valid_event_times
        )

    trials = np.asarray(
        trials,
        dtype=float
    )

    return (
        peri_time,
        trials,
        valid_event_times
    )


# ============================================================
# Peri-event event-rate extraction
# ============================================================

def get_perievent_event_rate(
    event_times,
    align_times,
    window=(-5, 10),
    bin_size=0.1
):
    """
    Calculate event rate around alignment times.

    Useful for timestamp events such as individual licks.

    Returns
    -------
    bin_centers : np.ndarray
        Time relative to alignment event.

    rate_trials : np.ndarray
        Trial x time-bin event rate in Hz.
    """

    event_times = np.asarray(
        event_times,
        dtype=float
    )

    align_times = np.asarray(
        align_times,
        dtype=float
    )

    bin_edges = np.arange(
        window[0],
        window[1] + bin_size,
        bin_size
    )

    bin_centers = (
        bin_edges[:-1]
        + np.diff(bin_edges) / 2
    )

    rate_trials = []

    for align_time in align_times:

        relative_events = (
            event_times - align_time
        )

        counts, _ = np.histogram(
            relative_events,
            bins=bin_edges
        )

        rate = (
            counts / bin_size
        )

        rate_trials.append(
            rate
        )

    rate_trials = np.asarray(
        rate_trials,
        dtype=float
    )

    return (
        bin_centers,
        rate_trials
    )


# ============================================================
# Trial normalization
# ============================================================

def normalize_perievent_trials(
    peri_time,
    trials,
    normalization=None,
    baseline=None
):
    """
    Normalize peri-event trials.

    normalization
        None:
            No normalization.

        "subtract":
            Subtract each trial's baseline mean.

        "zscore":
            Subtract each trial's baseline mean and divide
            by that trial's baseline standard deviation.
    """

    peri_time = np.asarray(
        peri_time,
        dtype=float
    )

    trials = np.asarray(
        trials,
        dtype=float
    )

    if normalization is None:
        return trials

    if normalization not in (
        "subtract",
        "zscore"
    ):
        raise ValueError(
            "normalization must be None, "
            "'subtract', or 'zscore'."
        )

    if baseline is None:
        raise ValueError(
            "baseline must be provided when "
            "normalization is 'subtract' or 'zscore'."
        )

    baseline_mask = (
        (peri_time >= baseline[0])
        &
        (peri_time <= baseline[1])
    )

    if not np.any(baseline_mask):
        raise ValueError(
            "The specified baseline window does not "
            "overlap the peri-event time vector."
        )

    baseline_mean = np.nanmean(
        trials[:, baseline_mask],
        axis=1,
        keepdims=True
    )

    normalized_trials = (
        trials - baseline_mean
    )

    if normalization == "zscore":

        baseline_std = np.nanstd(
            trials[:, baseline_mask],
            axis=1,
            keepdims=True,
            ddof=1
        )

        baseline_std[
            ~np.isfinite(baseline_std)
            | (baseline_std == 0)
        ] = np.nan

        normalized_trials = (
            normalized_trials
            / baseline_std
        )

    return normalized_trials


# ============================================================
# Mean +/- SEM peri-event plot
# ============================================================

def plot_perievent_mean(
    peri_time,
    trials,
    ax=None,
    ylabel="Signal",
    title=None,
    event_label="Event",
    normalization=None,
    baseline=None
):
    """
    Plot mean +/- SEM for peri-event trials.
    """

    peri_time = np.asarray(
        peri_time,
        dtype=float
    )

    trials = normalize_perievent_trials(
        peri_time,
        trials,
        normalization=normalization,
        baseline=baseline
    )

    mean_signal = np.nanmean(
        trials,
        axis=0
    )

    n_valid = np.sum(
        np.isfinite(trials),
        axis=0
    )

    sem_signal = (
        np.nanstd(
            trials,
            axis=0,
            ddof=1
        )
        / np.sqrt(n_valid)
    )

    if ax is None:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

    ax.plot(
        peri_time,
        mean_signal
    )

    ax.fill_between(
        peri_time,
        mean_signal - sem_signal,
        mean_signal + sem_signal,
        alpha=0.3
    )

    ax.axvline(
        0,
        linestyle="--"
    )

    ax.set_xlabel(
        f"Time from {event_label.lower()} (s)"
    )

    ax.set_ylabel(
        ylabel
    )

    if title is not None:
        ax.set_title(title)

    return ax


# ============================================================
# Peri-event heatmap
# ============================================================

def plot_perievent_heatmap(
    peri_time,
    trials,
    ax=None,
    title=None,
    ylabel="Trial",
    colorbar_label="Signal",
    normalization=None,
    baseline=None,
    cmap="bwr",
    center_zero=True,
    interpolation="nearest"
):
    """
    Plot peri-event trials as a heatmap.

    Parameters
    ----------
    peri_time : array-like
        Time relative to event.

    trials : array-like
        Trial x time matrix.

    ax : matplotlib axis, optional
        Axis on which to draw the plot.

    title : str, optional
        Plot title.

    ylabel : str
        Y-axis label.

    colorbar_label : str
        Colorbar label.

    normalization : {None, "subtract", "zscore"}
        Trial normalization method.

    baseline : tuple or None
        Baseline window in seconds.

    cmap : str
        Matplotlib colormap.
        Default = "bwr".

    center_zero : bool
        If True, use symmetric color limits around zero.
        This makes negative values blue, zero white,
        and positive values red when using "bwr".

    interpolation : str
        Image interpolation method.
        Default = "nearest" to keep individual samples
        and trials crisp.

    Returns
    -------
    ax : matplotlib axis
        Axis containing the heatmap.
    """

    peri_time = np.asarray(
        peri_time,
        dtype=float
    )

    trials = normalize_perievent_trials(
        peri_time,
        trials,
        normalization=normalization,
        baseline=baseline
    )

    # --------------------------------------------------------
    # Create axis
    # --------------------------------------------------------

    if ax is None:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

    # --------------------------------------------------------
    # Color limits
    # --------------------------------------------------------

    if center_zero:

        max_abs = np.nanmax(
            np.abs(trials)
        )

        vmin = -max_abs
        vmax = max_abs

    else:

        vmin = None
        vmax = None

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    image = ax.imshow(
        trials,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            peri_time[0],
            peri_time[-1],
            trials.shape[0],
            0
        ],
        interpolation=interpolation
    )

    # --------------------------------------------------------
    # Event onset
    # --------------------------------------------------------

    ax.axvline(
        0,
        linestyle="--"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Time from event (s)"
    )

    ax.set_ylabel(
        ylabel
    )

    if title is not None:

        ax.set_title(
            title
        )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    plt.colorbar(
        image,
        ax=ax,
        label=colorbar_label
    )

    return ax

# ============================================================
# Combined photometry + licking + locomotion plot
# ============================================================

def plot_aligned_photometry_behavior(
    dff,
    locomotion,
    lick_times,
    align_times,
    window=(-5, 10),
    dt=0.02,
    lick_bin_size=0.1,
    event_label="Event",
    normalization=None,
    baseline=None
):
    """
    Plot dF/F, lick rate, and locomotion aligned to an event.

    Returns
    -------
    fig : matplotlib Figure

    axes : np.ndarray
        Three axes:
            0 = dF/F
            1 = lick rate
            2 = locomotion

    valid_event_times : np.ndarray
        Events included in all three panels.
    """

    # --------------------------------------------------------
    # dF/F
    # --------------------------------------------------------

    (
        peri_time,
        dff_trials,
        valid_event_times
    ) = get_perievent_trials(
        signal=dff,
        event_times=align_times,
        window=window,
        dt=dt
    )

    # --------------------------------------------------------
    # Locomotion
    # --------------------------------------------------------

    (
        loco_time,
        loco_trials,
        valid_loco_events
    ) = get_perievent_trials(
        signal=locomotion,
        event_times=valid_event_times,
        window=window,
        dt=dt
    )

    # If both continuous signals have different valid events,
    # keep only events valid for both.
    if not np.array_equal(
        valid_event_times,
        valid_loco_events
    ):

        common_events = np.intersect1d(
            valid_event_times,
            valid_loco_events
        )

        (
            peri_time,
            dff_trials,
            valid_event_times
        ) = get_perievent_trials(
            signal=dff,
            event_times=common_events,
            window=window,
            dt=dt
        )

        (
            loco_time,
            loco_trials,
            _
        ) = get_perievent_trials(
            signal=locomotion,
            event_times=common_events,
            window=window,
            dt=dt
        )

    # --------------------------------------------------------
    # Lick rate
    # --------------------------------------------------------

    (
        lick_time,
        lick_trials
    ) = get_perievent_event_rate(
        event_times=lick_times,
        align_times=valid_event_times,
        window=window,
        bin_size=lick_bin_size
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True
    )

    # --------------------------------------------------------
    # dF/F
    # --------------------------------------------------------

    dff_ylabel = "dF/F"

    if normalization == "zscore":
        dff_ylabel = "dF/F z-score"

    plot_perievent_mean(
        peri_time,
        dff_trials,
        ax=axes[0],
        ylabel=dff_ylabel,
        title=(
            f"{event_label}-aligned "
            "photometry and behavior"
        ),
        event_label=event_label,
        normalization=normalization,
        baseline=baseline
    )

    # --------------------------------------------------------
    # Licking
    # --------------------------------------------------------

    plot_perievent_mean(
        lick_time,
        lick_trials,
        ax=axes[1],
        ylabel="Lick rate (Hz)",
        event_label=event_label
    )

    # --------------------------------------------------------
    # Locomotion
    # --------------------------------------------------------

    plot_perievent_mean(
        loco_time,
        loco_trials,
        ax=axes[2],
        ylabel="Processed locomotion",
        event_label=event_label
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    axes[2].set_xlim(
        window[0],
        window[1]
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # IMPORTANT:
    # exactly THREE return values
    # --------------------------------------------------------

    return (
        fig,
        axes,
        valid_event_times
    )

def plot_irls_qc(
    session,
    channel=1,
    figsize=(12, 8)
):
    """
    Plot IRLS photometry correction quality for one
    photoreceiver channel.

    Parameters
    ----------
    session : dict
        Preprocessed session.

    channel : int
        Photoreceiver channel (1 or 2).

    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib Figure

    axes : np.ndarray
        Plot axes.
    """

    if channel not in (1, 2):
        raise ValueError(
            "channel must be 1 or 2."
        )

    # --------------------------------------------------------
    # Get signals
    # --------------------------------------------------------

    time = session[
        f"photo_time_465_ch{channel}"
    ]

    signal_465 = session[
        f"photometry_465_ch{channel}"
    ]

    signal_405 = session[
        f"photometry_405_aligned_ch{channel}"
    ]

    fitted_405 = session[
        f"photometry_405_fitted_ch{channel}"
    ]

    dff = session[
        f"dff_ch{channel}"
    ]

    # --------------------------------------------------------
    # Simple fit statistics
    # --------------------------------------------------------

    residual = (
        signal_465 - fitted_405
    )

    correlation = np.corrcoef(
        signal_465,
        signal_405
    )[0, 1]

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True
    )

    # --------------------------------------------------------
    # Raw 465 and aligned 405
    # --------------------------------------------------------

    axes[0].plot(
        time,
        signal_465,
        label="465"
    )

    axes[0].plot(
        time,
        signal_405,
        label="405 aligned"
    )

    axes[0].set_ylabel(
        "Photometry"
    )

    axes[0].set_title(
        f"Photometry channel {channel} — IRLS QC"
    )

    axes[0].legend()

    # --------------------------------------------------------
    # 465 and fitted reference
    # --------------------------------------------------------

    axes[1].plot(
        time,
        signal_465,
        label="465"
    )

    axes[1].plot(
        time,
        fitted_405,
        label="IRLS fitted 405"
    )

    axes[1].set_ylabel(
        "Photometry"
    )

    axes[1].legend()

    # --------------------------------------------------------
    # dF/F
    # --------------------------------------------------------

    axes[2].plot(
        time,
        dff
    )

    axes[2].axhline(
        0,
        linestyle="--"
    )

    axes[2].set_ylabel(
        "dF/F"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    # --------------------------------------------------------
    # Fit statistics
    # --------------------------------------------------------

    fig.text(
        0.99,
        0.01,
        (
            f"465/405 correlation = {correlation:.3f}\n"
            f"IRLS residual RMSE = {rmse:.4f}\n"
            f"dF/F range = "
            f"{np.nanmin(dff):.3f} to {np.nanmax(dff):.3f}"
        ),
        ha="right",
        va="bottom"
    )

    plt.tight_layout()

    return fig, axes