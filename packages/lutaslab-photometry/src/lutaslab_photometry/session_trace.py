"""Single-session continuous traces with behavioral event annotations."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .lifetime_workflows import load_processed_lifetime_session
from .save_sessiondata import load_session
from .session_manifest import processed_session_path, resolve_session_channel
from .update_check import local_git_commit

CONVENTIONAL_TRACE_SIGNALS = {
    "dff": ("photo_time_465_ch{channel}", "dff_ch{channel}", "ΔF/F"),
    "raw465": (
        "photo_time_465_ch{channel}",
        "photometry_465_ch{channel}",
        "a.u.",
    ),
    "raw405": (
        "photo_time_405_ch{channel}",
        "photometry_405_ch{channel}",
        "a.u.",
    ),
    "fitted405": (
        "photo_time_465_ch{channel}",
        "photometry_405_fitted_ch{channel}",
        "a.u.",
    ),
}


@dataclass(frozen=True)
class SessionTraceData:
    """Continuous signal and measured behavioral timing for one session."""

    workflow: str
    session_id: str
    signal: str
    units: str
    time: np.ndarray
    values: np.ndarray
    cue_starts: np.ndarray
    cue_ends: np.ndarray
    ensure_starts: np.ndarray
    ensure_ends: np.ndarray
    lick_times: np.ndarray
    source_path: str


def _array(value):
    return np.atleast_1d(np.asarray(value, dtype=float).squeeze())


def _valid_signal(time, values, *, label):
    time = _array(time)
    values = _array(values)
    if time.shape != values.shape or time.size < 2:
        raise ValueError(f"{label} does not contain a usable continuous timecourse")
    valid = np.isfinite(time) & np.isfinite(values)
    if np.sum(valid) < 2:
        raise ValueError(f"{label} contains fewer than two finite samples")
    time = time[valid]
    values = values[valid]
    order = np.argsort(time, kind="stable")
    return time[order], values[order]


def _intervals(starts, ends):
    raw_starts = _array(starts)
    raw_ends = _array(ends)
    source_indices = np.flatnonzero(np.isfinite(raw_starts))
    if source_indices.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    starts = raw_starts[source_indices]
    matched_ends = np.full(starts.size, np.nan)
    for output_index, source_index in enumerate(source_indices):
        if source_index >= raw_ends.size:
            continue
        end = raw_ends[source_index]
        if np.isfinite(end) and end >= starts[output_index]:
            matched_ends[output_index] = end
    return starts, matched_ends


def load_conventional_session_trace(row, data_root, signal="dff", channel="manifest"):
    """Load one conventional processed session for continuous-trace display."""

    if signal not in CONVENTIONAL_TRACE_SIGNALS:
        raise ValueError(f"Unknown conventional trace signal: {signal}")
    selected_channel = resolve_session_channel(row, channel)
    path = processed_session_path(data_root, row)
    session = load_session(path)
    time_template, value_template, units = CONVENTIONAL_TRACE_SIGNALS[signal]
    time_key = time_template.format(channel=selected_channel)
    value_key = value_template.format(channel=selected_channel)
    missing = {time_key, value_key}.difference(session)
    if missing:
        raise ValueError(f"{path} is missing trace fields: {sorted(missing)}")
    time, values = _valid_signal(
        session[time_key], session[value_key], label=f"{signal} channel {selected_channel}"
    )
    cue_starts, cue_ends = _intervals(
        session.get("cue_onset", ()), session.get("cue_offset", ())
    )
    ensure_starts, ensure_ends = _intervals(
        session.get("solenoid_onset", ()), session.get("solenoid_offset", ())
    )
    session_id = (
        f"{row['mouse']}_{row['date']}_run{int(row['run']):03d}_ch{selected_channel}"
    )
    return SessionTraceData(
        workflow="conventional",
        session_id=session_id,
        signal=signal,
        units=units,
        time=time,
        values=values,
        cue_starts=cue_starts,
        cue_ends=cue_ends,
        ensure_starts=ensure_starts,
        ensure_ends=ensure_ends,
        lick_times=_array(session.get("lick_times", ())),
        source_path=str(path),
    )


def load_lifetime_session_trace(workflow, row, data_root, signal):
    """Load one FluoPulse or iFLIP3 processed session for trace display."""

    session, paths = load_processed_lifetime_session(workflow, row, data_root)
    if signal not in session.continuous:
        raise ValueError(
            f"Processed {workflow} session does not contain the {signal!r} signal"
        )
    continuous = session.continuous[signal]
    time, values = _valid_signal(
        continuous.timestamps, continuous.values, label=signal
    )
    cue = session.intervals.get("visual_cue")
    ensure = session.intervals.get("ensure")
    cue_events = session.events.get("visual_cue")
    ensure_events = session.events.get("ensure")
    cue_starts, cue_ends = _intervals(
        (
            cue.starts
            if cue is not None
            else (() if cue_events is None else cue_events.timestamps)
        ),
        () if cue is None else cue.ends,
    )
    ensure_starts, ensure_ends = _intervals(
        (
            ensure.starts
            if ensure is not None
            else (() if ensure_events is None else ensure_events.timestamps)
        ),
        () if ensure is None else ensure.ends,
    )
    return SessionTraceData(
        workflow=workflow,
        session_id=session.session_id,
        signal=signal,
        units=continuous.units,
        time=time,
        values=values,
        cue_starts=cue_starts,
        cue_ends=cue_ends,
        ensure_starts=ensure_starts,
        ensure_ends=ensure_ends,
        lick_times=_array(
            session.events.get("licks").timestamps
            if "licks" in session.events
            else ()
        ),
        source_path=paths["processed_path"],
    )


def transform_trace(values, mode):
    """Apply a display-only transform using the full recorded session."""

    values = np.asarray(values, dtype=float)
    if mode == "native":
        return values.copy(), None
    center = float(np.nanmean(values))
    if mode == "subtract_mean":
        return values - center, center
    if mode == "zscore":
        scale = float(np.nanstd(values))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("The full-session signal has zero variance and cannot be z-scored")
        return (values - center) / scale, (center, scale)
    raise ValueError("Trace transform must be native, subtract_mean, or zscore")


def _window_mask(time, window):
    start, end = map(float, window)
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError("Trace windows must contain increasing finite values")
    mask = (time >= start) & (time <= end)
    if np.sum(mask) < 2:
        raise ValueError(f"Trace window {start:g}–{end:g} s contains fewer than two samples")
    return mask


def _downsample(time, values, max_points=50_000):
    if time.size <= max_points:
        return time, values
    indices = np.linspace(0, time.size - 1, max_points, dtype=int)
    return time[indices], values[indices]


def _annotate_axis(ax, data, window, *, show_cues, show_ensures, show_licks):
    start, end = window
    if show_cues:
        for onset, offset in zip(data.cue_starts, data.cue_ends, strict=True):
            if np.isfinite(offset) and offset >= start and onset <= end:
                ax.axvspan(onset, offset, color="#6f78a8", alpha=0.15, linewidth=0)
            elif start <= onset <= end:
                ax.axvline(onset, color="#6f78a8", alpha=0.45, linewidth=1.0)
    if show_ensures:
        for onset, offset in zip(data.ensure_starts, data.ensure_ends, strict=True):
            if np.isfinite(offset) and offset >= start and onset <= end:
                ax.axvspan(onset, offset, color="#54b948", alpha=0.22, linewidth=0)
            elif start <= onset <= end:
                ax.axvline(onset, color="#54b948", alpha=0.7, linewidth=1.0)
    if show_licks:
        licks = data.lick_times[(data.lick_times >= start) & (data.lick_times <= end)]
        if licks.size:
            low, high = ax.get_ylim()
            span = high - low
            tick_low = high + 0.02 * span
            tick_high = high + 0.09 * span
            ax.vlines(licks, tick_low, tick_high, color="#3f3f3f", linewidth=0.7)
            ax.set_ylim(low, high + 0.12 * span)


def plot_session_trace(
    data,
    *,
    overview_window,
    zoom_window,
    transform="native",
    show_cues=True,
    show_ensures=True,
    show_licks=True,
):
    """Create a two-panel overview and zoomed continuous-session figure."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    transformed, parameters = transform_trace(data.values, transform)
    overview_mask = _window_mask(data.time, overview_window)
    zoom_mask = _window_mask(data.time, zoom_window)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
    for ax, mask, window, title in (
        (axes[0], overview_mask, overview_window, "Session overview"),
        (axes[1], zoom_mask, zoom_window, "Selected zoom"),
    ):
        plot_time, plot_values = _downsample(data.time[mask], transformed[mask])
        ax.plot(plot_time, plot_values, color="#174a8b", linewidth=1.1)
        ax.set_xlim(window)
        ax.set_title(title, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        _annotate_axis(
            ax,
            data,
            window,
            show_cues=show_cues,
            show_ensures=show_ensures,
            show_licks=show_licks,
        )
    ylabel = {
        "native": f"{data.signal} ({data.units or 'a.u.'})",
        "subtract_mean": f"Δ{data.signal} ({data.units or 'a.u.'})",
        "zscore": f"{data.signal} (full-session z-score)",
    }[transform]
    axes[0].set_ylabel(ylabel)
    axes[1].set(xlabel="Session time (s)", ylabel=ylabel)
    handles = []
    if show_licks:
        handles.append(Line2D([], [], color="#3f3f3f", marker="|", linestyle="None", label="Lick"))
    if show_cues:
        handles.append(Patch(facecolor="#6f78a8", alpha=0.25, label="Cue"))
    if show_ensures:
        handles.append(Patch(facecolor="#54b948", alpha=0.3, label="Ensure"))
    if handles:
        axes[0].legend(
            handles=handles,
            frameon=False,
            ncol=len(handles),
            loc="lower right",
            bbox_to_anchor=(1.0, 1.01),
            borderaxespad=0.0,
        )
    fig.suptitle(f"{data.session_id} · {data.signal}")
    return fig, transformed, parameters


def save_session_trace(
    data,
    output_dir,
    *,
    overview_window,
    zoom_window,
    transform="native",
    show_cues=True,
    show_ensures=True,
    show_licks=True,
    formats=("png", "svg"),
    dpi=300,
):
    """Save the figure, displayed numeric trace, events, and reproducibility metadata."""

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, transformed, parameters = plot_session_trace(
        data,
        overview_window=overview_window,
        zoom_window=zoom_window,
        transform=transform,
        show_cues=show_cues,
        show_ensures=show_ensures,
        show_licks=show_licks,
    )
    outputs = []
    for extension in formats:
        path = output_dir / f"session_trace.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    mask = _window_mask(data.time, overview_window)
    trace_path = output_dir / "session_trace_values.csv"
    with trace_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "value"))
        writer.writerows(zip(data.time[mask], transformed[mask], strict=True))
    outputs.append(trace_path)
    event_path = output_dir / "session_trace_events.csv"
    with event_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("event", "start_s", "end_s"))
        writer.writerows(
            ("cue", start, end)
            for start, end in zip(data.cue_starts, data.cue_ends, strict=True)
        )
        writer.writerows(
            ("ensure", start, end)
            for start, end in zip(data.ensure_starts, data.ensure_ends, strict=True)
        )
        writer.writerows(("lick", timestamp, timestamp) for timestamp in data.lick_times)
    outputs.append(event_path)
    metadata_path = output_dir / "session_trace_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "workflow": data.workflow,
                "session_id": data.session_id,
                "signal": data.signal,
                "native_units": data.units,
                "source_path": data.source_path,
                "overview_window": list(map(float, overview_window)),
                "zoom_window": list(map(float, zoom_window)),
                "transform": transform,
                "transform_parameters": parameters,
                "show_cues": show_cues,
                "show_ensures": show_ensures,
                "show_licks": show_licks,
                "generated_utc": datetime.now(UTC).isoformat(),
                "code_commit": local_git_commit(Path(__file__).resolve().parents[4]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs.append(metadata_path)
    return outputs
