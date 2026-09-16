from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np

from .save_sessiondata import load_session
from .session_manifest import processed_session_path


def _validate_window(window):
    if len(window) != 2 or window[0] >= window[1]:
        raise ValueError("window must contain increasing start and end values.")
    return float(window[0]), float(window[1])


def _peri_time(window, dt):
    start, end = _validate_window(window)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    count = int(np.floor((end - start) / dt + 0.5))
    return start + np.arange(count + 1, dtype=float) * dt


def extract_perievent_trials(signal_time, signal, event_times, window=(-5, 10), dt=0.02):
    """Interpolate a continuous signal around events with complete windows."""
    signal_time = np.asarray(signal_time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    if signal_time.ndim != 1 or signal.ndim != 1:
        raise ValueError("signal_time and signal must be one-dimensional.")
    if len(signal_time) != len(signal) or len(signal_time) < 2:
        raise ValueError("signal_time and signal must have equal nontrivial lengths.")
    if not np.all(np.diff(signal_time) > 0):
        raise ValueError("signal_time must be strictly increasing.")

    start, end = _validate_window(window)
    peri_time = _peri_time(window, dt)
    valid = np.flatnonzero(
        np.isfinite(event_times)
        & (event_times + start >= signal_time[0])
        & (event_times + end <= signal_time[-1])
    )
    trials = np.empty((len(valid), len(peri_time)), dtype=float)
    for row, event_index in enumerate(valid):
        trials[row] = np.interp(
            event_times[event_index] + peri_time, signal_time, signal
        )
    return peri_time, trials, valid


def normalize_trials(peri_time, trials, normalization="zscore", baseline=(-5, 0)):
    """Apply trial-local baseline subtraction or z-scoring."""
    peri_time = np.asarray(peri_time, dtype=float)
    trials = np.asarray(trials, dtype=float)
    if normalization in (None, "none"):
        return trials.copy()
    if normalization not in ("subtract", "zscore"):
        raise ValueError("normalization must be 'none', 'subtract', or 'zscore'.")
    if baseline is None or len(baseline) != 2 or baseline[0] >= baseline[1]:
        raise ValueError("baseline must contain increasing start and end values.")

    baseline_mask = (peri_time >= baseline[0]) & (peri_time < baseline[1])
    if not np.any(baseline_mask):
        raise ValueError("baseline does not overlap the peri-event time vector.")
    baseline_values = trials[:, baseline_mask]
    baseline_mean = np.nanmean(baseline_values, axis=1, keepdims=True)
    output = trials - baseline_mean
    if normalization == "zscore":
        baseline_std = np.nanstd(baseline_values, axis=1, ddof=1, keepdims=True)
        baseline_std[(baseline_std <= 0) | ~np.isfinite(baseline_std)] = np.nan
        output = output / baseline_std
    return output


def _mean_and_sem(rows):
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("rows must be a nonempty two-dimensional array.")
    mean = np.nanmean(rows, axis=0)
    count = np.sum(np.isfinite(rows), axis=0)
    sem = np.full(rows.shape[1], np.nan, dtype=float)
    enough = count > 1
    if np.any(enough):
        sem[enough] = np.nanstd(rows[:, enough], axis=0, ddof=1) / np.sqrt(
            count[enough]
        )
    return mean, sem


def compute_manifest_psth(
    sessions,
    data_root,
    *,
    event_key="cue_onset",
    channel=1,
    window=(-5, 10),
    dt=0.02,
    normalization="zscore",
    baseline=(-5, 0),
):
    """Compute session, mouse, and group PSTHs with mice as the group unit."""
    signal_key = f"dff_ch{int(channel)}"
    time_key = f"photo_time_465_ch{int(channel)}"
    session_results = []

    for info in sessions:
        path = processed_session_path(data_root, info)
        session = load_session(path)
        missing = {event_key, signal_key, time_key}.difference(session)
        if missing:
            raise ValueError(f"{path} is missing analysis keys: {sorted(missing)}")

        peri_time, trials, valid_indices = extract_perievent_trials(
            session[time_key],
            session[signal_key],
            session[event_key],
            window=window,
            dt=dt,
        )
        if len(trials) == 0:
            warnings.warn(
                f"Skipping {info['mouse']} {info['date']} run {info['run']}: "
                f"no complete {event_key} windows.",
                UserWarning,
                stacklevel=2,
            )
            continue
        normalized = normalize_trials(
            peri_time, trials, normalization=normalization, baseline=baseline
        )
        session_mean = np.nanmean(normalized, axis=0)
        if not np.any(np.isfinite(session_mean)):
            warnings.warn(
                f"Skipping {info['mouse']} {info['date']} run {info['run']}: "
                "normalization produced no finite samples.",
                UserWarning,
                stacklevel=2,
            )
            continue
        session_results.append(
            {
                **info,
                "path": path,
                "n_events": len(valid_indices),
                "mean": session_mean,
            }
        )

    if not session_results:
        raise ValueError("No sessions contained usable peri-event trials.")

    grouped = defaultdict(list)
    for result in session_results:
        grouped[result["mouse"]].append(result)

    mouse_results = {}
    for mouse, results in grouped.items():
        session_matrix = np.vstack([result["mean"] for result in results])
        mean, sem = _mean_and_sem(session_matrix)
        mouse_results[mouse] = {
            "mean": mean,
            "sem": sem,
            "session_matrix": session_matrix,
            "n_sessions": len(results),
            "n_events": sum(result["n_events"] for result in results),
        }

    mouse_names = sorted(mouse_results)
    mouse_matrix = np.vstack([mouse_results[mouse]["mean"] for mouse in mouse_names])
    group_mean, group_sem = _mean_and_sem(mouse_matrix)

    return {
        "time": peri_time,
        "session_results": session_results,
        "mouse_names": mouse_names,
        "mouse_results": mouse_results,
        "mouse_matrix": mouse_matrix,
        "group_mean": group_mean,
        "group_sem": group_sem,
        "n_mice": len(mouse_names),
        "event_key": event_key,
        "signal_key": signal_key,
        "normalization": "none" if normalization is None else normalization,
    }


def save_psth_figures(results, output_dir, *, figure_level="both", dpi=150):
    """Save one figure per mouse and/or a mouse-level group mean ± SEM."""
    import matplotlib.pyplot as plt

    if figure_level not in ("individual", "group", "both"):
        raise ValueError("figure_level must be 'individual', 'group', or 'both'.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    time = results["time"]
    ylabel = "dF/F" if results["normalization"] == "none" else results["normalization"]
    saved = []

    if figure_level in ("individual", "both"):
        for mouse in results["mouse_names"]:
            mouse_result = results["mouse_results"][mouse]
            fig, ax = plt.subplots(figsize=(8, 5))
            for trace in mouse_result["session_matrix"]:
                ax.plot(time, trace, color="0.7", linewidth=1)
            ax.plot(time, mouse_result["mean"], linewidth=2, label=f"{mouse} mean")
            if mouse_result["n_sessions"] > 1:
                ax.fill_between(
                    time,
                    mouse_result["mean"] - mouse_result["sem"],
                    mouse_result["mean"] + mouse_result["sem"],
                    alpha=0.25,
                    label="SEM across sessions",
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axhline(0, color="black", linestyle=":", linewidth=1)
            ax.set(
                xlabel=f"Time from {results['event_key']} (s)",
                ylabel=ylabel,
                title=(
                    f"{mouse}: {results['event_key']} PSTH "
                    f"({mouse_result['n_sessions']} sessions)"
                ),
            )
            ax.legend()
            fig.tight_layout()
            path = output_dir / f"{mouse}_{results['event_key']}_psth.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            saved.append(path)

    if figure_level in ("group", "both"):
        fig, ax = plt.subplots(figsize=(9, 6))
        for mouse, trace in zip(results["mouse_names"], results["mouse_matrix"]):
            ax.plot(time, trace, alpha=0.35, linewidth=1, label=mouse)
        ax.plot(time, results["group_mean"], color="black", linewidth=3, label="Group mean")
        if results["n_mice"] > 1:
            ax.fill_between(
                time,
                results["group_mean"] - results["group_sem"],
                results["group_mean"] + results["group_sem"],
                color="black",
                alpha=0.2,
                label="SEM across mice",
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        ax.set(
            xlabel=f"Time from {results['event_key']} (s)",
            ylabel=ylabel,
            title=f"Group {results['event_key']} PSTH ({results['n_mice']} mice)",
        )
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"group_{results['event_key']}_psth.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)

    return saved
