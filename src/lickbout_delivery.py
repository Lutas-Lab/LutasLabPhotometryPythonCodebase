from collections import defaultdict
from pathlib import Path
import csv
import warnings

import numpy as np

from .group_analysis import (
    _mean_and_sem,
    extract_perievent_trials,
    normalize_trials,
)
from .publication_figures import configure_publication_style, save_figure_formats
from .save_sessiondata import load_session
from .session_manifest import processed_session_path, resolve_session_channel


def match_cue_lickbout_delivery_trials(
    cue_onsets,
    lick_bout_onsets,
    delivery_onsets,
    recording_end,
    minimum_delivery_latency=0.0,
):
    """Pair the first lick bout and delivery within each cue-to-cue trial."""
    cues = np.sort(np.asarray(cue_onsets, dtype=float))
    bouts = np.sort(np.asarray(lick_bout_onsets, dtype=float))
    deliveries = np.sort(np.asarray(delivery_onsets, dtype=float))
    cues = cues[np.isfinite(cues)]
    bouts = bouts[np.isfinite(bouts)]
    deliveries = deliveries[np.isfinite(deliveries)]
    if not np.isfinite(recording_end):
        raise ValueError("recording_end must be finite.")
    if not np.isfinite(minimum_delivery_latency):
        raise ValueError("minimum_delivery_latency must be finite.")

    matches = []
    for cue_index, cue_onset in enumerate(cues):
        trial_end = cues[cue_index + 1] if cue_index + 1 < len(cues) else recording_end
        trial_bouts = bouts[(bouts >= cue_onset) & (bouts < trial_end)]
        trial_deliveries = deliveries[
            (deliveries >= cue_onset) & (deliveries < trial_end)
        ]
        if len(trial_bouts) == 0 or len(trial_deliveries) == 0:
            continue
        lick_bout_onset = float(trial_bouts[0])
        delivery_onset = float(trial_deliveries[0])
        delivery_latency = delivery_onset - lick_bout_onset
        if delivery_latency < minimum_delivery_latency:
            continue
        matches.append(
            {
                "cue_index": cue_index,
                "cue_onset": float(cue_onset),
                "lick_bout_onset": lick_bout_onset,
                "delivery_onset": delivery_onset,
                "delivery_latency": delivery_latency,
            }
        )
    return matches


def compute_lickbout_delivery_psth(
    sessions,
    data_root,
    *,
    window=(-5, 20),
    dt=0.02,
    normalization="zscore",
    baseline=(-5, 0),
    channel="manifest",
    minimum_delivery_latency=0.0,
):
    """Compute lick-bout-aligned photometry for cue trials with delivery."""
    session_results = []
    trial_rows = []
    trial_traces = []
    peri_time = None

    for info in sessions:
        selected_channel = resolve_session_channel(info, channel)
        signal_key = f"dff_ch{selected_channel}"
        time_key = f"photo_time_465_ch{selected_channel}"
        path = processed_session_path(data_root, info)
        session = load_session(path)
        required = {
            "cue_onset",
            "lick_bout_onset",
            "solenoid_onset",
            signal_key,
            time_key,
        }
        missing = required.difference(session)
        if missing:
            raise ValueError(f"{path} is missing analysis keys: {sorted(missing)}")

        signal_time = np.asarray(session[time_key], dtype=float)
        matches = match_cue_lickbout_delivery_trials(
            session["cue_onset"],
            session["lick_bout_onset"],
            session["solenoid_onset"],
            signal_time[-1],
            minimum_delivery_latency=minimum_delivery_latency,
        )
        if not matches:
            warnings.warn(
                f"Skipping {info['mouse']} {info['date']} run {info['run']}: "
                "no cue trials contained both a lick bout and delivery.",
                UserWarning,
                stacklevel=2,
            )
            continue

        bout_onsets = np.asarray(
            [match["lick_bout_onset"] for match in matches], dtype=float
        )
        this_time, traces, valid_indices = extract_perievent_trials(
            signal_time,
            session[signal_key],
            bout_onsets,
            window=window,
            dt=dt,
        )
        if len(traces) == 0:
            warnings.warn(
                f"Skipping {info['mouse']} {info['date']} run {info['run']}: "
                "no paired trials had complete lick-bout-aligned windows.",
                UserWarning,
                stacklevel=2,
            )
            continue
        normalized = normalize_trials(
            this_time,
            traces,
            normalization=normalization,
            baseline=baseline,
        )
        finite_rows = np.any(np.isfinite(normalized), axis=1)
        if not np.any(finite_rows):
            warnings.warn(
                f"Skipping {info['mouse']} {info['date']} run {info['run']}: "
                "normalization produced no finite trials.",
                UserWarning,
                stacklevel=2,
            )
            continue

        kept_indices = np.asarray(valid_indices, dtype=int)[finite_rows]
        normalized = normalized[finite_rows]
        session_mean = np.nanmean(normalized, axis=0)
        result = {
            **info,
            "path": path,
            "channel": selected_channel,
            "signal_key": signal_key,
            "n_events": len(normalized),
            "mean": session_mean,
        }
        session_results.append(result)
        for trace, match_index in zip(normalized, kept_indices):
            match = matches[match_index]
            trial_rows.append(
                {
                    "mouse": str(info["mouse"]),
                    "date": str(info["date"]),
                    "run": int(info["run"]),
                    "channel": selected_channel,
                    **match,
                }
            )
            trial_traces.append(trace)
        peri_time = this_time

    if not session_results:
        raise ValueError("No sessions contained usable paired lick-bout/delivery trials.")

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
            "channels": tuple(sorted({result["channel"] for result in results})),
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
        "event_key": "lick_bout_onset",
        "signal_key": (
            session_results[0]["signal_key"]
            if len({result["signal_key"] for result in session_results}) == 1
            else "manifest-selected dff channel"
        ),
        "signal_type": "photometry",
        "normalization": "none" if normalization is None else normalization,
        "null_method": "none",
        "n_shuffles": 0,
        "random_seed": 0,
        "null_exclusion": 0.0,
        "minimum_delivery_latency": float(minimum_delivery_latency),
        "trial_matrix": np.vstack(trial_traces),
        "trial_rows": trial_rows,
    }


def compute_lickbout_delivery_strata(sessions, data_root, **kwargs):
    """Compute paired-trial results for each manifest group and condition."""
    strata = defaultdict(list)
    for session in sessions:
        group = str(session.get("group", "all") or "all")
        condition = str(session.get("condition", "all") or "all")
        strata[(group, condition)].append(session)

    results = {}
    for (group, condition), stratum_sessions in sorted(strata.items()):
        result = compute_lickbout_delivery_psth(
            stratum_sessions,
            data_root,
            **kwargs,
        )
        result["group"] = group
        result["condition"] = condition
        results[(group, condition)] = result
    return results


def save_delivery_sorted_heatmap(
    results,
    output_dir,
    *,
    formats=("svg", "png"),
    dpi=300,
    font_family="Arial",
):
    """Save a lick-bout-aligned heatmap sorted by delivery latency."""
    import matplotlib.pyplot as plt

    configure_publication_style(font_family=font_family)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latencies = np.asarray(
        [row["delivery_latency"] for row in results["trial_rows"]], dtype=float
    )
    order = np.argsort(latencies)
    sorted_latencies = latencies[order]
    sorted_traces = np.asarray(results["trial_matrix"], dtype=float)[order]
    time = np.asarray(results["time"], dtype=float)
    half_step = np.median(np.diff(time)) / 2
    finite = np.abs(sorted_traces[np.isfinite(sorted_traces)])
    color_limit = float(np.percentile(finite, 98)) if len(finite) else 1.0
    if color_limit <= 0:
        color_limit = 1.0

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    image = ax.imshow(
        sorted_traces,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        extent=(time[0] - half_step, time[-1] + half_step, len(order), 0),
    )
    row_positions = np.arange(len(order), dtype=float) + 0.5
    ax.plot(
        sorted_latencies,
        row_positions,
        color="white",
        linewidth=1.2,
        label="Ensure delivery (solenoid onset)",
    )
    ax.scatter(
        sorted_latencies,
        row_positions,
        color="white",
        edgecolor="black",
        linewidth=0.25,
        s=8,
        zorder=3,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Time from lick-bout onset (s)",
        ylabel="Cue trials sorted by delivery latency",
        title=(
            f"{results.get('group', 'all')} / {results.get('condition', 'all')}: "
            f"lick-bout-aligned photometry\n"
            f"delivery latency ≥ {results['minimum_delivery_latency']:g} s; "
            f"{len(order)} paired trials, {results['n_mice']} mice"
        ),
    )
    ax.legend(loc="upper right", frameon=True)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(
        {
            "zscore": "Trial z-score",
            "subtract": "Baseline-subtracted dF/F",
            "none": "dF/F",
        }.get(results["normalization"], str(results["normalization"]))
    )
    fig.tight_layout()
    base = output_dir / "lick_bout_onset_delivery_sorted_heatmap"
    paths = save_figure_formats(fig, base, formats=formats, dpi=dpi)
    plt.close(fig)
    return paths


def save_delivery_trial_data(results, output_dir):
    """Save sorted heatmap values and trial metadata for audit and reuse."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latencies = np.asarray(
        [row["delivery_latency"] for row in results["trial_rows"]], dtype=float
    )
    order = np.argsort(latencies)
    npz_path = output_dir / "lick_bout_delivery_trials.npz"
    np.savez_compressed(
        npz_path,
        time=results["time"],
        trial_matrix=np.asarray(results["trial_matrix"])[order],
        delivery_latency=latencies[order],
        group_mean=results["group_mean"],
        group_sem=results["group_sem"],
        mouse_names=np.asarray(results["mouse_names"], dtype=str),
        mouse_matrix=results["mouse_matrix"],
        group=str(results.get("group", "all")),
        condition=str(results.get("condition", "all")),
        normalization=str(results["normalization"]),
        minimum_delivery_latency=results["minimum_delivery_latency"],
        mouse=np.asarray(
            [results["trial_rows"][index]["mouse"] for index in order], dtype=str
        ),
        date=np.asarray(
            [results["trial_rows"][index]["date"] for index in order], dtype=str
        ),
        run=np.asarray(
            [results["trial_rows"][index]["run"] for index in order], dtype=int
        ),
        cue_onset=np.asarray(
            [results["trial_rows"][index]["cue_onset"] for index in order],
            dtype=float,
        ),
        lick_bout_onset=np.asarray(
            [results["trial_rows"][index]["lick_bout_onset"] for index in order],
            dtype=float,
        ),
        delivery_onset=np.asarray(
            [results["trial_rows"][index]["delivery_onset"] for index in order],
            dtype=float,
        ),
    )
    csv_path = output_dir / "lick_bout_delivery_trials.csv"
    fields = (
        "mouse",
        "date",
        "run",
        "channel",
        "cue_index",
        "cue_onset",
        "lick_bout_onset",
        "delivery_onset",
        "delivery_latency",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in order:
            writer.writerow(results["trial_rows"][index])
    return npz_path, csv_path
