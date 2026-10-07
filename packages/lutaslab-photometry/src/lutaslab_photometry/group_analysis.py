import csv
import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from lutaslab_core.perievent import (
    extract_perievent_event_rate as _extract_perievent_event_rate,
)
from lutaslab_core.perievent import (
    extract_perievent_trials as _extract_perievent_trials,
)
from lutaslab_core.perievent import (
    generate_null_onsets as _generate_null_onsets,
)
from lutaslab_core.perievent import (
    normalize_trials as _normalize_trials,
)

from .save_sessiondata import load_session
from .session_manifest import processed_session_path, resolve_session_channel
from .trial_classification import TRIAL_CLASS_KEYS, cue_trial_mask

_BEHAVIORAL_EVENT_KEYS = (
    "cue_onset",
    "solenoid_onset",
    "lick_times",
    "lick_bout_onset",
    "lick_bout_duration",
    "lick_bout_lick_count",
)


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
    return _extract_perievent_trials(signal_time, signal, event_times, window, dt)


def extract_perievent_event_rate(
    event_times,
    alignment_times,
    recording_bounds,
    window=(-5, 10),
    dt=0.1,
):
    """Bin discrete events as rates around alignments with complete windows."""
    return _extract_perievent_event_rate(
        event_times,
        alignment_times,
        recording_bounds,
        window,
        dt,
    )


def normalize_trials(peri_time, trials, normalization="zscore", baseline=(-5, 0)):
    """Apply trial-local baseline subtraction or z-scoring."""
    return _normalize_trials(peri_time, trials, normalization, baseline)


def generate_null_onsets(
    event_times,
    onset_bounds,
    *,
    n_shuffles=500,
    method="random_onsets",
    exclusion=0.0,
    rng=None,
):
    """Generate session-local random or circularly shifted event onsets."""
    return _generate_null_onsets(
        event_times,
        onset_bounds,
        n_shuffles=n_shuffles,
        method=method,
        exclusion=exclusion,
        rng=rng,
    )


def _session_rng(seed, info):
    identifier = f"{seed}|{info['mouse']}|{info['date']}|{int(info['run'])}"
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _null_session_means(
    signal_time,
    signal,
    real_event_times,
    *,
    window,
    dt,
    normalization,
    baseline,
    n_shuffles,
    null_method,
    null_exclusion,
    rng,
    real_trials,
    behavioral_events,
):
    start, end = _validate_window(window)
    onset_bounds = (float(signal_time[0]) - start, float(signal_time[-1]) - end)
    shuffled_onsets = generate_null_onsets(
        real_event_times,
        onset_bounds,
        n_shuffles=n_shuffles,
        method=null_method,
        exclusion=null_exclusion,
        rng=rng,
    )
    means = []
    baseline_mask = (peri_time := _peri_time(window, dt)) >= baseline[0]
    baseline_mask &= peri_time < baseline[1]
    real_baseline_std = np.nanstd(
        np.asarray(real_trials, dtype=float)[:, baseline_mask], axis=1, ddof=1
    )
    null_baseline_std = []
    null_trial_max_abs_z = []
    event_distance = {"alignment": []}
    event_distance.update(
        {
            name: []
            for name, values in behavioral_events.items()
            if np.asarray(values).size
        }
    )
    amplitude_by_normalization = {"none": [], "subtract": [], "zscore": []}
    example_trials = None
    example_onsets = None
    for shuffle_index, onsets in enumerate(shuffled_onsets):
        peri_time, trials, _ = extract_perievent_trials(
            signal_time, signal, onsets, window=window, dt=dt
        )
        baseline_values = trials[:, baseline_mask]
        null_baseline_std.extend(
            np.nanstd(baseline_values, axis=1, ddof=1).tolist()
        )
        event_distance["alignment"].extend(
            np.min(
                np.abs(onsets[:, None] - real_event_times[None, :]), axis=1
            ).tolist()
        )
        for name, event_times in behavioral_events.items():
            event_times = np.asarray(event_times, dtype=float)
            if event_times.size:
                event_distance[name].extend(
                    np.min(
                        np.abs(onsets[:, None] - event_times[None, :]), axis=1
                    ).tolist()
                )
        normalized_by_method = {
            method: normalize_trials(
                peri_time,
                trials,
                normalization=method,
                baseline=baseline,
            )
            for method in ("none", "subtract", "zscore")
        }
        for trial in normalized_by_method["zscore"]:
            finite_trial = trial[np.isfinite(trial)]
            null_trial_max_abs_z.append(
                float(np.max(np.abs(finite_trial))) if finite_trial.size else np.nan
            )
        for method, method_trials in normalized_by_method.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                shuffle_mean = np.nanmean(method_trials, axis=0)
            finite = shuffle_mean[np.isfinite(shuffle_mean)]
            amplitude_by_normalization[method].append(
                float(np.max(np.abs(finite))) if finite.size else np.nan
            )
        normalized = normalized_by_method[
            "none" if normalization in (None, "none") else normalization
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means.append(np.nanmean(normalized, axis=0))
        if shuffle_index == 0:
            example_trials = normalized[: min(12, len(normalized))]
            example_onsets = onsets[: min(12, len(onsets))]
    return np.asarray(means, dtype=float), {
        "real_baseline_std": np.asarray(real_baseline_std, dtype=float),
        "null_baseline_std": np.asarray(null_baseline_std, dtype=float),
        "null_trial_max_abs_z": np.asarray(null_trial_max_abs_z, dtype=float),
        "null_onset_distance": np.asarray(event_distance["alignment"], dtype=float),
        "null_event_distances": {
            name: np.asarray(values, dtype=float)
            for name, values in event_distance.items()
        },
        "max_abs_shuffle_mean": {
            method: np.asarray(values, dtype=float)
            for method, values in amplitude_by_normalization.items()
        },
        "example_trials": np.asarray(example_trials, dtype=float),
        "example_onsets": np.asarray(example_onsets, dtype=float),
        "peri_time": np.asarray(peri_time, dtype=float),
        "onset_bounds": np.asarray(onset_bounds, dtype=float),
    }


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


def _psth_ylabel(normalization, signal_type="photometry"):
    if signal_type == "licking":
        return {
            "none": "Lick rate (Hz)",
            None: "Lick rate (Hz)",
            "subtract": "Baseline-subtracted lick rate (Hz)",
            "zscore": "Trial z-score of lick rate",
        }.get(normalization, str(normalization))
    return {
        "none": "dF/F",
        None: "dF/F",
        "subtract": "Baseline-subtracted dF/F",
        "zscore": "Trial z-score",
    }.get(normalization, str(normalization))


def _psth_context(results):
    values = [
        str(results[key])
        for key in ("group", "condition")
        if results.get(key) not in (None, "", "all")
    ]
    return " / ".join(values)


def _psth_description(results):
    if results.get("description"):
        return str(results["description"])
    event_label = str(results["event_key"]).replace("_", " ")
    trial_class = str(results.get("trial_class", "all")).replace("_", " ")
    trial_suffix = (
        ""
        if trial_class == "all"
        else (
            f" ({trial_class} trials; "
            f"post-cue {results.get('post_cue_window', 2.0):g} s)"
        )
    )
    if results.get("signal_type", "photometry") == "licking":
        return f"licking aligned to {event_label}{trial_suffix}"
    return f"{event_label}-aligned photometry{trial_suffix}"


def compute_manifest_psth(
    sessions,
    data_root,
    *,
    event_key="cue_onset",
    signal_type="photometry",
    channel="manifest",
    window=(-5, 10),
    dt=0.02,
    normalization="zscore",
    baseline=(-5, 0),
    null_method="none",
    n_shuffles=500,
    random_seed=0,
    null_exclusion=0.0,
    trial_class="all",
    post_cue_window=2.0,
):
    """Compute session, mouse, and group PSTHs with mice as the group unit."""
    if signal_type not in ("photometry", "licking"):
        raise ValueError("signal_type must be 'photometry' or 'licking'.")
    if signal_type == "licking" and null_method != "none":
        raise ValueError("Null alignment is not yet supported for the licking response.")
    if null_method not in ("none", "random_onsets", "circular_shift"):
        raise ValueError(
            "null_method must be 'none', 'random_onsets', or 'circular_shift'."
        )
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer.")
    if null_method != "none" and (
        not isinstance(n_shuffles, int)
        or isinstance(n_shuffles, bool)
        or n_shuffles < 1
    ):
        raise ValueError("n_shuffles must be a positive integer.")
    if trial_class not in TRIAL_CLASS_KEYS:
        raise ValueError(f"trial_class must be one of {TRIAL_CLASS_KEYS}.")
    if not np.isfinite(post_cue_window) or post_cue_window < 0:
        raise ValueError("post_cue_window must be finite and nonnegative.")
    if trial_class != "all" and event_key != "cue_onset":
        raise ValueError("Cue-trial classes can only filter cue_onset analyses.")

    session_results = []

    for info in sessions:
        selected_channel = resolve_session_channel(info, channel)
        signal_key = f"dff_ch{selected_channel}"
        time_key = f"photo_time_465_ch{selected_channel}"
        path = processed_session_path(data_root, info)
        session = load_session(path)
        required = {event_key, time_key}
        required.add(signal_key if signal_type == "photometry" else "lick_times")
        missing = required.difference(session)
        if missing:
            raise ValueError(f"{path} is missing analysis keys: {sorted(missing)}")
        event_times = np.asarray(session[event_key], dtype=float)
        if trial_class != "all":
            mask = cue_trial_mask(session, trial_class, post_cue_window)
            event_times = event_times[mask]

        if signal_type == "photometry":
            peri_time, trials, valid_indices = extract_perievent_trials(
                session[time_key],
                session[signal_key],
                event_times,
                window=window,
                dt=dt,
            )
        else:
            recording_time = np.asarray(session[time_key], dtype=float)
            peri_time, trials, valid_indices = extract_perievent_event_rate(
                session["lick_times"],
                event_times,
                (recording_time[0], recording_time[-1]),
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
        result = {
            **info,
            "path": path,
            "channel": selected_channel,
            "signal_key": signal_key if signal_type == "photometry" else "lick_times",
            "n_events": len(valid_indices),
            "alignment_times": event_times[valid_indices],
            "behavioral_events": {
                key: np.asarray(session.get(key, []), dtype=float)
                for key in _BEHAVIORAL_EVENT_KEYS
            },
            "trials": normalized,
            "mean": session_mean,
        }
        if null_method != "none":
            valid_event_times = event_times[valid_indices]
            result["null_means"], result["null_diagnostics"] = _null_session_means(
                np.asarray(session[time_key], dtype=float),
                np.asarray(session[signal_key], dtype=float),
                valid_event_times,
                window=window,
                dt=dt,
                normalization=normalization,
                baseline=baseline,
                n_shuffles=n_shuffles,
                null_method=null_method,
                null_exclusion=null_exclusion,
                rng=_session_rng(random_seed, info),
                real_trials=trials,
                behavioral_events={
                    key: np.asarray(session.get(key, []), dtype=float)
                    for key in (
                        "cue_onset",
                        "solenoid_onset",
                        "lick_times",
                        "lick_bout_onset",
                    )
                },
            )
        session_results.append(result)

    if not session_results:
        raise ValueError("No sessions contained usable peri-event trials.")

    grouped = defaultdict(list)
    for result in session_results:
        grouped[result["mouse"]].append(result)

    mouse_results = {}
    for mouse, results in grouped.items():
        session_matrix = np.vstack([result["mean"] for result in results])
        mean, sem = _mean_and_sem(session_matrix)
        mouse_result = {
            "mean": mean,
            "sem": sem,
            "session_matrix": session_matrix,
            "n_sessions": len(results),
            "n_events": sum(result["n_events"] for result in results),
            "channels": tuple(sorted({result["channel"] for result in results})),
        }
        if null_method != "none":
            null_session_stack = np.stack(
                [result["null_means"] for result in results], axis=0
            )
            null_matrix = np.nanmean(null_session_stack, axis=0)
            mouse_result.update(
                {
                    "null_matrix": null_matrix,
                    "null_mean": np.nanmean(null_matrix, axis=0),
                    "null_lower": np.nanpercentile(null_matrix, 2.5, axis=0),
                    "null_upper": np.nanpercentile(null_matrix, 97.5, axis=0),
                }
            )
        mouse_results[mouse] = mouse_result

    mouse_names = sorted(mouse_results)
    mouse_matrix = np.vstack([mouse_results[mouse]["mean"] for mouse in mouse_names])
    group_mean, group_sem = _mean_and_sem(mouse_matrix)

    null_results = {}
    if null_method != "none":
        mouse_null_stack = np.stack(
            [mouse_results[mouse]["null_matrix"] for mouse in mouse_names], axis=0
        )
        group_null_matrix = np.nanmean(mouse_null_stack, axis=0)
        null_results = {
            "mouse_null_mean_matrix": np.vstack(
                [mouse_results[mouse]["null_mean"] for mouse in mouse_names]
            ),
            "group_null_matrix": group_null_matrix,
            "group_null_mean": np.nanmean(group_null_matrix, axis=0),
            "group_null_lower": np.nanpercentile(group_null_matrix, 2.5, axis=0),
            "group_null_upper": np.nanpercentile(group_null_matrix, 97.5, axis=0),
        }

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
        "signal_type": signal_type,
        "signal_key": (
            session_results[0]["signal_key"]
            if len({result["signal_key"] for result in session_results}) == 1
            else "manifest-selected dff channel"
        ),
        "normalization": "none" if normalization is None else normalization,
        "null_method": null_method,
        "n_shuffles": n_shuffles if null_method != "none" else 0,
        "random_seed": random_seed,
        "null_exclusion": null_exclusion,
        "trial_class": trial_class,
        "post_cue_window": float(post_cue_window),
        **null_results,
    }


def compute_manifest_psth_strata(sessions, data_root, **kwargs):
    """Compute independent PSTHs for every manifest group and condition."""
    strata = defaultdict(list)
    for session in sessions:
        group = str(session.get("group", "all") or "all")
        condition = str(session.get("condition", "all") or "all")
        strata[(group, condition)].append(session)

    results = {}
    for (group, condition), stratum_sessions in sorted(strata.items()):
        result = compute_manifest_psth(stratum_sessions, data_root, **kwargs)
        result["group"] = group
        result["condition"] = condition
        results[(group, condition)] = result
    return results


def save_condition_comparison_figures(
    stratum_results,
    output_dir,
    *,
    formats=("svg", "png"),
    dpi=300,
    font_family="Arial",
):
    """Save condition PSTHs and paired within-mouse differences for each group."""
    import matplotlib.pyplot as plt

    from .publication_figures import configure_publication_style, save_figure_formats

    configure_publication_style(font_family=font_family)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(dict)
    for (group, condition), result in stratum_results.items():
        grouped[group][condition] = result

    saved = []
    colors = plt.get_cmap("tab10")

    def safe_label(value):
        return "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in str(value)
        )

    for group, condition_results in sorted(grouped.items()):
        conditions = sorted(condition_results)
        if len(conditions) < 2:
            continue
        reference = condition_results[conditions[0]]
        time = np.asarray(reference["time"], dtype=float)
        if any(
            not np.allclose(time, condition_results[condition]["time"])
            for condition in conditions[1:]
        ):
            raise ValueError(f"PSTH time vectors differ across conditions for {group}.")

        paired_names = set(condition_results[conditions[0]]["mouse_names"])
        for condition in conditions[1:]:
            paired_names.intersection_update(condition_results[condition]["mouse_names"])
        paired_names = sorted(paired_names)
        use_difference_panel = len(conditions) == 2 and len(paired_names) > 0
        n_rows = 2 if use_difference_panel else 1
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(4.2, 5.0 if use_difference_panel else 3.1),
            sharex=True,
        )
        axes = np.atleast_1d(axes)
        top = axes[0]
        for index, condition in enumerate(conditions):
            result = condition_results[condition]
            color = colors(index % 10)
            top.plot(time, result["group_mean"], color=color, label=condition)
            if result["n_mice"] > 1:
                top.fill_between(
                    time,
                    result["group_mean"] - result["group_sem"],
                    result["group_mean"] + result["group_sem"],
                    color=color,
                    alpha=0.2,
                )
        top.axvline(0, color="black", linestyle="--", linewidth=0.8)
        top.axhline(0, color="black", linestyle=":", linewidth=0.8)
        signal_ylabel = _psth_ylabel(
            reference["normalization"], reference.get("signal_type", "photometry")
        )
        top.set(
            title=f"{group}: {_psth_description(reference)} by condition",
            ylabel=signal_ylabel,
        )
        top.legend(title="Condition")
        top.spines["top"].set_visible(False)
        top.spines["right"].set_visible(False)

        if use_difference_panel:
            first, second = conditions
            first_lookup = {
                mouse: trace
                for mouse, trace in zip(
                    condition_results[first]["mouse_names"],
                    condition_results[first]["mouse_matrix"],
                    strict=True,
                )
            }
            second_lookup = {
                mouse: trace
                for mouse, trace in zip(
                    condition_results[second]["mouse_names"],
                    condition_results[second]["mouse_matrix"],
                    strict=True,
                )
            }
            differences = np.vstack(
                [second_lookup[mouse] - first_lookup[mouse] for mouse in paired_names]
            )
            difference_mean, difference_sem = _mean_and_sem(differences)
            bottom = axes[1]
            for trace in differences:
                bottom.plot(time, trace, color="0.75", linewidth=0.7, alpha=0.8)
            bottom.plot(time, difference_mean, color="black", linewidth=2)
            if len(paired_names) > 1:
                bottom.fill_between(
                    time,
                    difference_mean - difference_sem,
                    difference_mean + difference_sem,
                    color="black",
                    alpha=0.2,
                )
            bottom.axvline(0, color="black", linestyle="--", linewidth=0.8)
            bottom.axhline(0, color="black", linestyle=":", linewidth=0.8)
            bottom.set(
                xlabel=f"Time from {reference['event_key']} (s)",
                ylabel=f"{second} − {first}\n{signal_ylabel}",
                title=f"Paired difference ({len(paired_names)} mice)",
            )
            bottom.spines["top"].set_visible(False)
            bottom.spines["right"].set_visible(False)
        else:
            top.set_xlabel(f"Time from {reference['event_key']} (s)")

        fig.tight_layout()
        condition_label = "_vs_".join(safe_label(value) for value in conditions)
        response_suffix = (
            "_licking"
            if reference.get("signal_type", "photometry") == "licking"
            else ""
        )
        paths = save_figure_formats(
            fig,
            output_dir
            / (
                f"{safe_label(group)}_{condition_label}_"
                f"{safe_label(reference['event_key'])}{response_suffix}_psth"
            ),
            formats=formats,
            dpi=dpi,
        )
        plt.close(fig)
        saved.extend(paths)
    return saved


def save_psth_heatmaps(
    results,
    output_dir,
    *,
    sort="event_order",
    sort_window=None,
    direction="auto",
    unmatched="bottom",
    cmap="coolwarm",
    formats=("svg", "png"),
    dpi=300,
    font_family="Arial",
):
    """Save session, mouse-level, and descriptive pooled-trial heatmaps."""
    import matplotlib.pyplot as plt

    from .heatmap_ordering import HEATMAP_SORT_LABELS, order_heatmap_trials
    from .publication_figures import configure_publication_style, save_figure_formats

    output_dir = Path(output_dir) / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style(font_family=font_family)
    time = np.asarray(results["time"], dtype=float)
    colorbar_label = results.get("ylabel") or _psth_ylabel(
        results["normalization"], results.get("signal_type", "photometry")
    )
    context = _psth_context(results)
    context_prefix = f"{context}: " if context else ""
    response_suffix = (
        "_licking" if results.get("signal_type", "photometry") == "licking" else ""
    )
    order_label = HEATMAP_SORT_LABELS[sort]
    saved = []

    def save_matrix(rows, base, title, ylabel, row_labels=None, figsize=(8, 5)):
        rows = np.asarray(rows, dtype=float)
        finite = rows[np.isfinite(rows)]
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
        if limit == 0:
            limit = 1.0
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(
            rows,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            extent=(time[0], time[-1], rows.shape[0] + 0.5, 0.5),
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(
            xlabel=f"Time from {results['event_key']} (s)",
            ylabel=ylabel,
            title=title,
        )
        if row_labels is not None:
            ax.set_yticks(np.arange(1, rows.shape[0] + 1), labels=row_labels)
        fig.colorbar(image, ax=ax, label=colorbar_label)
        fig.tight_layout()
        saved.extend(save_figure_formats(fig, base, formats=formats, dpi=dpi))
        plt.close(fig)

    session_orderings = []
    metadata_rows = []
    for session_index, session in enumerate(results["session_results"]):
        ordered = order_heatmap_trials(
            session,
            time,
            sort=sort,
            sort_window=sort_window,
            direction=direction,
            unmatched=unmatched,
        )
        session_orderings.append(ordered)
        sorted_position = {int(index): row for row, index in enumerate(ordered["order"], start=1)}
        for original_index, (alignment, value, matched_time) in enumerate(
            zip(
                session["alignment_times"],
                ordered["values"],
                ordered["matched_event_times"],
                strict=True,
            )
        ):
            metadata_rows.append(
                {
                    "mouse": session["mouse"],
                    "date": session["date"],
                    "run": int(session["run"]),
                    "session_index": session_index + 1,
                    "original_trial_index": original_index + 1,
                    "session_sorted_index": sorted_position.get(original_index, ""),
                    "pooled_sorted_index": "",
                    "alignment_time": float(alignment),
                    "sort_method": sort,
                    "sort_value": float(value) if np.isfinite(value) else "",
                    "matched_event_time": (
                        float(matched_time) if np.isfinite(matched_time) else ""
                    ),
                    "matched": bool(np.isfinite(value)),
                    "included": original_index in sorted_position,
                    "sort_window_start": (
                        ordered["sort_window"][0] if ordered["sort_window"] else ""
                    ),
                    "sort_window_end": (
                        ordered["sort_window"][1] if ordered["sort_window"] else ""
                    ),
                    "direction": ordered["direction"],
                    "unmatched_policy": unmatched,
                }
            )
        if ordered["trials"].shape[0]:
            save_matrix(
                ordered["trials"],
                output_dir
                / (
                    f"{session['mouse']}_{session['date']}_run{int(session['run']):03d}_"
                    f"{results['event_key']}{response_suffix}_heatmap"
                ),
                (
                    f"{session['mouse']} {session['date']} run {int(session['run'])}: "
                    f"{_psth_description(results)}\nOrdered by {order_label}"
                ),
                "Trial",
            )

    mouse_names = list(results["mouse_names"])
    mouse_values = []
    for mouse in mouse_names:
        values = np.concatenate(
            [
                ordering["values"]
                for session, ordering in zip(
                    results["session_results"], session_orderings, strict=True
                )
                if session["mouse"] == mouse
            ]
        )
        mouse_values.append(float(np.nanmedian(values)) if np.any(np.isfinite(values)) else np.nan)
    mouse_values = np.asarray(mouse_values, dtype=float)
    mouse_matrix = np.asarray(results["mouse_matrix"], dtype=float)
    if sort != "event_order":
        finite = np.flatnonzero(np.isfinite(mouse_values))
        missing = np.flatnonzero(~np.isfinite(mouse_values))
        mouse_order = finite[np.argsort(mouse_values[finite], kind="stable")]
        if session_orderings[0]["direction"] == "descending":
            mouse_order = mouse_order[::-1]
        if unmatched == "bottom":
            mouse_order = np.concatenate((mouse_order, missing))
        mouse_matrix = mouse_matrix[mouse_order]
        mouse_names = [mouse_names[index] for index in mouse_order]
    if mouse_matrix.shape[0] == 0:
        raise ValueError("No heatmap rows matched the selected behavioral ordering.")
    mouse_height = max(3.5, min(12.0, 2.0 + 0.3 * len(mouse_names)))
    save_matrix(
        mouse_matrix,
        output_dir
        / f"group_{results['event_key']}{response_suffix}_mouse_means_heatmap",
        (
            f"{context_prefix}{_psth_description(results)}\n"
            f"Mouse-level means ordered by {order_label} "
            "(one equally weighted row per mouse)"
        ),
        "Mouse",
        row_labels=mouse_names,
        figsize=(8, mouse_height),
    )

    pooled_trials = np.vstack([session["trials"] for session in results["session_results"]])
    pooled_values = np.concatenate([ordering["values"] for ordering in session_orderings])
    pooled_order = np.arange(len(pooled_values), dtype=int)
    if sort != "event_order":
        finite = np.flatnonzero(np.isfinite(pooled_values))
        missing = np.flatnonzero(~np.isfinite(pooled_values))
        pooled_order = finite[np.argsort(pooled_values[finite], kind="stable")]
        if session_orderings[0]["direction"] == "descending":
            pooled_order = pooled_order[::-1]
        if unmatched == "bottom":
            pooled_order = np.concatenate((pooled_order, missing))
        pooled_trials = pooled_trials[pooled_order]
    for sorted_index, original_index in enumerate(pooled_order, start=1):
        metadata_rows[int(original_index)]["pooled_sorted_index"] = sorted_index
    save_matrix(
        pooled_trials,
        output_dir / f"group_{results['event_key']}{response_suffix}_pooled_trials_heatmap",
        (
            f"{context_prefix}{_psth_description(results)}\n"
            f"All pooled trials ordered by {order_label} "
            "(descriptive; rows are not independent units)"
        ),
        "Pooled trial",
        figsize=(8, 7),
    )

    metadata_path = output_dir / "heatmap_trial_order.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)
    saved.append(metadata_path)
    return saved


def save_null_diagnostics(
    results,
    output_dir,
    *,
    formats=("svg", "png"),
    dpi=300,
    font_family="Arial",
    near_zero_ratio=0.01,
    extreme_z=20.0,
):
    """Save an auditable report describing null locations and normalization."""
    import matplotlib.pyplot as plt

    from .publication_figures import configure_publication_style, save_figure_formats

    if results.get("null_method") == "none":
        raise ValueError("Null diagnostics require a random-onset or circular-shift run.")
    output_dir = Path(output_dir) / "null_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style(font_family=font_family)
    summary_rows = []
    all_real_std = []
    all_null_std = []
    distance_names = (
        "alignment",
        "cue_onset",
        "solenoid_onset",
        "lick_times",
        "lick_bout_onset",
    )
    all_event_distances = defaultdict(list)
    all_trial_max_z = []
    amplitudes = {"none": [], "subtract": [], "zscore": []}

    def finite(values):
        values = np.asarray(values, dtype=float)
        return values[np.isfinite(values)]

    for session in results["session_results"]:
        diagnostics = session["null_diagnostics"]
        real_std = finite(diagnostics["real_baseline_std"])
        null_std = finite(diagnostics["null_baseline_std"])
        event_distances = {
            name: finite(diagnostics["null_event_distances"].get(name, []))
            for name in distance_names
        }
        distances = event_distances["alignment"]
        trial_max_z = finite(diagnostics["null_trial_max_abs_z"])
        positive_reference = real_std[real_std > 0]
        reference_sd = (
            float(np.median(positive_reference)) if positive_reference.size else np.nan
        )
        near_zero_threshold = (
            max(np.finfo(float).eps, near_zero_ratio * reference_sd)
            if np.isfinite(reference_sd)
            else np.finfo(float).eps
        )
        near_zero_fraction = (
            float(np.mean(null_std <= near_zero_threshold)) if null_std.size else np.nan
        )
        extreme_fraction = (
            float(np.mean(trial_max_z >= extreme_z)) if trial_max_z.size else np.nan
        )
        summary_row = {
                "mouse": session["mouse"],
                "date": session["date"],
                "run": int(session["run"]),
                "channel": int(session["channel"]),
                "n_real_events": int(session["n_events"]),
                "n_null_trials": int(null_std.size),
                "real_baseline_sd_median": (
                    float(np.median(real_std)) if real_std.size else np.nan
                ),
                "null_baseline_sd_median": (
                    float(np.median(null_std)) if null_std.size else np.nan
                ),
                "near_zero_sd_threshold": near_zero_threshold,
                "near_zero_null_fraction": near_zero_fraction,
                "extreme_z_threshold": extreme_z,
                "extreme_null_fraction": extreme_fraction,
                "null_distance_median_seconds": (
                    float(np.median(distances)) if distances.size else np.nan
                ),
                "null_distance_min_seconds": (
                    float(np.min(distances)) if distances.size else np.nan
                ),
                "warning_near_zero_baseline": bool(near_zero_fraction > 0.01),
                "warning_extreme_z": bool(extreme_fraction > 0.01),
            }
        for name, values in event_distances.items():
            summary_row[f"distance_to_{name}_median_seconds"] = (
                float(np.median(values)) if values.size else np.nan
            )
            summary_row[f"distance_to_{name}_min_seconds"] = (
                float(np.min(values)) if values.size else np.nan
            )
            all_event_distances[name].extend(values.tolist())
        summary_rows.append(summary_row)
        all_real_std.extend(real_std.tolist())
        all_null_std.extend(null_std.tolist())
        all_trial_max_z.extend(trial_max_z.tolist())
        for method in amplitudes:
            amplitudes[method].extend(
                finite(diagnostics["max_abs_shuffle_mean"][method]).tolist()
            )

    summary_path = output_dir / "null_diagnostics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "null_method": results["null_method"],
        "n_shuffles": int(results["n_shuffles"]),
        "random_seed": int(results["random_seed"]),
        "null_exclusion_seconds": float(results["null_exclusion"]),
        "event_key": results["event_key"],
        "normalization": results["normalization"],
        "near_zero_definition": (
            "null baseline SD <= 1% of that session's median positive real-trial "
            "baseline SD"
        ),
        "near_zero_ratio": near_zero_ratio,
        "extreme_z_threshold": extreme_z,
        "warning_fraction_threshold": 0.01,
    }
    metadata_path = output_dir / "null_diagnostics_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths = [summary_path, metadata_path]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    positive_real = np.asarray(all_real_std)[np.asarray(all_real_std) > 0]
    positive_null = np.asarray(all_null_std)[np.asarray(all_null_std) > 0]
    combined_positive = np.concatenate((positive_real, positive_null))
    bins = 40
    if combined_positive.size and np.min(combined_positive) < np.max(combined_positive):
        bins = np.geomspace(
            float(np.min(combined_positive)), float(np.max(combined_positive)), 41
        )
    if positive_real.size:
        ax.hist(positive_real, bins=bins, alpha=0.6, label="Real trials")
    if positive_null.size:
        ax.hist(positive_null, bins=bins, alpha=0.5, label="Null trials")
    ax.set(xlabel="Baseline SD", ylabel="Count", title="Baseline variability")
    if positive_real.size or positive_null.size:
        ax.set_xscale("log")
        ax.legend()
    fig.tight_layout()
    paths.extend(
        save_figure_formats(
            fig, output_dir / "baseline_sd_distribution", formats=formats, dpi=dpi
        )
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    finite_max_z = finite(all_trial_max_z)
    if finite_max_z.size:
        ax.hist(finite_max_z, bins=50)
    ax.axvline(extreme_z, color="red", linestyle="--", label=f"Warning = {extreme_z:g}")
    ax.set(xlabel="Maximum absolute trial z-score", ylabel="Count", title="Null extremes")
    ax.legend()
    fig.tight_layout()
    paths.extend(
        save_figure_formats(
            fig, output_dir / "maximum_z_by_null_trial", formats=formats, dpi=dpi
        )
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in distance_names:
        values = finite(all_event_distances[name])
        if values.size:
            ax.hist(
                values,
                bins=50,
                histtype="step",
                linewidth=1.5,
                label=name.replace("_", " "),
            )
    ax.set(
        xlabel="Distance to nearest real event (s)",
        ylabel="Count",
        title="Null-onset separation from recorded behavioral events",
    )
    if any(all_event_distances.values()):
        ax.legend()
    fig.tight_layout()
    paths.extend(
        save_figure_formats(
            fig, output_dir / "null_event_distance", formats=formats, dpi=dpi
        )
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [method for method in ("none", "subtract", "zscore") if amplitudes[method]]
    if labels:
        ax.boxplot([amplitudes[label] for label in labels], tick_labels=labels)
    ax.set(
        ylabel="Maximum absolute shuffled mean",
        title="Effect of trial normalization on null amplitude",
    )
    fig.tight_layout()
    paths.extend(
        save_figure_formats(
            fig, output_dir / "normalization_comparison", formats=formats, dpi=dpi
        )
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    shown = 0
    for session in results["session_results"]:
        diagnostics = session["null_diagnostics"]
        for _onset, trace in zip(
            diagnostics["example_onsets"], diagnostics["example_trials"], strict=True
        ):
            ax.plot(diagnostics["peri_time"], trace, alpha=0.4, linewidth=0.8)
            shown += 1
            if shown >= 24:
                break
        if shown >= 24:
            break
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel=f"Time from random {results['event_key']} (s)",
        ylabel=_psth_ylabel(results["normalization"], results["signal_type"]),
        title=f"Example null trials ({shown} shown)",
    )
    fig.tight_layout()
    paths.extend(
        save_figure_formats(
            fig, output_dir / "example_null_trials", formats=formats, dpi=dpi
        )
    )
    plt.close(fig)
    return {"paths": paths, "summary_rows": summary_rows}


def save_psth_figures(
    results,
    output_dir,
    *,
    figure_level="both",
    formats=("svg", "png"),
    dpi=300,
    font_family="Arial",
):
    """Save one figure per mouse and/or a mouse-level group mean ± SEM."""
    import matplotlib.pyplot as plt

    from .publication_figures import configure_publication_style, save_figure_formats

    if figure_level not in ("individual", "group", "both"):
        raise ValueError("figure_level must be 'individual', 'group', or 'both'.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style(font_family=font_family)
    time = results["time"]
    ylabel = _psth_ylabel(
        results["normalization"], results.get("signal_type", "photometry")
    )
    context = _psth_context(results)
    description = _psth_description(results)
    response_suffix = (
        "_licking" if results.get("signal_type", "photometry") == "licking" else ""
    )
    saved = []

    if figure_level in ("individual", "both"):
        for mouse in results["mouse_names"]:
            mouse_result = results["mouse_results"][mouse]
            fig, ax = plt.subplots(figsize=(8, 5))
            for trace in mouse_result["session_matrix"]:
                ax.plot(time, trace, color="0.7", linewidth=1)
            if results["null_method"] != "none":
                ax.fill_between(
                    time,
                    mouse_result["null_lower"],
                    mouse_result["null_upper"],
                    color="0.6",
                    alpha=0.25,
                    label="95% shuffled envelope",
                )
                ax.plot(
                    time,
                    mouse_result["null_mean"],
                    color="0.35",
                    linestyle="--",
                    label="Shuffled mean",
                )
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
                    f"{mouse}: "
                    f"{context + ' / ' if context else ''}"
                    f"{description} PSTH "
                    f"({mouse_result['n_sessions']} sessions)"
                ),
            )
            ax.legend()
            fig.tight_layout()
            paths = save_figure_formats(
                fig,
                output_dir / f"{mouse}_{results['event_key']}{response_suffix}_psth",
                formats=formats,
                dpi=dpi,
            )
            plt.close(fig)
            saved.extend(paths)

    if figure_level in ("group", "both"):
        fig, ax = plt.subplots(figsize=(9, 6))
        for mouse, trace in zip(
            results["mouse_names"], results["mouse_matrix"], strict=True
        ):
            ax.plot(time, trace, alpha=0.35, linewidth=1, label=mouse)
        if results["null_method"] != "none":
            ax.fill_between(
                time,
                results["group_null_lower"],
                results["group_null_upper"],
                color="0.6",
                alpha=0.3,
                label="95% shuffled envelope",
            )
            ax.plot(
                time,
                results["group_null_mean"],
                color="0.35",
                linestyle="--",
                linewidth=2,
                label="Shuffled mean",
            )
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
            title=(
                f"{context + ': ' if context else 'Group '}"
                f"{description} PSTH ({results['n_mice']} mice)"
            ),
        )
        ax.legend()
        fig.tight_layout()
        paths = save_figure_formats(
            fig,
            output_dir / f"group_{results['event_key']}{response_suffix}_psth",
            formats=formats,
            dpi=dpi,
        )
        plt.close(fig)
        saved.extend(paths)

    return saved
