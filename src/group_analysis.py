from collections import defaultdict
import hashlib
from pathlib import Path
import warnings

import numpy as np

from .save_sessiondata import load_session
from .session_manifest import processed_session_path, resolve_session_channel


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
    event_times = np.asarray(event_times, dtype=float)
    event_times = event_times[np.isfinite(event_times)]
    if event_times.ndim != 1 or len(event_times) == 0:
        raise ValueError("event_times must contain at least one finite event.")
    if len(onset_bounds) != 2 or onset_bounds[0] >= onset_bounds[1]:
        raise ValueError("onset_bounds must contain increasing start and end values.")
    if not isinstance(n_shuffles, int) or isinstance(n_shuffles, bool) or n_shuffles < 1:
        raise ValueError("n_shuffles must be a positive integer.")
    if method not in ("random_onsets", "circular_shift"):
        raise ValueError("method must be 'random_onsets' or 'circular_shift'.")
    if not np.isfinite(exclusion) or exclusion < 0:
        raise ValueError("exclusion must be finite and nonnegative.")

    low, high = map(float, onset_bounds)
    rng = np.random.default_rng() if rng is None else rng
    output = np.empty((n_shuffles, len(event_times)), dtype=float)

    def sufficiently_far(candidates):
        if exclusion == 0:
            return np.ones(len(candidates), dtype=bool)
        distances = np.abs(candidates[:, None] - event_times[None, :])
        return np.all(distances >= exclusion, axis=1)

    if method == "random_onsets":
        for shuffle_index in range(n_shuffles):
            selected = []
            for _ in range(1000):
                candidates = rng.uniform(low, high, size=max(64, 2 * len(event_times)))
                selected.extend(candidates[sufficiently_far(candidates)].tolist())
                if len(selected) >= len(event_times):
                    break
            if len(selected) < len(event_times):
                raise ValueError(
                    "Could not sample enough random onsets. Reduce null exclusion."
                )
            output[shuffle_index] = selected[: len(event_times)]
        return output

    span = high - low
    wrapped_events = low + np.mod(event_times - low, span)
    for shuffle_index in range(n_shuffles):
        for _ in range(10000):
            shift = rng.uniform(0.0, span)
            candidates = low + np.mod(wrapped_events - low + shift, span)
            if np.all(sufficiently_far(candidates)):
                output[shuffle_index] = candidates
                break
        else:
            raise ValueError(
                "Could not find an eligible circular shift. Reduce null exclusion."
            )
    return output


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
    for onsets in shuffled_onsets:
        peri_time, trials, _ = extract_perievent_trials(
            signal_time, signal, onsets, window=window, dt=dt
        )
        normalized = normalize_trials(
            peri_time, trials, normalization=normalization, baseline=baseline
        )
        means.append(np.nanmean(normalized, axis=0))
    return np.asarray(means, dtype=float)


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


def _psth_ylabel(normalization):
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


def compute_manifest_psth(
    sessions,
    data_root,
    *,
    event_key="cue_onset",
    channel="manifest",
    window=(-5, 10),
    dt=0.02,
    normalization="zscore",
    baseline=(-5, 0),
    null_method="none",
    n_shuffles=500,
    random_seed=0,
    null_exclusion=0.0,
):
    """Compute session, mouse, and group PSTHs with mice as the group unit."""
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

    session_results = []

    for info in sessions:
        selected_channel = resolve_session_channel(info, channel)
        signal_key = f"dff_ch{selected_channel}"
        time_key = f"photo_time_465_ch{selected_channel}"
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
        result = {
            **info,
            "path": path,
            "channel": selected_channel,
            "signal_key": signal_key,
            "n_events": len(valid_indices),
            "mean": session_mean,
        }
        if null_method != "none":
            valid_event_times = np.asarray(session[event_key], dtype=float)[valid_indices]
            result["null_means"] = _null_session_means(
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
        signal_ylabel = _psth_ylabel(reference["normalization"])
        top.set(title=f"{group}: condition PSTHs", ylabel=signal_ylabel)
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
                )
            }
            second_lookup = {
                mouse: trace
                for mouse, trace in zip(
                    condition_results[second]["mouse_names"],
                    condition_results[second]["mouse_matrix"],
                )
            }
            differences = np.vstack(
                [second_lookup[mouse] - first_lookup[mouse] for mouse in paired_names]
            )
            difference_mean, difference_sem = _mean_and_sem(differences)
            bottom = axes[1]
            for mouse, trace in zip(paired_names, differences):
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
        paths = save_figure_formats(
            fig,
            output_dir
            / (
                f"{safe_label(group)}_{condition_label}_"
                f"{safe_label(reference['event_key'])}_psth"
            ),
            formats=formats,
            dpi=dpi,
        )
        plt.close(fig)
        saved.extend(paths)
    return saved


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
    ylabel = _psth_ylabel(results["normalization"])
    context = _psth_context(results)
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
                    f"{results['event_key']} PSTH "
                    f"({mouse_result['n_sessions']} sessions)"
                ),
            )
            ax.legend()
            fig.tight_layout()
            paths = save_figure_formats(
                fig,
                output_dir / f"{mouse}_{results['event_key']}_psth",
                formats=formats,
                dpi=dpi,
            )
            plt.close(fig)
            saved.extend(paths)

    if figure_level in ("group", "both"):
        fig, ax = plt.subplots(figsize=(9, 6))
        for mouse, trace in zip(results["mouse_names"], results["mouse_matrix"]):
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
                f"{results['event_key']} PSTH ({results['n_mice']} mice)"
            ),
        )
        ax.legend()
        fig.tight_layout()
        paths = save_figure_formats(
            fig,
            output_dir / f"group_{results['event_key']}_psth",
            formats=formats,
            dpi=dpi,
        )
        plt.close(fig)
        saved.extend(paths)

    return saved
