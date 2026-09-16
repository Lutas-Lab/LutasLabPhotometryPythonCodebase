from collections import defaultdict
import hashlib
from itertools import combinations

import numpy as np

from .group_analysis import (
    extract_perievent_trials,
    generate_null_onsets,
    normalize_trials,
)
from .save_sessiondata import load_session
from .session_manifest import processed_session_path


AVAILABLE_METRICS = (
    "mean",
    "auc",
    "positive_auc",
    "negative_auc",
    "peak",
    "peak_latency",
    "trough",
    "trough_latency",
)


def _trapezoid(values, x, axis):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=x, axis=axis)
    return np.trapz(values, x=x, axis=axis)


def _validate_response_window(peri_time, response_window):
    if len(response_window) != 2 or response_window[0] >= response_window[1]:
        raise ValueError("response_window must contain increasing start and end values.")
    mask = (peri_time >= response_window[0]) & (peri_time <= response_window[1])
    if np.sum(mask) < 2:
        raise ValueError("response_window must contain at least two peri-event samples.")
    return mask


def _smooth_rows(values, samples):
    samples = min(samples, values.shape[1])
    if samples % 2 == 0:
        samples -= 1
    if samples <= 1:
        return values.copy()
    kernel = np.ones(samples, dtype=float)
    output = np.full_like(values, np.nan, dtype=float)
    for index, row in enumerate(values):
        finite = np.isfinite(row).astype(float)
        numerator = np.convolve(np.nan_to_num(row), kernel, mode="same")
        denominator = np.convolve(finite, kernel, mode="same")
        np.divide(numerator, denominator, out=output[index], where=denominator > 0)
    return output


def compute_response_metrics(
    peri_time,
    traces,
    *,
    response_window=(0.0, 2.0),
    peak_smoothing=0.1,
    metrics=AVAILABLE_METRICS,
):
    """Calculate response metrics for each row of a peri-event trace matrix."""
    peri_time = np.asarray(peri_time, dtype=float)
    traces = np.asarray(traces, dtype=float)
    if traces.ndim == 1:
        traces = traces[None, :]
    if traces.ndim != 2 or traces.shape[1] != len(peri_time):
        raise ValueError("traces must have one column per peri-event time sample.")
    unknown = set(metrics).difference(AVAILABLE_METRICS)
    if unknown:
        raise ValueError(f"Unknown response metrics: {sorted(unknown)}")
    if not np.isfinite(peak_smoothing) or peak_smoothing < 0:
        raise ValueError("peak_smoothing must be finite and nonnegative.")

    mask = _validate_response_window(peri_time, response_window)
    response_time = peri_time[mask]
    response = traces[:, mask]
    dt = float(np.median(np.diff(peri_time)))
    smooth_samples = max(1, int(np.rint(peak_smoothing / dt)))
    smoothed = _smooth_rows(traces, smooth_samples)[:, mask]

    output = {}
    if "mean" in metrics:
        output["mean"] = np.nanmean(response, axis=1)
    if "auc" in metrics:
        output["auc"] = _trapezoid(response, response_time, axis=1)
    if "positive_auc" in metrics:
        output["positive_auc"] = _trapezoid(
            np.clip(response, 0, None), x=response_time, axis=1
        )
    if "negative_auc" in metrics:
        output["negative_auc"] = _trapezoid(
            np.clip(response, None, 0), x=response_time, axis=1
        )

    need_peak = "peak" in metrics or "peak_latency" in metrics
    need_trough = "trough" in metrics or "trough_latency" in metrics
    peak_values = np.full(len(traces), np.nan)
    peak_times = np.full(len(traces), np.nan)
    trough_values = np.full(len(traces), np.nan)
    trough_times = np.full(len(traces), np.nan)
    for index, row in enumerate(smoothed):
        if not np.any(np.isfinite(row)):
            continue
        if need_peak:
            location = int(np.nanargmax(row))
            peak_values[index] = row[location]
            peak_times[index] = response_time[location]
        if need_trough:
            location = int(np.nanargmin(row))
            trough_values[index] = row[location]
            trough_times[index] = response_time[location]
    if "peak" in metrics:
        output["peak"] = peak_values
    if "peak_latency" in metrics:
        output["peak_latency"] = peak_times
    if "trough" in metrics:
        output["trough"] = trough_values
    if "trough_latency" in metrics:
        output["trough_latency"] = trough_times
    return output


def _session_rng(seed, info):
    identifier = f"{seed}|{info['mouse']}|{info['date']}|{int(info['run'])}"
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _null_mean_traces(
    signal_time,
    signal,
    event_times,
    *,
    window,
    dt,
    normalization,
    baseline,
    null_method,
    n_shuffles,
    null_exclusion,
    rng,
):
    low = float(signal_time[0]) - float(window[0])
    high = float(signal_time[-1]) - float(window[1])
    onset_sets = generate_null_onsets(
        event_times,
        (low, high),
        n_shuffles=n_shuffles,
        method=null_method,
        exclusion=null_exclusion,
        rng=rng,
    )
    means = []
    for onsets in onset_sets:
        peri_time, trials, _ = extract_perievent_trials(
            signal_time, signal, onsets, window=window, dt=dt
        )
        normalized = normalize_trials(
            peri_time, trials, normalization=normalization, baseline=baseline
        )
        means.append(np.nanmean(normalized, axis=0))
    return np.asarray(means, dtype=float)


def analyze_manifest_metrics(
    sessions,
    data_root,
    *,
    event_key="cue_onset",
    channel=1,
    window=(-5.0, 10.0),
    dt=0.02,
    normalization="zscore",
    baseline=(-5.0, 0.0),
    response_window=(0.0, 2.0),
    peak_smoothing=0.1,
    metrics=("mean", "auc", "peak", "peak_latency"),
    null_method="none",
    n_shuffles=500,
    random_seed=0,
    null_exclusion=0.0,
):
    """Extract hierarchical PSTH metrics and optional shuffled null statistics."""
    if null_method not in ("none", "random_onsets", "circular_shift"):
        raise ValueError("Invalid null_method.")
    mouse_groups = defaultdict(set)
    for info in sessions:
        mouse_groups[info["mouse"]].add(str(info.get("group", "all") or "all"))
    inconsistent = [mouse for mouse, groups in mouse_groups.items() if len(groups) > 1]
    if inconsistent:
        raise ValueError(
            "Each mouse must belong to one group; conflicting assignments for "
            f"{sorted(inconsistent)}."
        )
    signal_key = f"dff_ch{int(channel)}"
    time_key = f"photo_time_465_ch{int(channel)}"
    trial_rows = []
    session_rows = []
    null_sessions = []

    for info in sessions:
        path = processed_session_path(data_root, info)
        session = load_session(path)
        missing = {event_key, signal_key, time_key}.difference(session)
        if missing:
            raise ValueError(f"{path} is missing analysis keys: {sorted(missing)}")
        event_times = np.asarray(session[event_key], dtype=float)
        peri_time, trials, valid_indices = extract_perievent_trials(
            session[time_key],
            session[signal_key],
            event_times,
            window=window,
            dt=dt,
        )
        if len(trials) == 0:
            continue
        normalized = normalize_trials(
            peri_time, trials, normalization=normalization, baseline=baseline
        )
        trial_metrics = compute_response_metrics(
            peri_time,
            normalized,
            response_window=response_window,
            peak_smoothing=peak_smoothing,
            metrics=metrics,
        )
        group = str(info.get("group", "all") or "all")
        condition = str(info.get("condition", "all") or "all")
        base = {
            "mouse": info["mouse"],
            "date": info["date"],
            "run": info["run"],
            "group": group,
            "condition": condition,
            "event_key": event_key,
        }
        valid_event_times = event_times[valid_indices]
        for row_index, event_index in enumerate(valid_indices):
            for metric, values in trial_metrics.items():
                trial_rows.append(
                    {
                        **base,
                        "trial": int(row_index + 1),
                        "event_index": int(event_index),
                        "event_time": float(valid_event_times[row_index]),
                        "metric": metric,
                        "value": float(values[row_index]),
                    }
                )

        session_trace = np.nanmean(normalized, axis=0, keepdims=True)
        session_metrics = compute_response_metrics(
            peri_time,
            session_trace,
            response_window=response_window,
            peak_smoothing=peak_smoothing,
            metrics=metrics,
        )
        for metric, values in session_metrics.items():
            session_rows.append(
                {
                    **base,
                    "metric": metric,
                    "value": float(values[0]),
                    "n_trials": int(len(trials)),
                    "path": str(path),
                }
            )

        if null_method != "none":
            null_traces = _null_mean_traces(
                np.asarray(session[time_key], dtype=float),
                np.asarray(session[signal_key], dtype=float),
                valid_event_times,
                window=window,
                dt=dt,
                normalization=normalization,
                baseline=baseline,
                null_method=null_method,
                n_shuffles=n_shuffles,
                null_exclusion=null_exclusion,
                rng=_session_rng(random_seed, info),
            )
            null_metrics = compute_response_metrics(
                peri_time,
                null_traces,
                response_window=response_window,
                peak_smoothing=peak_smoothing,
                metrics=metrics,
            )
            for metric, values in null_metrics.items():
                null_sessions.append({**base, "metric": metric, "values": values})

    if not session_rows:
        raise ValueError("No sessions contained usable peri-event trials.")
    mouse_rows = summarize_mouse_metrics(session_rows)
    group_rows = summarize_group_metrics(mouse_rows)
    shuffle_rows = summarize_shuffle_tests(mouse_rows, null_sessions)
    return {
        "trial_rows": trial_rows,
        "session_rows": session_rows,
        "mouse_rows": mouse_rows,
        "group_rows": group_rows,
        "shuffle_rows": shuffle_rows,
    }


def summarize_mouse_metrics(session_rows):
    grouped = defaultdict(list)
    for row in session_rows:
        key = (row["mouse"], row["group"], row["condition"], row["metric"])
        grouped[key].append(row)
    output = []
    for (mouse, group, condition, metric), rows in sorted(grouped.items()):
        values = np.asarray([row["value"] for row in rows], dtype=float)
        output.append(
            {
                "mouse": mouse,
                "group": group,
                "condition": condition,
                "metric": metric,
                "value": float(np.nanmean(values)),
                "n_sessions": len(rows),
                "n_trials": int(sum(row["n_trials"] for row in rows)),
            }
        )
    return output


def summarize_group_metrics(mouse_rows):
    grouped = defaultdict(list)
    for row in mouse_rows:
        grouped[(row["group"], row["condition"], row["metric"])].append(row["value"])
    output = []
    for (group, condition, metric), raw_values in sorted(grouped.items()):
        values = np.asarray(raw_values, dtype=float)
        values = values[np.isfinite(values)]
        output.append(
            {
                "group": group,
                "condition": condition,
                "metric": metric,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "sem": (
                    float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else np.nan
                ),
                "n_mice": int(len(values)),
            }
        )
    return output


def _holm_adjust(rows, key="p_value"):
    valid = [(index, row[key]) for index, row in enumerate(rows) if np.isfinite(row[key])]
    valid.sort(key=lambda item: item[1])
    adjusted = np.full(len(rows), np.nan)
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    for index, row in enumerate(rows):
        row["p_adjusted_holm"] = float(adjusted[index])
    return rows


def summarize_shuffle_tests(mouse_rows, null_sessions):
    if not null_sessions:
        return []
    observed = {
        (row["mouse"], row["group"], row["condition"], row["metric"]): row["value"]
        for row in mouse_rows
    }
    mouse_null = defaultdict(list)
    for row in null_sessions:
        key = (row["mouse"], row["group"], row["condition"], row["metric"])
        mouse_null[key].append(np.asarray(row["values"], dtype=float))
    group_null = defaultdict(list)
    group_observed = defaultdict(list)
    for key, arrays in mouse_null.items():
        mouse, group, condition, metric = key
        group_key = (group, condition, metric)
        group_null[group_key].append(np.nanmean(np.vstack(arrays), axis=0))
        group_observed[group_key].append(observed[key])

    rows = []
    for (group, condition, metric), arrays in sorted(group_null.items()):
        null_values = np.nanmean(np.vstack(arrays), axis=0)
        observed_value = float(np.nanmean(group_observed[(group, condition, metric)]))
        center = float(np.nanmedian(null_values))
        distance = abs(observed_value - center)
        p_value = (1 + np.sum(np.abs(null_values - center) >= distance)) / (
            len(null_values) + 1
        )
        rows.append(
            {
                "group": group,
                "condition": condition,
                "metric": metric,
                "observed_mean": observed_value,
                "null_mean": float(np.nanmean(null_values)),
                "null_lower": float(np.nanpercentile(null_values, 2.5)),
                "null_upper": float(np.nanpercentile(null_values, 97.5)),
                "empirical_p_two_sided": float(p_value),
                "n_mice": len(arrays),
                "n_shuffles": len(null_values),
            }
        )
    return _holm_adjust(rows, key="empirical_p_two_sided")


def _paired_test(values_a, values_b, test):
    from scipy import stats

    differences = values_a - values_b
    if test == "wilcoxon":
        if np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
            effect = 0.0
        else:
            statistic, p_value = stats.wilcoxon(values_a, values_b)
            nonzero = differences[differences != 0]
            ranks = stats.rankdata(np.abs(nonzero))
            effect = float(
                (np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0]))
                / np.sum(ranks)
            )
        effect_name = "rank_biserial"
    else:
        statistic, p_value = stats.ttest_rel(values_a, values_b)
        std = np.std(differences, ddof=1)
        effect = np.mean(differences) / std if std > 0 else np.nan
        effect_name = "cohen_dz"
    mean_difference = float(np.mean(differences))
    if len(differences) > 1:
        standard_error = stats.sem(differences)
        margin = stats.t.ppf(0.975, len(differences) - 1) * standard_error
        ci = (mean_difference - margin, mean_difference + margin)
    else:
        ci = (np.nan, np.nan)
    return statistic, p_value, mean_difference, effect, effect_name, ci


def _independent_test(values_a, values_b, test):
    from scipy import stats

    if test == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        effect = 2 * statistic / (len(values_a) * len(values_b)) - 1
        effect_name = "rank_biserial"
    else:
        statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
    difference = float(np.mean(values_a) - np.mean(values_b))
    variance_a = np.var(values_a, ddof=1)
    variance_b = np.var(values_b, ddof=1)
    standard_error = np.sqrt(variance_a / len(values_a) + variance_b / len(values_b))
    numerator = (variance_a / len(values_a) + variance_b / len(values_b)) ** 2
    denominator = (variance_a / len(values_a)) ** 2 / (len(values_a) - 1)
    denominator += (variance_b / len(values_b)) ** 2 / (len(values_b) - 1)
    degrees = numerator / denominator if denominator > 0 else np.nan
    margin = stats.t.ppf(0.975, degrees) * standard_error
    pooled_numerator = (len(values_a) - 1) * variance_a + (len(values_b) - 1) * variance_b
    pooled = np.sqrt(pooled_numerator / (len(values_a) + len(values_b) - 2))
    d = difference / pooled if pooled > 0 else np.nan
    correction = 1 - 3 / (4 * (len(values_a) + len(values_b)) - 9)
    if test != "mannwhitney":
        effect = d * correction
        effect_name = "hedges_g"
    return (
        statistic,
        p_value,
        difference,
        effect,
        effect_name,
        (difference - margin, difference + margin),
    )


def run_pairwise_tests(mouse_rows, *, test="auto"):
    """Compare conditions within groups and groups within conditions at mouse level."""
    allowed = {"auto", "paired_t", "wilcoxon", "welch", "mannwhitney"}
    if test not in allowed:
        raise ValueError(f"test must be one of {sorted(allowed)}.")
    lookup = defaultdict(dict)
    for row in mouse_rows:
        lookup[(row["group"], row["condition"], row["metric"])][row["mouse"]] = row[
            "value"
        ]
    groups = sorted({row["group"] for row in mouse_rows})
    conditions = sorted({row["condition"] for row in mouse_rows})
    metrics = sorted({row["metric"] for row in mouse_rows})
    comparisons = []

    def compare(
        mapping_a,
        mapping_b,
        *,
        comparison,
        stratum,
        level_a,
        level_b,
        metric,
        paired_allowed,
    ):
        shared = sorted(set(mapping_a).intersection(mapping_b))
        if test == "auto" and paired_allowed and len(shared) == 1:
            return
        use_paired = paired_allowed and (
            test in ("paired_t", "wilcoxon")
            or (test == "auto" and len(shared) >= 2)
        )
        if use_paired:
            if len(shared) < 2:
                return
            values_a = np.asarray([mapping_a[mouse] for mouse in shared], dtype=float)
            values_b = np.asarray([mapping_b[mouse] for mouse in shared], dtype=float)
            selected_test = "wilcoxon" if test == "wilcoxon" else "paired_t"
            result = _paired_test(values_a, values_b, selected_test)
        else:
            values_a = np.asarray(list(mapping_a.values()), dtype=float)
            values_b = np.asarray(list(mapping_b.values()), dtype=float)
            if len(values_a) < 2 or len(values_b) < 2:
                return
            selected_test = "mannwhitney" if test == "mannwhitney" else "welch"
            result = _independent_test(values_a, values_b, selected_test)
        statistic, p_value, difference, effect, effect_name, ci = result
        comparisons.append(
            {
                "comparison": comparison,
                "stratum": stratum,
                "level_a": level_a,
                "level_b": level_b,
                "metric": metric,
                "test": selected_test,
                "n_a": len(values_a),
                "n_b": len(values_b),
                "n_pairs": len(shared) if use_paired else 0,
                "mean_a": float(np.mean(values_a)),
                "mean_b": float(np.mean(values_b)),
                "difference_a_minus_b": difference,
                "effect_size": float(effect),
                "effect_size_name": effect_name,
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

    for group in groups:
        for metric in metrics:
            for condition_a, condition_b in combinations(conditions, 2):
                mapping_a = lookup.get((group, condition_a, metric), {})
                mapping_b = lookup.get((group, condition_b, metric), {})
                if mapping_a and mapping_b:
                    compare(
                        mapping_a,
                        mapping_b,
                        comparison="condition_within_group",
                        stratum=group,
                        level_a=condition_a,
                        level_b=condition_b,
                        metric=metric,
                        paired_allowed=True,
                    )
    for condition in conditions:
        for metric in metrics:
            for group_a, group_b in combinations(groups, 2):
                mapping_a = lookup.get((group_a, condition, metric), {})
                mapping_b = lookup.get((group_b, condition, metric), {})
                if mapping_a and mapping_b:
                    forced_test = test
                    if forced_test in ("paired_t", "wilcoxon"):
                        continue
                    compare(
                        mapping_a,
                        mapping_b,
                        comparison="group_within_condition",
                        stratum=condition,
                        level_a=group_a,
                        level_b=group_b,
                        metric=metric,
                        paired_allowed=False,
                    )
    return _holm_adjust(comparisons)


def prism_wide_rows(mouse_rows):
    """Return one row per mouse with group/condition/metric columns for Prism."""
    rows = defaultdict(dict)
    for item in mouse_rows:
        rows[item["mouse"]]["mouse"] = item["mouse"]
        column = f"{item['group']}__{item['condition']}__{item['metric']}"
        rows[item["mouse"]][column] = item["value"]
    return [rows[mouse] for mouse in sorted(rows)]
