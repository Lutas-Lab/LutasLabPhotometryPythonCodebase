"""Portable PSTH timecourse exports and statistics from saved analysis runs."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .psth_statistics import (
    AVAILABLE_METRICS,
    compute_response_metrics,
    prism_wide_rows,
    run_pairwise_tests,
)
from .publication_figures import save_metric_figures
from .update_check import local_git_commit


def _code_commit():
    return local_git_commit(Path(__file__).resolve().parents[4])


def _mean_sem(matrix):
    matrix = np.asarray(matrix, dtype=float)
    count = np.sum(np.isfinite(matrix), axis=0)
    mean = np.divide(
        np.nansum(matrix, axis=0),
        count,
        out=np.full(matrix.shape[1], np.nan),
        where=count > 0,
    )
    sem = np.full(matrix.shape[1], np.nan, dtype=float)
    usable = count > 1
    if np.any(usable):
        sem[usable] = np.nanstd(matrix[:, usable], axis=0, ddof=1) / np.sqrt(
            count[usable]
        )
    return mean, sem, count


def _write_rows(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
        extras = {key for row in rows for key in row}.difference(fieldnames)
        fieldnames.extend(sorted(extras))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                key: ""
                if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                else value
                for key, value in row.items()
            }
            for row in rows
        )
    return path


def _safe_label(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    return text or "all"


def _saved_float(value, *, field, path, allow_blank=False):
    """Parse one saved numeric CSV field, preserving intentional missing values."""

    text = "" if value is None else str(value).strip()
    if not text:
        if allow_blank:
            return float("nan")
        raise ValueError(f"{path} contains a blank {field} value")
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"{path} contains an invalid {field} value: {text!r}") from error


def _mean_trace(matrix):
    return _mean_sem(np.asarray(matrix, dtype=float))[0]


def _result_identity(results):
    signal_type = str(results.get("signal_type", "photometry"))
    signal = str(
        results.get("signal")
        or results.get("photometry_signal")
        or results.get("signal_key")
        or signal_type
    )
    normalization = str(results.get("normalization", "none"))
    units = str(results.get("units") or results.get("ylabel") or "")
    if not units:
        if signal_type == "licking":
            units = "licks/s"
        elif signal_type == "running":
            units = "cm/s"
        elif normalization == "zscore":
            units = "z-score"
        elif signal == "dff":
            units = "ΔF/F"
        else:
            units = "a.u."
    return {
        "workflow": str(results.get("workflow", "conventional")),
        "signal": signal,
        "signal_type": signal_type,
        "event": str(results.get("event_key", "event")),
        "normalization": normalization,
        "units": units,
    }


def _session_trace_rows(results):
    time = np.asarray(results["time"], dtype=float)
    identity = _result_identity(results)
    records = []
    traces = []
    for index, session in enumerate(results["session_results"], start=1):
        trials = np.asarray(session["trials"], dtype=float)
        if trials.ndim != 2 or trials.shape[1] != time.size:
            raise ValueError("Each PSTH session must contain trials on the shared time grid")
        trace = _mean_trace(trials)
        record = {
            **identity,
            "mouse": str(session.get("mouse", "")),
            "date": str(session.get("date", "")),
            "run": int(session.get("run", index)),
            "group": str(session.get("group") or "all"),
            "condition": str(session.get("condition") or "all"),
            "session_id": str(
                session.get("session_id")
                or (
                    f"{session.get('mouse', '')}_{session.get('date', '')}_"
                    f"run{int(session.get('run', index)):03d}"
                )
            ),
            "n_trials": int(trials.shape[0]),
        }
        records.append(record)
        traces.append(trace)
    if not traces:
        raise ValueError("No session PSTH traces were available to export")
    return time, records, np.vstack(traces)


def _aggregate_mouse_traces(records, session_matrix):
    grouped = defaultdict(list)
    for record, trace in zip(records, session_matrix, strict=True):
        key = (record["mouse"], record["group"], record["condition"])
        grouped[key].append(trace)
    mouse_records = []
    traces = []
    for (mouse, group, condition), rows in sorted(grouped.items()):
        mouse_records.append(
            {
                "mouse": mouse,
                "group": group,
                "condition": condition,
                "n_sessions": len(rows),
            }
        )
        traces.append(_mean_trace(np.vstack(rows)))
    return mouse_records, np.vstack(traces)


def _aggregate_group_traces(mouse_records, mouse_matrix):
    grouped = defaultdict(list)
    for record, trace in zip(mouse_records, mouse_matrix, strict=True):
        grouped[(record["group"], record["condition"])].append(trace)
    group_records = []
    means = []
    sems = []
    for (group, condition), rows in sorted(grouped.items()):
        mean, sem, _ = _mean_sem(np.vstack(rows))
        group_records.append(
            {"group": group, "condition": condition, "n_mice": len(rows)}
        )
        means.append(mean)
        sems.append(sem)
    return group_records, np.vstack(means), np.vstack(sems)


def save_psth_timecourse_tables(results, output_dir):
    """Save portable session, mouse, group, Prism, and compressed PSTH values."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    time, session_records, session_matrix = _session_trace_rows(results)
    mouse_records, mouse_matrix = _aggregate_mouse_traces(
        session_records, session_matrix
    )
    group_records, group_mean, group_sem = _aggregate_group_traces(
        mouse_records, mouse_matrix
    )
    identity = _result_identity(results)
    session_rows = []
    for record, trace in zip(session_records, session_matrix, strict=True):
        for time_s, value in zip(time, trace, strict=True):
            session_rows.append({**record, "time_s": time_s, "value": value})
    mouse_rows = []
    for record, trace in zip(mouse_records, mouse_matrix, strict=True):
        for time_s, value in zip(time, trace, strict=True):
            mouse_rows.append({**identity, **record, "time_s": time_s, "value": value})
    group_rows = []
    for record, mean, sem in zip(
        group_records, group_mean, group_sem, strict=True
    ):
        for time_s, mean_value, sem_value in zip(time, mean, sem, strict=True):
            group_rows.append(
                {
                    **identity,
                    **record,
                    "time_s": time_s,
                    "mean": mean_value,
                    "sem": sem_value,
                }
            )
    outputs = [
        _write_rows(output_dir / "psth_session_timecourses.csv", session_rows),
        _write_rows(output_dir / "psth_mouse_timecourses.csv", mouse_rows),
        _write_rows(output_dir / "psth_group_timecourse.csv", group_rows),
    ]
    prism_columns = [
        f"{record['group']}__{record['condition']}__{record['mouse']}"
        for record in mouse_records
    ]
    prism_rows = []
    for time_index, time_s in enumerate(time):
        row = {"time_s": time_s}
        row.update(
            {
                column: mouse_matrix[index, time_index]
                for index, column in enumerate(prism_columns)
            }
        )
        prism_rows.append(row)
    outputs.append(
        _write_rows(
            output_dir / "psth_prism_timecourses.csv",
            prism_rows,
            ["time_s", *prism_columns],
        )
    )
    npz_path = output_dir / "psth_timecourses.npz"
    np.savez_compressed(
        npz_path,
        time=time,
        session_matrix=session_matrix,
        session_labels=np.asarray(
            [record["session_id"] for record in session_records], dtype=str
        ),
        mouse_matrix=mouse_matrix,
        mouse_labels=np.asarray(
            [
                f"{record['group']}__{record['condition']}__{record['mouse']}"
                for record in mouse_records
            ],
            dtype=str,
        ),
        group_mean=group_mean,
        group_sem=group_sem,
        group_labels=np.asarray(
            [f"{record['group']}__{record['condition']}" for record in group_records],
            dtype=str,
        ),
        **{key: np.asarray(value) for key, value in identity.items()},
    )
    outputs.append(npz_path)
    metadata_path = output_dir / "psth_timecourse_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                **identity,
                "generated_utc": datetime.now(UTC).isoformat(),
                "code_commit": _code_commit(),
                "hierarchy": "trials averaged within sessions; sessions averaged within mice",
                "session_count": len(session_records),
                "mouse_count": len(mouse_records),
                "group_condition_count": len(group_records),
                "time_samples": int(time.size),
                "files": [path.name for path in outputs if path is not None],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs.append(metadata_path)
    return [path for path in outputs if path is not None]


def load_saved_psth_timecourses(input_dir):
    """Load and combine compatible saved session-level PSTH CSV exports."""

    input_path = Path(input_dir).expanduser()
    paths = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.rglob("psth_session_timecourses.csv"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No psth_session_timecourses.csv files were found below {input_path}"
        )
    sessions = {}
    identity = None
    reference_time = None
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        current_identity = {
            key: rows[0].get(key, "")
            for key in ("workflow", "signal", "signal_type", "event", "normalization", "units")
        }
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(
                "The selected folder contains incompatible PSTH signals or settings. "
                "Select one PSTH run or a folder containing only matching runs."
            )
        grouped = defaultdict(list)
        for row in rows:
            key = (
                row["mouse"],
                row["date"],
                int(row["run"]),
                row["group"],
                row["condition"],
                row["session_id"],
            )
            grouped[key].append(row)
        for key, session_rows in grouped.items():
            session_rows.sort(
                key=lambda row: _saved_float(
                    row.get("time_s"), field="time_s", path=path
                )
            )
            time = np.asarray(
                [
                    _saved_float(row.get("time_s"), field="time_s", path=path)
                    for row in session_rows
                ]
            )
            trace = np.asarray(
                [
                    _saved_float(
                        row.get("value"), field="value", path=path, allow_blank=True
                    )
                    for row in session_rows
                ]
            )
            if reference_time is None:
                reference_time = time
            elif time.shape != reference_time.shape or not np.allclose(
                time, reference_time, rtol=1e-8, atol=1e-10
            ):
                raise ValueError("Saved PSTH runs use different time grids")
            if key in sessions:
                if not np.allclose(sessions[key], trace, equal_nan=True):
                    raise ValueError(f"Conflicting duplicate session export: {key[-1]}")
                continue
            sessions[key] = trace
    if not sessions or identity is None or reference_time is None:
        raise ValueError("The saved PSTH exports contained no timecourses")
    session_records = [
        {
            "mouse": key[0],
            "date": key[1],
            "run": key[2],
            "group": key[3],
            "condition": key[4],
            "session_id": key[5],
        }
        for key in sorted(sessions)
    ]
    session_matrix = np.vstack([sessions[key] for key in sorted(sessions)])
    mouse_records, mouse_matrix = _aggregate_mouse_traces(
        session_records, session_matrix
    )
    return {
        "time": reference_time,
        "identity": identity,
        "session_records": session_records,
        "session_matrix": session_matrix,
        "mouse_records": mouse_records,
        "mouse_matrix": mouse_matrix,
        "source_files": [str(path) for path in paths],
    }


def _metric_units(metric, signal_units):
    if metric in {"peak_latency", "trough_latency"}:
        return "s"
    if metric in {"auc", "positive_auc", "negative_auc"}:
        return f"{signal_units}·s" if signal_units else "value·s"
    return signal_units or "value"


def _metric_rows(records, metric_values, identity, level):
    rows = []
    for metric, values in metric_values.items():
        for record, value in zip(records, values, strict=True):
            rows.append(
                {
                    **identity,
                    **record,
                    "level": level,
                    "metric": metric,
                    "metric_units": _metric_units(metric, identity.get("units", "")),
                    "value": float(value),
                }
            )
    return rows


def save_saved_psth_statistics(
    input_dir,
    output_dir,
    *,
    response_window=(0.0, 2.0),
    peak_smoothing=0.1,
    metrics=("mean", "auc", "peak", "peak_latency"),
    test="auto",
    bootstrap_resamples=5000,
    bootstrap_seed=0,
    formats=("svg", "png"),
    dpi=300,
    font_family="Arial",
    progress_callback=None,
):
    """Derive hierarchical metrics and Prism tables from saved PSTH exports."""

    unknown = set(metrics).difference(AVAILABLE_METRICS)
    if unknown:
        raise ValueError(f"Unknown response metrics: {sorted(unknown)}")
    if progress_callback is not None:
        progress_callback(0, 4, "Loading saved PSTH timecourses")
    bundle = load_saved_psth_timecourses(input_dir)
    session_metrics = compute_response_metrics(
        bundle["time"],
        bundle["session_matrix"],
        response_window=response_window,
        peak_smoothing=peak_smoothing,
        metrics=metrics,
    )
    mouse_metrics = compute_response_metrics(
        bundle["time"],
        bundle["mouse_matrix"],
        response_window=response_window,
        peak_smoothing=peak_smoothing,
        metrics=metrics,
    )
    identity = bundle["identity"]
    session_rows = _metric_rows(
        bundle["session_records"], session_metrics, identity, "session"
    )
    mouse_rows = _metric_rows(bundle["mouse_records"], mouse_metrics, identity, "mouse")
    if progress_callback is not None:
        progress_callback(1, 4, "Computed session- and mouse-level metrics")
    group_rows = []
    inference_mouse_rows = [row for row in mouse_rows if np.isfinite(row["value"])]
    grouped = defaultdict(list)
    for row in inference_mouse_rows:
        grouped[(row["group"], row["condition"], row["metric"], row["metric_units"])].append(
            row["value"]
        )
    for (group, condition, metric, units), values in sorted(grouped.items()):
        values = np.asarray(values, dtype=float)
        group_rows.append(
            {
                **identity,
                "group": group,
                "condition": condition,
                "metric": metric,
                "metric_units": units,
                "mean": float(np.nanmean(values)),
                "sem": (
                    float(np.nanstd(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else np.nan
                ),
                "n_mice": len(values),
            }
        )
    test_rows = run_pairwise_tests(
        inference_mouse_rows,
        test=test,
        n_bootstrap=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    if progress_callback is not None:
        progress_callback(2, 4, "Computed mouse-level comparisons")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_mouse_timecourse_rows = []
    for record, trace in zip(
        bundle["mouse_records"], bundle["mouse_matrix"], strict=True
    ):
        for time_s, value in zip(bundle["time"], trace, strict=True):
            combined_mouse_timecourse_rows.append(
                {**identity, **record, "time_s": time_s, "value": value}
            )
    prism_timecourse_columns = [
        f"{record['group']}__{record['condition']}__{record['mouse']}"
        for record in bundle["mouse_records"]
    ]
    prism_timecourse_rows = []
    for time_index, time_s in enumerate(bundle["time"]):
        row = {"time_s": time_s}
        row.update(
            {
                column: bundle["mouse_matrix"][index, time_index]
                for index, column in enumerate(prism_timecourse_columns)
            }
        )
        prism_timecourse_rows.append(row)
    outputs = [
        _write_rows(
            output_dir / "psth_mouse_timecourses.csv",
            combined_mouse_timecourse_rows,
        ),
        _write_rows(
            output_dir / "psth_prism_timecourses.csv",
            prism_timecourse_rows,
            ["time_s", *prism_timecourse_columns],
        ),
        _write_rows(output_dir / "psth_session_metrics.csv", session_rows),
        _write_rows(output_dir / "psth_mouse_metrics.csv", mouse_rows),
        _write_rows(output_dir / "psth_group_summary.csv", group_rows),
        _write_rows(output_dir / "psth_statistical_tests.csv", test_rows),
        _write_rows(output_dir / "psth_prism_wide.csv", prism_wide_rows(mouse_rows)),
    ]
    for metric in metrics:
        selected = [row for row in mouse_rows if row["metric"] == metric]
        columns = sorted({f"{row['group']}__{row['condition']}" for row in selected})
        by_mouse = defaultdict(dict)
        for row in selected:
            by_mouse[row["mouse"]]["mouse"] = row["mouse"]
            by_mouse[row["mouse"]][f"{row['group']}__{row['condition']}"] = row["value"]
        outputs.append(
            _write_rows(
                output_dir / f"prism_{_safe_label(metric)}.csv",
                [by_mouse[mouse] for mouse in sorted(by_mouse)],
                ["mouse", *columns],
            )
        )
    figure_paths = save_metric_figures(
        inference_mouse_rows,
        test_rows,
        output_dir,
        formats=formats,
        dpi=dpi,
        font_family=font_family,
    )
    outputs.extend(figure_paths)
    if progress_callback is not None:
        progress_callback(3, 4, "Saved Prism tables and metric figures")
    metadata_path = output_dir / "psth_statistics_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                **identity,
                "input_directory": str(Path(input_dir).resolve()),
                "source_files": bundle["source_files"],
                "response_window": list(response_window),
                "peak_smoothing": peak_smoothing,
                "metrics": list(metrics),
                "test": test,
                "bootstrap_resamples": bootstrap_resamples,
                "bootstrap_seed": bootstrap_seed,
                "hierarchy": "sessions averaged within mice; mice are the inferential N",
                "generated_utc": datetime.now(UTC).isoformat(),
                "code_commit": _code_commit(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs.append(metadata_path)
    if progress_callback is not None:
        progress_callback(4, 4, "PSTH summary statistics complete")
    return [path for path in outputs if path is not None]
