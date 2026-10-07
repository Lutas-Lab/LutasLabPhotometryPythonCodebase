"""Shared batch workflows for FluoPulse and iFLiP3 GUI/CLI analyses."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from lutaslab_core.glm import event_times_to_counts, fit_ridge, group_folds, r2_score
from lutaslab_core.perievent import (
    extract_perievent_trials,
    normalize_trials,
    summarize_trials,
)

from .group_analysis import save_psth_heatmaps

LIFETIME_COMMON_COLUMNS = ("mouse", "date", "run", "group", "condition")
LIFETIME_PATH_COLUMNS = {
    "fluopulse": ("doric_path", "nidaq_path", "running_path"),
    "iflip3": ("iflip_path", "nidaq_path", "running_path", "background_path"),
}
LIFETIME_SIGNALS = {
    "fluopulse": ("tau", "amplitude", "fit_r_square"),
    "iflip3": ("mpet", "raw_intensity"),
}
LIFETIME_EVENTS = ("ensure", "visual_cue", "licks")


def lifetime_manifest_columns(workflow: str) -> tuple[str, ...]:
    """Return the manifest columns for one lifetime sensor."""

    if workflow not in LIFETIME_PATH_COLUMNS:
        raise ValueError("workflow must be 'fluopulse' or 'iflip3'")
    return LIFETIME_COMMON_COLUMNS + LIFETIME_PATH_COLUMNS[workflow]


def normalize_lifetime_manifest_rows(workflow: str, rows) -> list[dict]:
    """Validate lifetime manifest rows without requiring files to be online."""

    columns = lifetime_manifest_columns(workflow)
    normalized = []
    seen = set()
    for index, raw in enumerate(rows, start=1):
        row = {column: raw.get(column, "") for column in columns}
        if not any(str(value or "").strip() for value in row.values()):
            continue
        mouse = str(row["mouse"] or "").strip()
        date = str(row["date"] or "").strip()
        run_text = str(row["run"] or "").strip()
        if not mouse or not date or not run_text:
            raise ValueError(f"Row {index} has an empty mouse, date, or run value.")
        if len(date) != 6 or not date.isdigit():
            raise ValueError(f"Row {index} date must contain six digits (YYMMDD).")
        try:
            run = int(run_text)
        except ValueError as error:
            raise ValueError(f"Row {index} run must be an integer.") from error
        if run < 0:
            raise ValueError(f"Row {index} run cannot be negative.")
        identity = (mouse.casefold(), date, run)
        if identity in seen:
            raise ValueError(f"Row {index} duplicates {mouse} {date} run {run}.")
        seen.add(identity)
        clean = {
            "mouse": mouse,
            "date": date,
            "run": run,
            "group": str(row["group"] or "").strip(),
            "condition": str(row["condition"] or "").strip(),
        }
        clean.update(
            {column: str(row[column] or "").strip() for column in LIFETIME_PATH_COLUMNS[workflow]}
        )
        if workflow == "iflip3" and not clean["background_path"]:
            raise ValueError(
                f"Row {index} needs a matched background_path for iFLiP3 analysis."
            )
        normalized.append(clean)
    if not normalized:
        raise ValueError("Add at least one session to the manifest.")
    return normalized


def write_lifetime_manifest(workflow: str, path: str | Path, rows) -> Path:
    """Atomically write a validated lifetime manifest."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    validated = normalize_lifetime_manifest_rows(workflow, rows)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=lifetime_manifest_columns(workflow)
            )
            writer.writeheader()
            writer.writerows(validated)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def read_lifetime_manifest(workflow: str, path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return normalize_lifetime_manifest_rows(workflow, csv.DictReader(stream))


def _optional_path(value: str, data_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else data_root / path


def _load_fluopulse(row: dict, data_root: Path):
    from fluopulse_analysis import (
        find_doric_files,
        infer_session_identity,
        nidaq_paths,
        process_aligned_session,
    )

    doric = _optional_path(row["doric_path"], data_root)
    if doric is None:
        candidates = find_doric_files(
            row["mouse"], row["date"], doric_root=data_root / "FLIM FLIP"
        )
        candidates = [
            path
            for path in candidates
            if infer_session_identity(path).run == int(row["run"])
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one Doric file for {row['mouse']} {row['date']} run "
                f"{row['run']}; found {len(candidates)}. Enter doric_path explicitly."
            )
        doric = candidates[0]
    inferred = nidaq_paths(
        row["mouse"],
        row["date"],
        row["run"],
        photometry_root=data_root / "Photometry",
    )
    nidaq = _optional_path(row["nidaq_path"], data_root) or inferred.nidaq
    running = _optional_path(row["running_path"], data_root)
    if running is None and inferred.running.is_file():
        running = inferred.running
    session = process_aligned_session(doric, nidaq, running_path=running)
    return session.to_core_session(_session_id(row)), {
        "source_path": str(doric),
        "nidaq_path": str(nidaq),
        "running_path": "" if running is None else str(running),
    }


def _load_iflip3(row: dict, data_root: Path):
    from iflip3 import process_aligned_session, session_paths

    inferred = session_paths(row["mouse"], row["date"], row["run"], data_root=data_root)
    iflip = _optional_path(row["iflip_path"], data_root) or inferred.iflip
    nidaq = _optional_path(row["nidaq_path"], data_root) or inferred.nidaq
    running = _optional_path(row["running_path"], data_root)
    if running is None and inferred.running.is_file():
        running = inferred.running
    background = _optional_path(row["background_path"], data_root)
    if background is None:  # guarded by manifest validation
        raise ValueError("background_path is required")
    session = process_aligned_session(
        iflip,
        nidaq,
        background,
        running_path=running,
    )
    return session.to_core_session(_session_id(row)), {
        "source_path": str(iflip),
        "nidaq_path": str(nidaq),
        "running_path": "" if running is None else str(running),
        "background_path": str(background),
    }


def _session_id(row: dict) -> str:
    return f"{row['mouse']}_{row['date']}_run{int(row['run']):03d}"


def load_lifetime_session(workflow: str, row: dict, data_root: str | Path):
    """Load and align one lifetime session into the shared core representation."""

    root = Path(data_root).expanduser()
    if workflow == "fluopulse":
        return _load_fluopulse(row, root)
    if workflow == "iflip3":
        return _load_iflip3(row, root)
    raise ValueError("workflow must be 'fluopulse' or 'iflip3'")


def export_aligned_sessions(workflow, rows, data_root, output_dir) -> list[Path]:
    """Export aligned continuous/event arrays and provenance as compressed NPZ files."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    summary = []
    for row in rows:
        session, paths = load_lifetime_session(workflow, row, data_root)
        arrays = {}
        for name, signal in session.continuous.items():
            arrays[f"continuous_{name}_time"] = signal.timestamps
            arrays[f"continuous_{name}_values"] = signal.values
        for name, events in session.events.items():
            arrays[f"events_{name}"] = events.timestamps
        output = destination / f"{session.session_id}_aligned.npz"
        np.savez_compressed(output, **arrays)
        outputs.append(output)
        summary.append({**row, **paths, "output_path": str(output), **session.metadata})
    summary_path = destination / "aligned_sessions.csv"
    _write_dict_rows(summary_path, summary)
    outputs.append(summary_path)
    return outputs


def _behavioral_events(session) -> dict[str, np.ndarray]:
    return {
        "solenoid_onset": session.events.get("ensure", _EMPTY_EVENT).timestamps,
        "cue_onset": session.events.get("visual_cue", _EMPTY_EVENT).timestamps,
        "lick_times": session.events.get("licks", _EMPTY_EVENT).timestamps,
    }


class _EmptyEvent:
    timestamps = np.array([], dtype=float)


_EMPTY_EVENT = _EmptyEvent()


def run_lifetime_psth(
    workflow,
    rows,
    data_root,
    output_dir,
    *,
    signal,
    event,
    window=(-5.0, 20.0),
    dt=0.1,
    normalization="zscore",
    baseline=(-5.0, 0.0),
    heatmaps=True,
    heatmap_sort="event_order",
    heatmap_sort_window=None,
    heatmap_sort_direction="auto",
    heatmap_unmatched="bottom",
    heatmap_cmap="coolwarm",
) -> list[Path]:
    """Run shared event-aligned lifetime analysis and save auditable figures."""

    import matplotlib.pyplot as plt

    if signal not in LIFETIME_SIGNALS[workflow]:
        raise ValueError(f"Unsupported {workflow} signal: {signal}")
    if event not in LIFETIME_EVENTS:
        raise ValueError(f"Unsupported alignment event: {event}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        session, paths = load_lifetime_session(workflow, row, data_root)
        continuous = session.continuous[signal]
        alignment_times = session.events[event].timestamps
        time, trials, valid = extract_perievent_trials(
            continuous.timestamps,
            continuous.values,
            alignment_times,
            window=window,
            dt=dt,
        )
        normalized = normalize_trials(time, trials, normalization, baseline)
        if normalized.shape[0] == 0:
            raise ValueError(f"{session.session_id} has no complete {event} windows")
        results.append(
            {
                **row,
                **paths,
                "trials": normalized,
                "alignment_times": alignment_times[valid],
                "behavioral_events": _behavioral_events(session),
                "session_id": session.session_id,
                "units": continuous.units,
            }
        )
    by_mouse = defaultdict(list)
    for item in results:
        by_mouse[item["mouse"]].append(np.nanmean(item["trials"], axis=0))
    mouse_names = sorted(by_mouse)
    mouse_matrix = np.vstack(
        [np.nanmean(np.vstack(by_mouse[mouse]), axis=0) for mouse in mouse_names]
    )
    bundle = {
        "time": time,
        "normalization": normalization,
        "signal_type": "photometry",
        "event_key": event,
        "description": f"{signal} aligned to {event}",
        "ylabel": f"{signal} ({normalization})",
        "session_results": results,
        "mouse_names": mouse_names,
        "mouse_matrix": mouse_matrix,
    }
    outputs = []
    mean, sem = summarize_trials(mouse_matrix)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, mean)
    ax.fill_between(time, mean - sem, mean + sem, alpha=0.3)
    ax.axvline(0, color="black", linestyle="--")
    ax.set(
        xlabel=f"Time from {event} (s)",
        ylabel=f"{signal} ({normalization})",
        title=f"{workflow}: {signal} aligned to {event} (mouse mean ± SEM)",
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = destination / f"{workflow}_{signal}_{event}_psth.{suffix}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)
    if heatmaps:
        outputs.extend(
            save_psth_heatmaps(
                bundle,
                destination,
                sort=heatmap_sort,
                sort_window=heatmap_sort_window,
                direction=heatmap_sort_direction,
                unmatched=heatmap_unmatched,
                cmap=heatmap_cmap,
            )
        )
    metadata = {
        "workflow": workflow,
        "signal": signal,
        "event": event,
        "window": list(window),
        "dt": dt,
        "normalization": normalization,
        "baseline": list(baseline),
        "session_count": len(results),
        "mouse_count": len(mouse_names),
    }
    metadata_path = destination / "analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    outputs.append(metadata_path)
    return outputs


def run_lifetime_glm(
    workflow,
    rows,
    data_root,
    output_dir,
    *,
    signal,
    lick_kernel_seconds=10.0,
    ensure_kernel_seconds=20.0,
) -> list[Path]:
    """Fit the maintained lick/Ensure adaptive ridge model across sessions."""

    from fluopulse_analysis.glm import (
        build_adaptive_ensure_design,
        reconstruct_adaptive_kernels,
    )

    designs = []
    outcomes = []
    labels = []
    for row in rows:
        session, _ = load_lifetime_session(workflow, row, data_root)
        response = session.continuous[signal]
        uniform_time = np.linspace(
            response.timestamps[0], response.timestamps[-1], response.timestamps.size
        )
        outcome = np.interp(uniform_time, response.timestamps, response.values)
        lick_counts = event_times_to_counts(session.events["licks"].timestamps, uniform_time)
        ensure_counts = event_times_to_counts(
            session.events["ensure"].timestamps, uniform_time
        )
        finite = np.isfinite(outcome)
        if not np.all(finite):
            outcome = np.interp(
                uniform_time,
                uniform_time[finite],
                outcome[finite],
            )
        designs.append(
            build_adaptive_ensure_design(
                uniform_time,
                lick_counts,
                ensure_counts,
                lick_kernel_seconds=lick_kernel_seconds,
                ensure_kernel_seconds=ensure_kernel_seconds,
            )
        )
        outcomes.append(outcome)
        labels.append(_session_id(row))
    if len(designs) < 3:
        raise ValueError(
            "Lifetime GLM requires at least three sessions for nested held-out scoring"
        )
    alphas = np.logspace(-4, 4, 17)
    matrix = np.vstack([design.matrix for design in designs])
    response = np.concatenate(outcomes)
    groups = np.concatenate(
        [
            np.full(outcome.size, label, dtype=object)
            for outcome, label in zip(outcomes, labels, strict=True)
        ]
    )
    penalty = designs[0].penalty_weights
    outer_rows = []
    selected_alphas = []
    for outer_fold, (outer_train, outer_test) in enumerate(group_folds(groups), start=1):
        training_groups = groups[outer_train]
        inner_scores = np.full((alphas.size, len(np.unique(training_groups))))
        for inner_fold, (inner_train_local, inner_test_local) in enumerate(
            group_folds(training_groups)
        ):
            inner_train = outer_train[inner_train_local]
            inner_test = outer_train[inner_test_local]
            scale = np.sqrt(np.mean(matrix[inner_train] ** 2, axis=0))
            scale[(scale == 0) | ~np.isfinite(scale)] = 1.0
            for alpha_index, alpha in enumerate(alphas):
                coefficients = fit_ridge(
                    matrix[inner_train] / scale,
                    response[inner_train],
                    float(alpha),
                    penalize=penalty,
                )
                inner_scores[alpha_index, inner_fold] = r2_score(
                    response[inner_test], matrix[inner_test] / scale @ coefficients
                )
        best_alpha = float(alphas[int(np.nanargmax(np.nanmean(inner_scores, axis=1)))])
        selected_alphas.append(best_alpha)
        outer_scale = np.sqrt(np.mean(matrix[outer_train] ** 2, axis=0))
        outer_scale[(outer_scale == 0) | ~np.isfinite(outer_scale)] = 1.0
        coefficients = fit_ridge(
            matrix[outer_train] / outer_scale,
            response[outer_train],
            best_alpha,
            penalize=penalty,
        )
        held_out = str(np.unique(groups[outer_test])[0])
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "held_out_session": held_out,
                "selected_alpha": best_alpha,
                "baseline_r2": r2_score(
                    response[outer_test],
                    np.full(outer_test.size, np.mean(response[outer_train])),
                ),
                "r2": r2_score(
                    response[outer_test], matrix[outer_test] / outer_scale @ coefficients
                ),
            }
        )
    final_alpha = float(np.median(selected_alphas))
    final_scale = np.sqrt(np.mean(matrix**2, axis=0))
    final_scale[(final_scale == 0) | ~np.isfinite(final_scale)] = 1.0
    final_coefficients = (
        fit_ridge(
            matrix / final_scale,
            response,
            final_alpha,
            penalize=penalty,
        )
        / final_scale
    )
    kernels = reconstruct_adaptive_kernels(designs[0], final_coefficients)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    score_path = destination / "glm_cross_validation.csv"
    _write_dict_rows(score_path, outer_rows)
    kernel_path = destination / "glm_kernels.npz"
    np.savez_compressed(
        kernel_path,
        lick_time=designs[0].lick_basis.lag_times,
        ensure_time=designs[0].ensure_basis.lag_times,
        **kernels,
    )
    import matplotlib.pyplot as plt

    figure_paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(designs[0].lick_basis.lag_times, kernels["lick"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Time from lick (s)", ylabel=signal, title="Lick kernel")
    for name, values in kernels.items():
        if name.startswith("ensure"):
            axes[1].plot(
                designs[0].ensure_basis.lag_times,
                values,
                label=name.replace("ensure_", ""),
            )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Time from Ensure (s)", ylabel=signal, title="Ensure/adaptation kernels"
    )
    axes[1].legend()
    fig.suptitle(f"{workflow} lifetime GLM")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        figure_path = destination / f"{workflow}_{signal}_glm_kernels.{suffix}"
        fig.savefig(figure_path, dpi=300)
        figure_paths.append(figure_path)
    plt.close(fig)
    metadata_path = destination / "glm_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "workflow": workflow,
                "signal": signal,
                "sessions": labels,
                "final_alpha": final_alpha,
                "selection": (
                    "nested leave-one-session-out cross-validation; final alpha is "
                    "the median outer-fold selection"
                ),
                "interpretation": "predictive association; not causal",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [score_path, kernel_path, metadata_path, *figure_paths]


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
