"""Shared batch workflows for FluoPulse and iFLiP3 GUI/CLI analyses."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lutaslab_core.events import find_lick_bouts
from lutaslab_core.glm import event_times_to_counts, fit_ridge, group_folds, r2_score
from lutaslab_core.perievent import (
    extract_perievent_trials,
    normalize_trials,
    summarize_trials,
)
from lutaslab_core.session import AlignedSession, ContinuousSignal, EventSeries

from .group_analysis import save_psth_heatmaps
from .processed_provenance import build_processed_provenance, provenance_json

LIFETIME_COMMON_COLUMNS = ("mouse", "date", "run", "group", "condition")
LIFETIME_PATH_COLUMNS = {
    "fluopulse": ("doric_path", "nidaq_path", "running_path"),
    "iflip3": ("iflip_path", "nidaq_path", "running_path", "background_path"),
}
LIFETIME_SIGNALS = {
    "fluopulse": ("tau", "amplitude", "fit_r_square"),
    "iflip3": (
        "mpet",
        "long_lifetime_fraction",
        "short_component_counts",
        "long_component_counts",
        "raw_intensity",
    ),
}
LIFETIME_EVENTS = (
    "ensure",
    "visual_cue",
    "licks",
    "lick_bout_onset",
    "lick_bout_offset",
)
LIFETIME_PROCESSED_SCHEMA_VERSION = "1.0"

IFLIP3_FIT_PARAMETER_DEFAULTS = {
    "tau1": {"mode": "auto", "value": 0.6, "lower": 0.03, "upper": 5.0},
    "tau2": {"mode": "auto", "value": 2.5, "lower": 0.2, "upper": 12.5},
    "t0": {"mode": "auto", "value": 1.0, "lower": -0.5, "upper": 3.0},
    "sigma": {"mode": "auto", "value": 0.13, "lower": 0.01, "upper": 1.0},
    "background": {"mode": "auto", "value": 0.0, "lower": -1e6, "upper": 1e6},
}


@dataclass(frozen=True)
class IFLIP3FitPreview:
    """Aggregate decay fit and fixed-basis time-resolved decomposition."""

    lifetime_time: np.ndarray
    aggregate_decay: np.ndarray
    fitted_decay: np.ndarray
    short_component: np.ndarray
    long_component: np.ndarray
    residuals: np.ndarray
    lifetimes: np.ndarray
    t0: float
    irf_sigma: float
    background: float
    rmse: float
    r_squared: float
    success: bool
    message: str
    short_component_counts: np.ndarray
    long_component_counts: np.ndarray
    long_lifetime_fraction: np.ndarray
    fit_rmse_by_sample: np.ndarray


def normalize_iflip3_fit_settings(settings: dict | None) -> dict[str, dict[str, float | str]]:
    """Validate GUI/CLI controls for the aggregate iFLIP3 decay fit."""

    supplied = settings or {}
    unknown = set(supplied) - set(IFLIP3_FIT_PARAMETER_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown iFLIP3 fit parameter(s): {', '.join(sorted(unknown))}")
    normalized: dict[str, dict[str, float | str]] = {}
    for name, defaults in IFLIP3_FIT_PARAMETER_DEFAULTS.items():
        raw = supplied.get(name, {})
        mode = str(raw.get("mode", defaults["mode"])).strip().lower()
        if mode not in {"auto", "fixed", "bounded"}:
            raise ValueError(f"{name} mode must be auto, fixed, or bounded")
        value = float(raw.get("value", defaults["value"]))
        lower = float(raw.get("lower", defaults["lower"]))
        upper = float(raw.get("upper", defaults["upper"]))
        if not all(np.isfinite(item) for item in (value, lower, upper)):
            raise ValueError(f"{name} settings must be finite")
        if mode == "bounded" and lower >= upper:
            raise ValueError(f"{name} lower bound must be less than its upper bound")
        if mode == "bounded" and not lower <= value <= upper:
            raise ValueError(f"{name} starting value must lie inside its bounds")
        normalized[name] = {
            "mode": mode,
            "value": value,
            "lower": lower,
            "upper": upper,
        }
    return normalized


def lifetime_preprocessing_parameters(
    workflow: str,
    iflip3_fit_settings: dict | None = None,
) -> dict[str, Any]:
    """Return the effective defaults recorded with a lifetime processed file."""

    if workflow == "fluopulse":
        return {
            "nidaq_threshold": 1.5,
            "doric_sync_start_index": 0,
            "nidaq_sync_start_index": 0,
        }
    if workflow == "iflip3":
        return {
            "spc_range_ns": [0.4, 12.3],
            "afterpulse_ratio": 0.03,
            "sync_threshold": 1.5,
            "behavior_threshold": 1.5,
            "iflip_sync_start_index": 0,
            "nidaq_sync_start_index": 0,
            "lifetime_fit": (
                None
                if iflip3_fit_settings is None
                else normalize_iflip3_fit_settings(iflip3_fit_settings)
            ),
        }
    raise ValueError("workflow must be 'fluopulse' or 'iflip3'")


def _files_with_suffix(folders, suffix: str) -> list[Path]:
    """Return unique files from a small set of likely session folders."""

    found: dict[str, Path] = {}
    for folder in folders:
        try:
            paths = folder.rglob("*") if folder.is_dir() else ()
            for path in paths:
                if path.is_file() and path.suffix.casefold() == suffix.casefold():
                    found[str(path.resolve()).casefold()] = path
        except OSError:
            continue
    return sorted(found.values(), key=lambda path: str(path).casefold())


def _session_folders(root: Path, area: str, mouse: str, date: str) -> tuple[Path, ...]:
    area_root = root / area
    return (
        area_root / mouse / f"{mouse}_{date}",
        area_root / f"{mouse}_{date}",
        area_root / mouse / date,
    )


def _iflip_recording_matches(path: Path, mouse: str, date: str, run: int) -> bool:
    """Match standard iFLIP names while allowing descriptive suffix text."""

    prefix = re.escape(f"{mouse}_{date}")
    run_forms = (str(run), f"{run:03d}")
    patterns = [
        rf"^{prefix}_(?:run)?{re.escape(run_text)}(?:$|[_\- ].*)"
        for run_text in run_forms
    ]
    patterns.append(rf"^{prefix}{run:03d}(?:$|[_\- ].*)")
    stem = path.stem
    return any(re.match(pattern, stem, flags=re.IGNORECASE) for pattern in patterns)


def _mat_candidates(
    folder: Path,
    mouse: str,
    date: str,
    run: int,
    kind: str,
) -> list[Path]:
    """Find NI-DAQ/running files, preferring the exact conventional filename."""

    exact = folder / f"{mouse}-{date}-{run:03d}-{kind}.mat"
    candidates = _files_with_suffix((folder,), ".mat")
    stem_prefix = f"{mouse}-{date}-{run:03d}".casefold()
    marker = kind.casefold()
    matches = [
        path
        for path in candidates
        if path.stem.casefold().startswith(stem_prefix)
        and marker in path.stem.casefold()
    ]
    return sorted(
        matches,
        key=lambda path: (path.resolve() != exact.resolve(), path.name.casefold()),
    )


def discover_lifetime_paths(
    workflow: str,
    mouse: str,
    date: str,
    run: int,
    data_root: str | Path = "Z:/",
) -> dict[str, list[Path]]:
    """Find path choices for a lifetime manifest row from its session identity.

    The search is deliberately limited to the conventional final mouse/date folders.
    Descriptive text appended to a standard recording filename is allowed. iFLIP3
    backgrounds are returned as choices and are never selected by the backend.
    """

    if workflow not in LIFETIME_PATH_COLUMNS:
        raise ValueError("workflow must be 'fluopulse' or 'iflip3'")
    mouse = str(mouse).strip()
    date = str(date).strip()
    if not mouse or len(date) != 6 or not date.isdigit():
        raise ValueError("Enter a mouse and a six-digit date (YYMMDD).")
    try:
        run = int(run)
    except (TypeError, ValueError) as error:
        raise ValueError("Run must be a nonnegative integer.") from error
    if run < 0:
        raise ValueError("Run must be a nonnegative integer.")

    root = Path(data_root).expanduser()
    photometry_folder = root / "Photometry" / mouse / f"{mouse}_{date}"
    result = {
        "nidaq_path": _mat_candidates(
            photometry_folder, mouse, date, run, "nidaq"
        ),
        "running_path": _mat_candidates(
            photometry_folder, mouse, date, run, "running"
        ),
    }

    lifetime_folders = _session_folders(root, "FLIM FLIP", mouse, date)
    if workflow == "fluopulse":
        from fluopulse_analysis import infer_session_identity

        recordings = _files_with_suffix(lifetime_folders, ".doric")
        matching_recordings = []
        for path in recordings:
            try:
                identity = infer_session_identity(path)
            except ValueError:
                continue
            if (
                identity.mouse.casefold() == mouse.casefold()
                and identity.date == date
                and identity.run == run
            ):
                matching_recordings.append(path)
        result["doric_path"] = matching_recordings or recordings
        return result

    all_iflip = _files_with_suffix(lifetime_folders, ".iFLiP3")
    recordings = [
        path
        for path in all_iflip
        if _iflip_recording_matches(path, mouse, date, run)
        and not re.search(r"(?:^|[_\- ])(?:bg|background|dark)(?:$|[_\- ])", path.stem, re.I)
    ]
    background_folders = (*lifetime_folders, root / "FLIM FLIP" / "backgrounds")
    background_pool = _files_with_suffix(background_folders, ".iFLiP3")
    marked_backgrounds = [
        path
        for path in background_pool
        if re.search(r"(?:^|[_\- ])(?:bg|background|dark)(?:$|[_\- ])", path.stem, re.I)
    ]
    non_background_files = [
        path
        for path in all_iflip
        if not re.search(
            r"(?:^|[_\- ])(?:bg|background|dark)(?:$|[_\- ])", path.stem, re.I
        )
    ]
    result["iflip_path"] = recordings or non_background_files
    result["background_path"] = marked_backgrounds or [
        path for path in background_pool if path not in recordings
    ]
    return result


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


def _fluopulse_paths(row: dict, data_root: Path) -> dict[str, Path | None]:
    from fluopulse_analysis import (
        find_doric_files,
        infer_session_identity,
        nidaq_paths,
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
    explicit_nidaq = _optional_path(row["nidaq_path"], data_root)
    nidaq = explicit_nidaq or (inferred.nidaq if inferred.nidaq.is_file() else None)
    running = _optional_path(row["running_path"], data_root)
    if running is None and nidaq is not None and inferred.running.is_file():
        running = inferred.running
    return {"source_path": doric, "nidaq_path": nidaq, "running_path": running}


def _load_fluopulse(row: dict, data_root: Path):
    from fluopulse_analysis import process_aligned_session

    resolved = _fluopulse_paths(row, data_root)
    doric = resolved["source_path"]
    nidaq = resolved["nidaq_path"]
    running = resolved["running_path"]
    session = process_aligned_session(doric, nidaq, running_path=running)
    return session.to_core_session(_session_id(row)), _stringify_paths(resolved)


def _iflip3_paths(row: dict, data_root: Path) -> dict[str, Path | None]:
    from iflip3 import session_paths

    inferred = session_paths(row["mouse"], row["date"], row["run"], data_root=data_root)
    iflip = _optional_path(row["iflip_path"], data_root) or inferred.iflip
    explicit_nidaq = _optional_path(row["nidaq_path"], data_root)
    nidaq = explicit_nidaq or (inferred.nidaq if inferred.nidaq.is_file() else None)
    running = _optional_path(row["running_path"], data_root)
    if running is None and nidaq is not None and inferred.running.is_file():
        running = inferred.running
    background = _optional_path(row["background_path"], data_root)
    return {
        "source_path": iflip,
        "nidaq_path": nidaq,
        "running_path": running,
        "background_path": background,
    }


def preview_iflip3_fit(
    row: dict,
    data_root: str | Path,
    settings: dict | None = None,
    *,
    spc_range: tuple[float, float] = (0.4, 12.3),
    afterpulse_ratio: float = 0.03,
) -> IFLIP3FitPreview:
    """Fit one session aggregate decay and decompose every photometry sample."""

    from iflip3 import (
        average_background,
        calculate_mpet,
        fit_decay,
        fit_target,
        lifetime_window,
        periodic_exgaussian_basis,
        read_iflip3,
    )

    root = Path(data_root).expanduser()
    resolved = _iflip3_paths(row, root)
    recording = read_iflip3(resolved["source_path"])
    measured_background = None
    if resolved["background_path"] is not None:
        measured_background = average_background([read_iflip3(resolved["background_path"])])
    _, correction = calculate_mpet(
        recording,
        spc_range,
        t0=float(recording.header.get_path("state.t0.Value")),
        measured_background=measured_background,
        afterpulse_ratio=afterpulse_ratio,
    )
    corrected = np.asarray(correction.corrected, dtype=float)
    if corrected.ndim == 3:
        corrected = corrected[:, :, 0]
    use = lifetime_window(recording.lifetime_time, spc_range)
    time = np.asarray(recording.lifetime_time[use], dtype=float)
    curves = corrected[use]
    aggregate = curves.sum(axis=1)
    controls = normalize_iflip3_fit_settings(settings)
    initial = {name: float(control["value"]) for name, control in controls.items()}
    fixed = {
        name: float(control["value"])
        for name, control in controls.items()
        if control["mode"] == "fixed"
    }
    bounds = {
        name: (float(control["lower"]), float(control["upper"]))
        for name, control in controls.items()
        if control["mode"] == "bounded"
    }
    weighting = "none" if np.any(aggregate < 0) else "poisson"
    fit = fit_decay(
        time,
        aggregate,
        n_components=2,
        pulse_interval=recording.header.pulse_interval_ns,
        initial=initial,
        fixed=fixed,
        bounds=bounds,
        weighting=weighting,
    )
    per_sample_background = fit.background / max(curves.shape[1], 1)
    target = fit_target(
        time,
        curves,
        lifetimes=fit.lifetimes,
        t0=fit.t0,
        irf_sigma=fit.irf_sigma,
        pulse_interval=recording.header.pulse_interval_ns,
        residual_background_per_bin=per_sample_background,
        weighting="none" if np.any(curves < 0) else "poisson",
    )
    short_basis = periodic_exgaussian_basis(
        time,
        fit.lifetimes[0],
        fit.t0,
        fit.irf_sigma,
        recording.header.pulse_interval_ns,
    )
    long_basis = periodic_exgaussian_basis(
        time,
        fit.lifetimes[1],
        fit.t0,
        fit.irf_sigma,
        recording.header.pulse_interval_ns,
    )
    short_component = fit.amplitudes[0] * short_basis
    long_component = fit.amplitudes[1] * long_basis
    residual_sum_squares = float(np.sum(fit.residuals**2))
    total_sum_squares = float(np.sum((aggregate - aggregate.mean()) ** 2))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else float("nan")
    )
    counts = target.component_counts
    return IFLIP3FitPreview(
        lifetime_time=time,
        aggregate_decay=aggregate,
        fitted_decay=fit.fitted,
        short_component=short_component,
        long_component=long_component,
        residuals=fit.residuals,
        lifetimes=fit.lifetimes,
        t0=fit.t0,
        irf_sigma=fit.irf_sigma,
        background=fit.background,
        rmse=float(np.sqrt(np.mean(fit.residuals**2))),
        r_squared=r_squared,
        success=fit.success,
        message=fit.message,
        short_component_counts=counts[:, 0],
        long_component_counts=counts[:, 1],
        long_lifetime_fraction=target.component_fractions[:, 1],
        fit_rmse_by_sample=np.sqrt(np.mean(target.residuals**2, axis=0)),
    )


def _load_iflip3(row: dict, data_root: Path, fit_settings: dict | None = None):
    from iflip3 import process_aligned_session

    resolved = _iflip3_paths(row, data_root)
    iflip = resolved["source_path"]
    nidaq = resolved["nidaq_path"]
    running = resolved["running_path"]
    background = resolved["background_path"]
    session = process_aligned_session(
        iflip,
        nidaq,
        background,
        running_path=running,
    )
    core = session.to_core_session(_session_id(row))
    if fit_settings is not None:
        preview = preview_iflip3_fit(row, data_root, fit_settings)
        timestamps = core.continuous["mpet"].timestamps
        core.continuous.update(
            {
                "short_component_counts": ContinuousSignal(
                    timestamps, preview.short_component_counts, "counts"
                ),
                "long_component_counts": ContinuousSignal(
                    timestamps, preview.long_component_counts, "counts"
                ),
                "long_lifetime_fraction": ContinuousSignal(
                    timestamps, preview.long_lifetime_fraction, "fraction"
                ),
                "lifetime_fit_rmse": ContinuousSignal(
                    timestamps, preview.fit_rmse_by_sample, "counts/bin"
                ),
            }
        )
        core.metadata["lifetime_fit"] = {
            "lifetimes_ns": preview.lifetimes.tolist(),
            "t0_ns": preview.t0,
            "irf_sigma_ns": preview.irf_sigma,
            "aggregate_background_counts_per_bin": preview.background,
            "aggregate_rmse": preview.rmse,
            "aggregate_r_squared": preview.r_squared,
            "fit_settings": normalize_iflip3_fit_settings(fit_settings),
        }
        core.metadata["lifetime_fit_qc"] = {
            "lifetime_time_ns": preview.lifetime_time.tolist(),
            "aggregate_decay": preview.aggregate_decay.tolist(),
            "fitted_decay": preview.fitted_decay.tolist(),
            "short_component": preview.short_component.tolist(),
            "long_component": preview.long_component.tolist(),
            "residuals": preview.residuals.tolist(),
        }
    return core, _stringify_paths(resolved)


def _stringify_paths(paths: dict[str, Path | None]) -> dict[str, str]:
    return {
        name: "" if path is None else str(path)
        for name, path in paths.items()
    }


def _session_id(row: dict) -> str:
    return f"{row['mouse']}_{row['date']}_run{int(row['run']):03d}"


def load_lifetime_session(
    workflow: str,
    row: dict,
    data_root: str | Path,
    *,
    iflip3_fit_settings: dict | None = None,
):
    """Load and align one lifetime session into the shared core representation."""

    root = Path(data_root).expanduser()
    if workflow == "fluopulse":
        return _load_fluopulse(row, root)
    if workflow == "iflip3":
        return _load_iflip3(row, root, iflip3_fit_settings)
    raise ValueError("workflow must be 'fluopulse' or 'iflip3'")


def lifetime_processed_path(workflow: str, row: dict, data_root: str | Path) -> Path:
    """Return the processed lifetime file beside the primary raw recording."""

    root = Path(data_root).expanduser()
    if workflow == "fluopulse":
        source = _fluopulse_paths(row, root)["source_path"]
    elif workflow == "iflip3":
        source = _iflip3_paths(row, root)["source_path"]
    else:
        raise ValueError("workflow must be 'fluopulse' or 'iflip3'")
    if source is None:  # pragma: no cover - resolvers always provide a source
        raise FileNotFoundError("The primary lifetime recording could not be resolved.")
    return source.with_name(f"{source.stem}-processed.npz")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _save_processed_lifetime_session(
    workflow: str,
    session: AlignedSession,
    paths: dict[str, str],
    destination: Path,
    preprocessing_parameters: dict[str, Any],
) -> None:
    provenance = build_processed_provenance(
        workflow,
        preprocessing_parameters,
        processed_schema_version=LIFETIME_PROCESSED_SCHEMA_VERSION,
    )
    arrays: dict[str, Any] = {
        "processed_schema_version": np.asarray(LIFETIME_PROCESSED_SCHEMA_VERSION),
        "workflow": np.asarray(workflow),
        "session_id": np.asarray(session.session_id),
        "metadata_json": np.asarray(json.dumps(session.metadata, default=_json_default)),
        "source_paths_json": np.asarray(json.dumps(paths)),
        "provenance_json": np.asarray(provenance_json(provenance)),
    }
    for name, signal in session.continuous.items():
        arrays[f"continuous_{name}_time"] = signal.timestamps
        arrays[f"continuous_{name}_values"] = signal.values
        arrays[f"continuous_{name}_units"] = np.asarray(signal.units)
    for name, events in session.events.items():
        arrays[f"events_{name}"] = events.timestamps
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".npz",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def preprocess_lifetime_sessions(
    workflow,
    rows,
    data_root,
    *,
    overwrite=False,
    iflip3_fit_settings: dict | None = None,
) -> list[Path]:
    """Process raw lifetime sessions and save each result beside its recording."""

    outputs = []
    preprocessing_parameters = lifetime_preprocessing_parameters(
        workflow,
        iflip3_fit_settings,
    )
    for row in rows:
        output = lifetime_processed_path(workflow, row, data_root)
        if output.exists() and not overwrite:
            outputs.append(output)
            continue
        if workflow == "iflip3" and iflip3_fit_settings is not None:
            session, paths = load_lifetime_session(
                workflow,
                row,
                data_root,
                iflip3_fit_settings=iflip3_fit_settings,
            )
        else:
            session, paths = load_lifetime_session(workflow, row, data_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        _save_processed_lifetime_session(
            workflow,
            session,
            paths,
            output,
            preprocessing_parameters,
        )
        outputs.append(output)
    return outputs


def load_processed_lifetime_session(
    workflow: str,
    row: dict,
    data_root: str | Path,
) -> tuple[AlignedSession, dict[str, str]]:
    """Load the required processed lifetime artifact for downstream analysis."""

    path = lifetime_processed_path(workflow, row, data_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed lifetime file not found: {path}. Run lifetime preprocessing "
            "for this session before running PSTH or GLM analysis."
        )
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["processed_schema_version"].item())
        stored_workflow = str(data["workflow"].item())
        if schema != LIFETIME_PROCESSED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported processed lifetime schema {schema!r} in {path}; "
                "rerun preprocessing."
            )
        if stored_workflow != workflow:
            raise ValueError(
                f"Processed file {path} contains {stored_workflow}, not {workflow}."
            )
        continuous = {}
        for key in data.files:
            if not key.startswith("continuous_") or not key.endswith("_values"):
                continue
            name = key[len("continuous_") : -len("_values")]
            continuous[name] = ContinuousSignal(
                data[f"continuous_{name}_time"],
                data[key],
                str(data[f"continuous_{name}_units"].item()),
            )
        events = {
            key[len("events_") :]: EventSeries(data[key], key[len("events_") :])
            for key in data.files
            if key.startswith("events_")
        }
        metadata = json.loads(str(data["metadata_json"].item()))
        if "provenance_json" in data.files:
            metadata["provenance"] = json.loads(str(data["provenance_json"].item()))
        paths = json.loads(str(data["source_paths_json"].item()))
        session_id = str(data["session_id"].item())
    return (
        AlignedSession(
            session_id=session_id,
            continuous=continuous,
            events=events,
            metadata=metadata,
        ),
        {**paths, "processed_path": str(path)},
    )


def _behavioral_events(session) -> dict[str, np.ndarray]:
    bouts = find_lick_bouts(session.events.get("licks", _EMPTY_EVENT).timestamps)
    return {
        "solenoid_onset": session.events.get("ensure", _EMPTY_EVENT).timestamps,
        "cue_onset": session.events.get("visual_cue", _EMPTY_EVENT).timestamps,
        "lick_times": session.events.get("licks", _EMPTY_EVENT).timestamps,
        "lick_bout_onset": bouts.onset_times,
        "lick_bout_offset": bouts.offset_times,
        "lick_bout_duration": bouts.durations,
        "lick_bout_lick_count": bouts.lick_counts,
    }


def _lifetime_event_times(session, event: str) -> np.ndarray:
    if event in {"lick_bout_onset", "lick_bout_offset"}:
        bouts = find_lick_bouts(session.events.get("licks", _EMPTY_EVENT).timestamps)
        return bouts.onset_times if event.endswith("onset") else bouts.offset_times
    return session.events[event].timestamps


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
    first_event_only=False,
    allow_partial_windows=False,
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
        session, paths = load_processed_lifetime_session(workflow, row, data_root)
        continuous = session.continuous[signal]
        alignment_times = _lifetime_event_times(session, event)
        if first_event_only:
            alignment_times = alignment_times[:1]
        time, trials, valid = extract_perievent_trials(
            continuous.timestamps,
            continuous.values,
            alignment_times,
            window=window,
            dt=dt,
            require_complete=not allow_partial_windows,
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
        by_mouse[item["mouse"]].append(summarize_trials(item["trials"])[0])
    mouse_names = sorted(by_mouse)
    mouse_matrix = np.vstack(
        [summarize_trials(np.vstack(by_mouse[mouse]))[0] for mouse in mouse_names]
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
        "first_event_only": bool(first_event_only),
        "allow_partial_windows": bool(allow_partial_windows),
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
        session, _ = load_processed_lifetime_session(workflow, row, data_root)
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
