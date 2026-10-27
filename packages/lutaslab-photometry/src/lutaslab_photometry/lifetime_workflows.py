"""Shared batch workflows for FluoPulse and iFLiP3 GUI/CLI analyses."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
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

from .group_analysis import extract_perievent_event_rate, save_psth_heatmaps
from .processed_provenance import build_processed_provenance, provenance_json
from .update_check import local_git_commit

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
LIFETIME_BEHAVIOR_SIGNALS = ("licking", "running_speed")
LIFETIME_TIME_SIGNALS = ("tau", "mpet")
LIFETIME_EVENTS = (
    "ensure",
    "visual_cue",
    "licks",
    "lick_bout_onset",
    "lick_bout_offset",
)
LIFETIME_PROCESSED_SCHEMA_VERSION = "1.0"
IFLIP3_CALIBRATION_SCHEMA_VERSION = "2.0"
# The header sync rate may be a measured rather than nominal value. Differences
# below 0.1% are small relative to the 0.1 ns lifetime-bin grid used by current
# recordings, while material changes such as 80 MHz versus 40 MHz remain clear.
IFLIP3_PULSE_INTERVAL_RTOL = 1e-3

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


def iflip3_residual_summary(data, residuals) -> dict[str, float]:
    """Return scale-free aggregate-decay residual metrics as percentages."""

    observed = np.asarray(data, dtype=float)
    errors = np.asarray(residuals, dtype=float)
    if observed.shape != errors.shape:
        raise ValueError("data and residuals must have the same shape")
    finite = np.isfinite(observed) & np.isfinite(errors)
    if not np.any(finite):
        return {
            "max_absolute_residual_percent_peak": float("nan"),
            "rmse_percent_peak": float("nan"),
            "absolute_residual_area_percent": float("nan"),
        }
    observed = observed[finite]
    errors = errors[finite]
    peak = float(np.max(np.abs(observed)))
    decay_area = float(np.sum(np.abs(observed)))
    return {
        "max_absolute_residual_percent_peak": (
            100.0 * float(np.max(np.abs(errors))) / peak if peak > 0 else float("nan")
        ),
        "rmse_percent_peak": (
            100.0 * float(np.sqrt(np.mean(errors**2))) / peak
            if peak > 0
            else float("nan")
        ),
        "absolute_residual_area_percent": (
            100.0 * float(np.sum(np.abs(errors))) / decay_area
            if decay_area > 0
            else float("nan")
        ),
    }


@dataclass(frozen=True)
class IFLIP3Calibration:
    """Saved shared decay-shape calibration for one acquisition configuration."""

    lifetimes: np.ndarray
    t0: float
    irf_sigma: float
    lifetime_time: np.ndarray
    pulse_interval: float
    spc_range: tuple[float, float]
    afterpulse_ratio: float
    session_qc: tuple[dict[str, Any], ...]
    success: bool
    message: str
    cost: float
    nfev: int
    created_utc: str
    code_commit: str | None = None
    schema_version: str = IFLIP3_CALIBRATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        model = {
            "n_components": 2,
            "lifetimes_ns": self.lifetimes.tolist(),
            "irf_sigma_ns": self.irf_sigma,
            "residual_background_counts_per_bin": 0.0,
            "time_point_specific_parameters": [
                "short_component_counts",
                "long_component_counts",
            ],
        }
        if self.schema_version == "1.0":
            model.update(
                {
                    "t0_ns": self.t0,
                    "shared_parameters": ["tau1", "tau2", "t0", "irf_sigma"],
                }
            )
        else:
            model.update(
                {
                    "reference_t0_ns": self.t0,
                    "shared_parameters": ["tau1", "tau2", "irf_sigma"],
                    "session_specific_parameters": ["t0"],
                }
            )
        return {
            "calibration_schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "code_commit": self.code_commit,
            "model": model,
            "acquisition": {
                "lifetime_time_ns": self.lifetime_time.tolist(),
                "pulse_interval_ns": self.pulse_interval,
                "spc_range_ns": list(self.spc_range),
                "afterpulse_ratio": self.afterpulse_ratio,
            },
            "fit": {
                "success": self.success,
                "message": self.message,
                "cost": self.cost,
                "nfev": self.nfev,
                "session_qc": list(self.session_qc),
            },
        }


def normalize_iflip3_fit_settings(
    settings: dict | None,
) -> dict[str, dict[str, float | str]]:
    """Validate GUI/CLI controls for the aggregate iFLIP3 decay fit."""

    supplied = settings or {}
    unknown = set(supplied) - set(IFLIP3_FIT_PARAMETER_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown iFLIP3 fit parameter(s): {', '.join(sorted(unknown))}"
        )
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


def _calibration_from_dict(payload: dict[str, Any]) -> IFLIP3Calibration:
    """Validate and reconstruct a saved iFLIP3 calibration."""

    schema = str(payload.get("calibration_schema_version", ""))
    if schema not in {"1.0", IFLIP3_CALIBRATION_SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported iFLIP3 calibration schema {schema!r}; "
            f"expected '1.0' or {IFLIP3_CALIBRATION_SCHEMA_VERSION!r}"
        )
    model = payload.get("model") or {}
    acquisition = payload.get("acquisition") or {}
    fit = payload.get("fit") or {}
    lifetimes = np.asarray(model.get("lifetimes_ns"), dtype=float)
    lifetime_time = np.asarray(acquisition.get("lifetime_time_ns"), dtype=float)
    if (
        lifetimes.shape != (2,)
        or np.any(~np.isfinite(lifetimes))
        or np.any(lifetimes <= 0)
    ):
        raise ValueError("Calibration must contain two positive finite lifetimes")
    if (
        lifetime_time.ndim != 1
        or lifetime_time.size < 3
        or np.any(~np.isfinite(lifetime_time))
    ):
        raise ValueError("Calibration lifetime-time axis is invalid")
    background = float(model.get("residual_background_counts_per_bin", np.nan))
    if background != 0.0:
        raise ValueError(
            "This calibration schema requires residual background fixed at zero"
        )
    spc_range = tuple(map(float, acquisition.get("spc_range_ns", ())))
    if len(spc_range) != 2 or spc_range[0] >= spc_range[1]:
        raise ValueError("Calibration SPC range is invalid")
    calibration = IFLIP3Calibration(
        lifetimes=np.sort(lifetimes),
        t0=float(model.get("reference_t0_ns", model.get("t0_ns"))),
        irf_sigma=float(model.get("irf_sigma_ns")),
        lifetime_time=lifetime_time,
        pulse_interval=float(acquisition.get("pulse_interval_ns")),
        spc_range=(spc_range[0], spc_range[1]),
        afterpulse_ratio=float(acquisition.get("afterpulse_ratio")),
        session_qc=tuple(fit.get("session_qc") or ()),
        success=bool(fit.get("success")),
        message=str(fit.get("message", "")),
        cost=float(fit.get("cost", np.nan)),
        nfev=int(fit.get("nfev", 0)),
        created_utc=str(payload.get("created_utc", "")),
        code_commit=(
            None
            if payload.get("code_commit") in (None, "")
            else str(payload["code_commit"])
        ),
        schema_version=schema,
    )
    scalar_values = (
        calibration.t0,
        calibration.irf_sigma,
        calibration.pulse_interval,
        calibration.afterpulse_ratio,
    )
    if not all(np.isfinite(value) for value in scalar_values):
        raise ValueError("Calibration contains non-finite instrument parameters")
    if calibration.irf_sigma <= 0 or calibration.pulse_interval <= 0:
        raise ValueError("Calibration IRF width and pulse interval must be positive")
    if not calibration.success:
        raise ValueError("Calibration does not record a successful global fit")
    if schema == IFLIP3_CALIBRATION_SCHEMA_VERSION:
        session_t0s = np.asarray(
            [item.get("t0_ns", np.nan) for item in calibration.session_qc],
            dtype=float,
        )
        if session_t0s.size == 0 or np.any(~np.isfinite(session_t0s)):
            raise ValueError("Calibration must record a finite t0 for every session")
    return calibration


def next_iflip3_calibration_path(
    path: str | Path,
    calibration: IFLIP3Calibration,
    *,
    now: datetime | None = None,
) -> Path:
    """Return a unique, descriptive path for a new calibration JSON version."""

    requested = Path(path).expanduser()
    frequency_mhz = 1e3 / calibration.pulse_interval
    frequency_label = f"{frequency_mhz:.6g}".replace(".", "p")
    stem = re.sub(r"__\d{4}-\d{2}-\d{2}_\d{6}Z(?:__r\d{2})?$", "", requested.stem)
    if stem.casefold() in {"shared", "calibration", "iflip3_calibration"}:
        stem = f"iflip3_{frequency_label}mhz_{len(calibration.session_qc)}sessions"
    safe_stem = "_".join(
        part
        for part in "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in stem
        ).split("_")
        if part
    )
    safe_stem = safe_stem or "iflip3_calibration"
    timestamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d_%H%M%SZ")
    candidate = requested.with_name(f"{safe_stem}__{timestamp}.json")
    revision = 2
    while candidate.exists():
        candidate = requested.with_name(
            f"{safe_stem}__{timestamp}__r{revision:02d}.json"
        )
        revision += 1
    return candidate


def save_iflip3_calibration(calibration: IFLIP3Calibration, path: str | Path) -> Path:
    """Atomically save a versioned iFLIP3 calibration JSON artifact."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        suffix=".json",
        prefix=f".{destination.stem}-",
        dir=destination.parent,
        delete=False,
    ) as stream:
        json.dump(calibration.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        _replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_iflip3_calibration(path: str | Path) -> IFLIP3Calibration:
    """Load and validate a saved iFLIP3 calibration JSON artifact."""

    source = Path(path).expanduser()
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Calibration JSON must contain an object")
    return _calibration_from_dict(payload)


def lifetime_preprocessing_parameters(
    workflow: str,
    iflip3_fit_settings: dict | None = None,
    iflip3_calibration: IFLIP3Calibration | None = None,
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
                {
                    "mode": "shared_multi_session_calibration",
                    "lifetimes_ns": iflip3_calibration.lifetimes.tolist(),
                    "t0_mode": (
                        "shared"
                        if iflip3_calibration.schema_version == "1.0"
                        else "session_specific"
                    ),
                    "reference_t0_ns": iflip3_calibration.t0,
                    "irf_sigma_ns": iflip3_calibration.irf_sigma,
                    "residual_background_counts_per_bin": 0.0,
                    "calibration_created_utc": iflip3_calibration.created_utc,
                    "calibration_code_commit": iflip3_calibration.code_commit,
                    "calibration_session_ids": [
                        item.get("session_id") for item in iflip3_calibration.session_qc
                    ],
                }
                if iflip3_calibration is not None
                else (
                    None
                    if iflip3_fit_settings is None
                    else normalize_iflip3_fit_settings(iflip3_fit_settings)
                )
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
        "nidaq_path": _mat_candidates(photometry_folder, mouse, date, run, "nidaq"),
        "running_path": _mat_candidates(photometry_folder, mouse, date, run, "running"),
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
        and not re.search(
            r"(?:^|[_\- ])(?:bg|background|dark)(?:$|[_\- ])", path.stem, re.I
        )
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
            {
                column: str(row[column] or "").strip()
                for column in LIFETIME_PATH_COLUMNS[workflow]
            }
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


def _load_corrected_iflip3_curves(
    row: dict,
    data_root: Path,
    *,
    spc_range: tuple[float, float],
    afterpulse_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float, bool, dict[str, Path | None]]:
    """Load one recording and return its corrected curves inside the fit window."""

    from iflip3 import average_background, calculate_mpet, lifetime_window, read_iflip3

    resolved = _iflip3_paths(row, data_root)
    recording = read_iflip3(resolved["source_path"])
    measured_background = None
    if resolved["background_path"] is not None:
        measured_background = average_background(
            [read_iflip3(resolved["background_path"])]
        )
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
    return (
        np.asarray(recording.lifetime_time[use], dtype=float),
        corrected[use],
        float(recording.header.pulse_interval_ns),
        measured_background is not None,
        resolved,
    )


def _pulse_intervals_compatible(first: float, second: float) -> bool:
    """Accept small measured sync-rate variation, not a changed repetition rate."""

    return bool(
        np.isclose(
            first,
            second,
            rtol=IFLIP3_PULSE_INTERVAL_RTOL,
            atol=0.0,
        )
    )


def _pulse_interval_description(interval: float) -> str:
    return f"{interval:.9g} ns ({1e3 / interval:.9g} MHz)"


def calibrate_iflip3_sessions(
    rows: list[dict] | tuple[dict, ...],
    data_root: str | Path,
    *,
    spc_range: tuple[float, float] = (0.4, 12.3),
    afterpulse_ratio: float = 0.03,
    initial_lifetimes: tuple[float, float] = (0.6, 2.5),
    initial_t0: float | tuple[float, ...] | None = None,
    initial_sigma: float = 0.13,
) -> IFLIP3Calibration:
    """Estimate shared lifetimes/IRF width and one t0 per session.

    Each selected session contributes one positive-area-normalized aggregate
    decay. Residual fit background is deliberately excluded. This gives each
    session comparable influence while allowing a separate pair of aggregate
    component amplitudes for every recording. Each recording's aggregate decay
    supplies its own bounded nonlinear t0 parameter.
    """

    from iflip3 import fit_global

    if len(rows) < 2:
        raise ValueError("Multi-session calibration requires at least two sessions")
    root = Path(data_root).expanduser()
    normalized_decays: list[np.ndarray] = []
    estimated_t0s: list[float] = []
    loaded: list[dict[str, Any]] = []
    reference_time = None
    reference_pulse_interval = None
    reference_session_id = None
    for row in rows:
        time_axis, curves, pulse_interval, has_background, resolved = (
            _load_corrected_iflip3_curves(
                row,
                root,
                spc_range=spc_range,
                afterpulse_ratio=afterpulse_ratio,
            )
        )
        if reference_time is None:
            reference_time = time_axis
            reference_pulse_interval = pulse_interval
            reference_session_id = _session_id(row)
        elif time_axis.shape != reference_time.shape or not np.allclose(
            time_axis, reference_time, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                f"Session {_session_id(row)} has a different lifetime-bin axis; "
                "create a separate calibration for that acquisition configuration"
            )
        elif not _pulse_intervals_compatible(pulse_interval, reference_pulse_interval):
            raise ValueError(
                f"Session {_session_id(row)} has pulse interval "
                f"{_pulse_interval_description(pulse_interval)}, but reference session "
                f"{reference_session_id} has "
                f"{_pulse_interval_description(reference_pulse_interval)}. "
                "create a separate calibration for that acquisition configuration"
            )
        aggregate = np.sum(curves, axis=1)
        positive_area = float(np.sum(np.clip(aggregate, 0.0, None)))
        if not np.isfinite(positive_area) or positive_area <= 0:
            raise ValueError(
                f"Session {_session_id(row)} has no positive aggregate decay area"
            )
        normalized = aggregate / positive_area
        normalized_decays.append(normalized[:, None])
        estimated_t0s.append(float(time_axis[int(np.argmax(normalized))] - 0.5))
        loaded.append(
            {
                "session_id": _session_id(row),
                "mouse": str(row["mouse"]),
                "date": str(row["date"]),
                "run": int(row["run"]),
                "source_path": str(resolved["source_path"]),
                "background_path": (
                    ""
                    if resolved["background_path"] is None
                    else str(resolved["background_path"])
                ),
                "measured_background_applied": has_background,
                "positive_aggregate_area": positive_area,
            }
        )
    assert reference_time is not None and reference_pulse_interval is not None
    result = fit_global(
        reference_time,
        normalized_decays,
        n_components=2,
        pulse_interval=reference_pulse_interval,
        initial_lifetimes=initial_lifetimes,
        initial_t0=estimated_t0s if initial_t0 is None else initial_t0,
        initial_sigma=initial_sigma,
        shared_t0=False,
        shared_sigma=True,
        include_background=False,
        nonnegative=True,
        weighting="none",
        normalize="area",
        max_nfev=250,
    )
    if not result.success:
        raise RuntimeError(
            f"Multi-session calibration did not converge: {result.message}"
        )
    session_qc = []
    for info, recording, normalized in zip(
        loaded, result.recordings, normalized_decays, strict=True
    ):
        observed = normalized[:, 0]
        fitted = recording.fitted[:, 0]
        residuals = observed - fitted
        total_sum_squares = float(np.sum((observed - observed.mean()) ** 2))
        residual_sum_squares = float(np.sum(residuals**2))
        residual_summary = iflip3_residual_summary(observed, residuals)
        session_qc.append(
            {
                **info,
                "rmse_normalized": float(np.sqrt(np.mean(residuals**2))),
                **residual_summary,
                "peak_time_ns": float(reference_time[int(np.argmax(observed))]),
                "t0_ns": recording.t0,
                "r_squared": (
                    1.0 - residual_sum_squares / total_sum_squares
                    if total_sum_squares > 0
                    else float("nan")
                ),
                "short_fraction_aggregate": float(recording.component_fractions[0, 0]),
                "long_fraction_aggregate": float(recording.component_fractions[0, 1]),
                "normalized_decay": observed.tolist(),
                "fitted_decay": fitted.tolist(),
                "residuals": residuals.tolist(),
            }
        )
    reference_t0 = float(np.median([item["t0_ns"] for item in session_qc]))
    bin_width = float(np.median(np.diff(reference_time)))
    offset_warning_threshold = max(2.0 * abs(bin_width), 0.25)
    for item in session_qc:
        item["t0_offset_from_median_ns"] = float(item["t0_ns"] - reference_t0)
        item["t0_offset_flag"] = bool(
            abs(item["t0_offset_from_median_ns"]) > offset_warning_threshold
        )
    from .update_check import local_git_commit

    project_root = Path(__file__).resolve().parents[4]
    return IFLIP3Calibration(
        lifetimes=np.asarray(result.lifetimes, dtype=float),
        t0=reference_t0,
        irf_sigma=result.recordings[0].irf_sigma,
        lifetime_time=reference_time,
        pulse_interval=reference_pulse_interval,
        spc_range=tuple(map(float, spc_range)),
        afterpulse_ratio=float(afterpulse_ratio),
        session_qc=tuple(session_qc),
        success=result.success,
        message=result.message,
        cost=result.cost,
        nfev=result.nfev,
        created_utc=datetime.now(UTC).isoformat(),
        code_commit=local_git_commit(project_root),
    )


def preview_iflip3_fit(
    row: dict,
    data_root: str | Path,
    settings: dict | None = None,
    *,
    spc_range: tuple[float, float] = (0.4, 12.3),
    afterpulse_ratio: float = 0.03,
) -> IFLIP3FitPreview:
    """Fit one session aggregate decay and decompose every photometry sample."""

    from iflip3 import fit_decay, fit_target, periodic_exgaussian_basis

    root = Path(data_root).expanduser()
    time, curves, pulse_interval, _, _ = _load_corrected_iflip3_curves(
        row,
        root,
        spc_range=spc_range,
        afterpulse_ratio=afterpulse_ratio,
    )
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
        pulse_interval=pulse_interval,
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
        pulse_interval=pulse_interval,
        residual_background_per_bin=per_sample_background,
        weighting="none" if np.any(curves < 0) else "poisson",
    )
    short_basis = periodic_exgaussian_basis(
        time,
        fit.lifetimes[0],
        fit.t0,
        fit.irf_sigma,
        pulse_interval,
    )
    long_basis = periodic_exgaussian_basis(
        time,
        fit.lifetimes[1],
        fit.t0,
        fit.irf_sigma,
        pulse_interval,
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


def _session_t0_from_calibration(
    row: dict,
    time: np.ndarray,
    curves: np.ndarray,
    calibration: IFLIP3Calibration,
) -> tuple[float, str]:
    """Resolve or estimate the fixed t0 used for one calibrated recording."""

    if calibration.schema_version == "1.0":
        return calibration.t0, "legacy shared calibration t0"
    session_id = _session_id(row)
    for item in calibration.session_qc:
        if item.get("session_id") == session_id and np.isfinite(
            item.get("t0_ns", np.nan)
        ):
            return float(item["t0_ns"]), "saved calibration-session t0"

    from iflip3 import fit_target
    from scipy.optimize import minimize_scalar

    aggregate = np.sum(curves, axis=1, keepdims=True)
    positive_area = float(np.sum(np.clip(aggregate[:, 0], 0.0, None)))
    if not np.isfinite(positive_area) or positive_area <= 0:
        raise ValueError(f"Session {session_id} has no positive aggregate decay area")
    aggregate /= positive_area
    bin_width = float(np.median(np.diff(time)))
    initial = float(time[int(np.argmax(aggregate[:, 0]))] - 0.5)
    half_width = max(1.0, 10.0 * abs(bin_width))
    lower = max(float(time.min() - 1.0), initial - half_width)
    upper = min(float(time.max()), initial + half_width)

    def objective(t0):
        result = fit_target(
            time,
            aggregate,
            lifetimes=calibration.lifetimes,
            t0=float(t0),
            irf_sigma=calibration.irf_sigma,
            pulse_interval=calibration.pulse_interval,
            residual_background_per_bin=0.0,
            weighting="none",
        )
        return float(np.sum(result.residuals**2))

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(lower, upper),
        options={"xatol": max(abs(bin_width) / 100.0, 1e-5)},
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f"Could not estimate session-specific t0 for {session_id}")
    return float(result.x), "estimated from this session's aggregate decay"


def apply_iflip3_calibration(
    row: dict,
    data_root: str | Path,
    calibration: IFLIP3Calibration,
) -> IFLIP3FitPreview:
    """Decompose one recording with all nonlinear shape parameters fixed."""

    from iflip3 import fit_target, periodic_exgaussian_basis

    root = Path(data_root).expanduser()
    time, curves, pulse_interval, _, _ = _load_corrected_iflip3_curves(
        row,
        root,
        spc_range=calibration.spc_range,
        afterpulse_ratio=calibration.afterpulse_ratio,
    )
    if time.shape != calibration.lifetime_time.shape or not np.allclose(
        time, calibration.lifetime_time, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            f"Session {_session_id(row)} does not match the calibration lifetime-bin axis"
        )
    if not _pulse_intervals_compatible(pulse_interval, calibration.pulse_interval):
        raise ValueError(
            f"Session {_session_id(row)} has pulse interval "
            f"{_pulse_interval_description(pulse_interval)}, but the calibration uses "
            f"{_pulse_interval_description(calibration.pulse_interval)}"
        )
    session_t0, t0_source = _session_t0_from_calibration(row, time, curves, calibration)
    weighting = "none" if np.any(curves < 0) else "poisson"
    target = fit_target(
        time,
        curves,
        lifetimes=calibration.lifetimes,
        t0=session_t0,
        irf_sigma=calibration.irf_sigma,
        pulse_interval=calibration.pulse_interval,
        residual_background_per_bin=0.0,
        weighting=weighting,
    )
    aggregate = curves.sum(axis=1)
    aggregate_coefficients = target.component_counts.sum(axis=0)
    short_basis = periodic_exgaussian_basis(
        time,
        calibration.lifetimes[0],
        session_t0,
        calibration.irf_sigma,
        calibration.pulse_interval,
        normalize="area",
    )
    long_basis = periodic_exgaussian_basis(
        time,
        calibration.lifetimes[1],
        session_t0,
        calibration.irf_sigma,
        calibration.pulse_interval,
        normalize="area",
    )
    short_component = aggregate_coefficients[0] * short_basis
    long_component = aggregate_coefficients[1] * long_basis
    fitted = target.fitted.sum(axis=1)
    residuals = aggregate - fitted
    residual_sum_squares = float(np.sum(residuals**2))
    total_sum_squares = float(np.sum((aggregate - aggregate.mean()) ** 2))
    return IFLIP3FitPreview(
        lifetime_time=time,
        aggregate_decay=aggregate,
        fitted_decay=fitted,
        short_component=short_component,
        long_component=long_component,
        residuals=residuals,
        lifetimes=calibration.lifetimes.copy(),
        t0=session_t0,
        irf_sigma=calibration.irf_sigma,
        background=0.0,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        r_squared=(
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else float("nan")
        ),
        success=True,
        message=f"Applied saved calibration; t0 {t0_source}",
        short_component_counts=target.component_counts[:, 0],
        long_component_counts=target.component_counts[:, 1],
        long_lifetime_fraction=target.component_fractions[:, 1],
        fit_rmse_by_sample=np.sqrt(np.mean(target.residuals**2, axis=0)),
    )


def _load_iflip3(
    row: dict,
    data_root: Path,
    fit_settings: dict | None = None,
    calibration: IFLIP3Calibration | None = None,
):
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
    if fit_settings is not None or calibration is not None:
        preview = (
            apply_iflip3_calibration(row, data_root, calibration)
            if calibration is not None
            else preview_iflip3_fit(row, data_root, fit_settings)
        )
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
            "fit_settings": (
                None
                if calibration is not None
                else normalize_iflip3_fit_settings(fit_settings)
            ),
            "calibration": (
                None
                if calibration is None
                else {
                    "schema_version": calibration.schema_version,
                    "created_utc": calibration.created_utc,
                    "code_commit": calibration.code_commit,
                    "calibration_session_ids": [
                        item.get("session_id") for item in calibration.session_qc
                    ],
                    "residual_background_counts_per_bin": 0.0,
                    "session_t0_source": preview.message,
                }
            ),
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
    return {name: "" if path is None else str(path) for name, path in paths.items()}


def _session_id(row: dict) -> str:
    return f"{row['mouse']}_{row['date']}_run{int(row['run']):03d}"


def load_lifetime_session(
    workflow: str,
    row: dict,
    data_root: str | Path,
    *,
    iflip3_fit_settings: dict | None = None,
    iflip3_calibration: IFLIP3Calibration | None = None,
):
    """Load and align one lifetime session into the shared core representation."""

    root = Path(data_root).expanduser()
    if workflow == "fluopulse":
        return _load_fluopulse(row, root)
    if workflow == "iflip3":
        return _load_iflip3(row, root, iflip3_fit_settings, iflip3_calibration)
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


def _replace_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 6,
    initial_delay: float = 0.1,
) -> None:
    """Atomically replace a file, tolerating short-lived Windows/SMB locks."""

    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if attempt == attempts - 1:
                raise PermissionError(
                    f"Could not replace {destination} because it remained in use. "
                    "Close any program reading that processed file, wait for any "
                    "other analysis of the session to finish, and retry preprocessing."
                ) from error
            time.sleep(initial_delay * (2**attempt))


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
        "metadata_json": np.asarray(
            json.dumps(session.metadata, default=_json_default)
        ),
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
        _replace_with_retry(temporary_path, destination)
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
    iflip3_calibration: IFLIP3Calibration | str | Path | None = None,
) -> list[Path]:
    """Process raw lifetime sessions and save each result beside its recording."""

    outputs = []
    calibration = (
        load_iflip3_calibration(iflip3_calibration)
        if isinstance(iflip3_calibration, (str, Path))
        else iflip3_calibration
    )
    if calibration is not None and iflip3_fit_settings is not None:
        raise ValueError(
            "Use either an iFLIP3 calibration or per-session fit settings, not both"
        )
    preprocessing_parameters = lifetime_preprocessing_parameters(
        workflow,
        iflip3_fit_settings,
        calibration,
    )
    for row in rows:
        output = lifetime_processed_path(workflow, row, data_root)
        if output.exists() and not overwrite:
            outputs.append(output)
            continue
        identity = (
            f"{row.get('mouse', '<unknown>')} {row.get('date', '<unknown>')} "
            f"run {row.get('run', '<unknown>')}"
        )
        source_column = "iflip_path" if workflow == "iflip3" else "doric_path"
        try:
            if workflow == "iflip3" and iflip3_fit_settings is not None:
                session, paths = load_lifetime_session(
                    workflow,
                    row,
                    data_root,
                    iflip3_fit_settings=iflip3_fit_settings,
                    iflip3_calibration=calibration,
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
        except Exception as error:
            raise RuntimeError(
                f"Failed to preprocess {workflow} session {identity}; "
                f"source={row.get(source_column, '<missing>')!r}, "
                f"nidaq={row.get('nidaq_path', '<missing>')!r}: {error}"
            ) from error
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
                f"Unsupported processed lifetime schema {schema!r} in {path}; rerun preprocessing."
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


def _lifetime_plot_values(signal: str, continuous: ContinuousSignal):
    """Return plotting values and units, expressing lifetime signals in ps."""

    values = np.asarray(continuous.values, dtype=float)
    units = str(continuous.units).strip()
    if signal not in LIFETIME_TIME_SIGNALS:
        return values, units
    if units == "ns":
        return values * 1000.0, "ps"
    if units == "ps":
        return values, units
    raise ValueError(f"{signal} must use ns or ps units, not {units!r}")


def _lifetime_psth_ylabel(signal: str, normalization: str, units: str) -> str:
    if signal == "licking":
        return "Lick rate (licks/s)"
    if signal == "running_speed":
        return "Running speed (cm/s)"
    label = {"tau": "τ", "mpet": "MPET"}.get(signal, signal.replace("_", " "))
    prefix = "Δ" if normalization == "subtract" else ""
    return f"{prefix}{label} ({units})" if units else f"{prefix}{label}"


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
    normalization="subtract",
    baseline=(-5.0, 0.0),
    y_limits=None,
    heatmaps=True,
    heatmap_sort="event_order",
    heatmap_sort_window=None,
    heatmap_sort_direction="auto",
    heatmap_unmatched="bottom",
    heatmap_cmap="coolwarm",
    first_event_only=False,
    allow_partial_windows=False,
    progress_callback=None,
) -> list[Path]:
    """Run shared event-aligned lifetime analysis and save auditable figures."""

    import matplotlib.pyplot as plt

    supported_signals = (*LIFETIME_SIGNALS[workflow], *LIFETIME_BEHAVIOR_SIGNALS)
    if signal not in supported_signals:
        raise ValueError(f"Unsupported {workflow} signal: {signal}")
    if event not in LIFETIME_EVENTS:
        raise ValueError(f"Unsupported alignment event: {event}")
    if signal not in LIFETIME_BEHAVIOR_SIGNALS and normalization not in (
        "subtract",
        "none",
        None,
    ):
        raise ValueError(
            "Lifetime PSTHs support only baseline subtraction or no normalization."
        )
    effective_normalization = (
        "none" if signal in LIFETIME_BEHAVIOR_SIGNALS else normalization or "none"
    )
    if y_limits is not None and y_limits[0] >= y_limits[1]:
        raise ValueError("y_limits minimum must be less than maximum")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    total_sessions = len(rows)
    for session_index, row in enumerate(rows, start=1):
        if progress_callback is not None:
            progress_callback(
                session_index,
                total_sessions,
                f"{row['mouse']} {row['date']} run {int(row['run'])}",
            )
        session, paths = load_processed_lifetime_session(workflow, row, data_root)
        alignment_times = _lifetime_event_times(session, event)
        if first_event_only:
            alignment_times = alignment_times[:1]
        if signal == "licking":
            reference = session.continuous[LIFETIME_SIGNALS[workflow][0]]
            time, trials, valid = extract_perievent_event_rate(
                session.events.get("licks", _EMPTY_EVENT).timestamps,
                alignment_times,
                (float(reference.timestamps[0]), float(reference.timestamps[-1])),
                window=window,
                dt=dt,
                require_complete=not allow_partial_windows,
            )
            units = "licks/s"
            signal_type = "licking"
        else:
            if signal not in session.continuous:
                if signal == "running_speed":
                    warnings.warn(
                        f"Skipping {session.session_id}: running data are unavailable.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                raise ValueError(f"{session.session_id} has no {signal} signal")
            continuous = session.continuous[signal]
            plot_values, units = _lifetime_plot_values(signal, continuous)
            time, trials, valid = extract_perievent_trials(
                continuous.timestamps,
                plot_values,
                alignment_times,
                window=window,
                dt=dt,
                require_complete=not allow_partial_windows,
            )
            if signal == "running_speed":
                units = "cm/s"
            signal_type = "running" if signal == "running_speed" else "photometry"
        normalized = normalize_trials(
            time,
            trials,
            effective_normalization,
            baseline,
        )
        if normalized.shape[0] == 0:
            warnings.warn(
                f"Skipping {session.session_id}: no usable {event} windows.",
                UserWarning,
                stacklevel=2,
            )
            continue
        results.append(
            {
                **row,
                **paths,
                "trials": normalized,
                "alignment_times": alignment_times[valid],
                "behavioral_events": _behavioral_events(session),
                "session_id": session.session_id,
                "units": units,
            }
        )
    if not results:
        raise ValueError(
            f"No sessions contained usable {signal} data and {event} windows"
        )
    by_mouse = defaultdict(list)
    for item in results:
        by_mouse[item["mouse"]].append(summarize_trials(item["trials"])[0])
    mouse_names = sorted(by_mouse)
    mouse_matrix = np.vstack(
        [summarize_trials(np.vstack(by_mouse[mouse]))[0] for mouse in mouse_names]
    )
    ylabel = _lifetime_psth_ylabel(signal, effective_normalization, units)
    bundle = {
        "time": time,
        "normalization": effective_normalization,
        "signal_type": signal_type,
        "event_key": event,
        "description": f"{signal} aligned to {event}",
        "ylabel": ylabel,
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
        ylabel=ylabel,
        title=f"{workflow}: {signal} aligned to {event} (mouse mean ± SEM)",
    )
    if y_limits is not None:
        ax.set_ylim(*y_limits)
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
    session_metadata = []
    for item in results:
        record = {}
        for key in (
            "session_id",
            "mouse",
            "date",
            "run",
            "group",
            "condition",
            "source_path",
            "nidaq_path",
            "running_path",
            "processed_path",
        ):
            if key in item and item[key] not in (None, ""):
                value = item[key]
                record[key] = str(value) if isinstance(value, Path) else value
        session_metadata.append(record)
    metadata = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_label": destination.name,
        "output_directory": str(destination.resolve()),
        "code_commit": local_git_commit(Path(__file__).resolve().parents[4]),
        "workflow": workflow,
        "signal": signal,
        "event": event,
        "window": list(window),
        "dt": dt,
        "normalization": effective_normalization,
        "units": units,
        "ylabel": ylabel,
        "baseline": list(baseline),
        "y_limits": None if y_limits is None else list(y_limits),
        "first_event_only": bool(first_event_only),
        "allow_partial_windows": bool(allow_partial_windows),
        "heatmaps": {
            "enabled": bool(heatmaps),
            "sort": heatmap_sort,
            "sort_window": (
                None if heatmap_sort_window is None else list(heatmap_sort_window)
            ),
            "sort_direction": heatmap_sort_direction,
            "unmatched": heatmap_unmatched,
            "color_map": heatmap_cmap,
        },
        "selected_groups": sorted(
            {str(item.get("group", "")) for item in results if item.get("group")}
        ),
        "selected_conditions": sorted(
            {
                str(item.get("condition", ""))
                for item in results
                if item.get("condition")
            }
        ),
        "session_count": len(results),
        "mouse_count": len(mouse_names),
        "sessions": session_metadata,
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
    response_units = None
    for row in rows:
        session, _ = load_processed_lifetime_session(workflow, row, data_root)
        response = session.continuous[signal]
        plot_values, units = _lifetime_plot_values(signal, response)
        if response_units is None:
            response_units = units
        elif units != response_units:
            raise ValueError(
                f"Lifetime GLM sessions have inconsistent {signal} units: "
                f"{response_units!r} and {units!r}"
            )
        uniform_time = np.linspace(
            response.timestamps[0], response.timestamps[-1], response.timestamps.size
        )
        outcome = np.interp(uniform_time, response.timestamps, plot_values)
        lick_counts = event_times_to_counts(
            session.events["licks"].timestamps, uniform_time
        )
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
    for outer_fold, (outer_train, outer_test) in enumerate(
        group_folds(groups), start=1
    ):
        training_groups = groups[outer_train]
        inner_scores = np.full(
            (alphas.size, len(np.unique(training_groups))), np.nan, dtype=float
        )
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
                "mouse": next(
                    str(row["mouse"]) for row in rows if _session_id(row) == held_out
                ),
                "selected_alpha": best_alpha,
                "baseline_r2": r2_score(
                    response[outer_test],
                    np.full(outer_test.size, np.mean(response[outer_train])),
                ),
                "r2": r2_score(
                    response[outer_test],
                    matrix[outer_test] / outer_scale @ coefficients,
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
    reference_times = {
        "lick": np.asarray(designs[0].lick_basis.lag_times, dtype=float),
        "ensure": np.asarray(designs[0].ensure_basis.lag_times, dtype=float),
    }
    session_kernel_rows = []
    for row, label, design, outcome in zip(
        rows, labels, designs, outcomes, strict=True
    ):
        session_scale = np.sqrt(np.mean(design.matrix**2, axis=0))
        session_scale[(session_scale == 0) | ~np.isfinite(session_scale)] = 1.0
        session_coefficients = (
            fit_ridge(
                design.matrix / session_scale,
                outcome,
                final_alpha,
                penalize=design.penalty_weights,
            )
            / session_scale
        )
        session_kernels = reconstruct_adaptive_kernels(design, session_coefficients)
        for name, values in session_kernels.items():
            time_key = "lick" if name == "lick" else "ensure"
            source_time = (
                design.lick_basis.lag_times
                if time_key == "lick"
                else design.ensure_basis.lag_times
            )
            reference_time = reference_times[time_key]
            values = np.interp(reference_time, source_time, values)
            for lag_s, coefficient in zip(reference_time, values, strict=True):
                session_kernel_rows.append(
                    {
                        "mouse": str(row["mouse"]),
                        "date": str(row.get("date", "")),
                        "run": row.get("run", ""),
                        "session": label,
                        "kernel": name,
                        "lag_s": float(lag_s),
                        "coefficient": float(coefficient),
                    }
                )
    mouse_kernel_rows, group_kernel_rows = _summarize_lifetime_kernel_rows(
        session_kernel_rows
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    score_path = destination / "glm_cross_validation.csv"
    _write_dict_rows(score_path, outer_rows)
    mouse_score_path = destination / "glm_mouse_cross_validation.csv"
    mouse_score_rows = []
    for mouse in sorted({row["mouse"] for row in outer_rows}):
        selected = [row for row in outer_rows if row["mouse"] == mouse]
        mouse_score_rows.append(
            {
                "mouse": mouse,
                "baseline_r2": float(np.mean([row["baseline_r2"] for row in selected])),
                "r2": float(np.mean([row["r2"] for row in selected])),
                "n_sessions": len(selected),
            }
        )
    _write_dict_rows(mouse_score_path, mouse_score_rows)
    session_kernel_path = destination / "glm_session_kernels.csv"
    mouse_kernel_path = destination / "glm_mouse_kernels.csv"
    group_kernel_path = destination / "glm_group_kernels.csv"
    _write_dict_rows(session_kernel_path, session_kernel_rows)
    _write_dict_rows(mouse_kernel_path, mouse_kernel_rows)
    _write_dict_rows(group_kernel_path, group_kernel_rows)
    kernel_path = destination / "glm_kernels.npz"
    np.savez_compressed(
        kernel_path,
        lick_time=designs[0].lick_basis.lag_times,
        ensure_time=designs[0].ensure_basis.lag_times,
        **kernels,
    )
    import matplotlib.pyplot as plt

    figure_paths = []
    ylabel = _lifetime_psth_ylabel(signal, "subtract", response_units or "")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _plot_lifetime_kernel_summary(
        axes[0], mouse_kernel_rows, group_kernel_rows, "lick", "Lick kernel", "C0"
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Time from lick (s)", ylabel=ylabel, title="Lick kernel")
    ensure_colors = ("C1", "C2", "C3", "C4")
    ensure_labels = (
        "First Ensure",
        "Subsequent Ensure: early",
        "Subsequent Ensure: middle",
        "Subsequent Ensure: late",
    )
    for color, name, label in zip(
        ensure_colors,
        (
            "ensure_first",
            "ensure_progress_0",
            "ensure_progress_0.5",
            "ensure_progress_1",
        ),
        ensure_labels,
        strict=True,
    ):
        if name.startswith("ensure"):
            _plot_lifetime_kernel_summary(
                axes[1],
                mouse_kernel_rows,
                group_kernel_rows,
                name,
                label,
                color,
            )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Time from Ensure (s)", ylabel=ylabel, title="Ensure/adaptation kernels"
    )
    axes[1].legend()
    fig.suptitle(f"{workflow} lifetime GLM")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        figure_path = destination / f"{workflow}_{signal}_glm_kernels.{suffix}"
        fig.savefig(figure_path, dpi=300)
        figure_paths.append(figure_path)
    plt.close(fig)
    performance_fig, performance_ax = plt.subplots(figsize=(5.5, 4.5))
    for row in mouse_score_rows:
        performance_ax.plot(
            (0, 1),
            (row["baseline_r2"], row["r2"]),
            color="0.65",
            marker="o",
            linewidth=0.8,
            alpha=0.7,
        )
    score_values = np.asarray(
        [[row["baseline_r2"], row["r2"]] for row in mouse_score_rows], dtype=float
    )
    score_mean = np.nanmean(score_values, axis=0)
    score_sem = (
        np.nanstd(score_values, axis=0, ddof=1) / np.sqrt(score_values.shape[0])
        if score_values.shape[0] > 1
        else np.full(2, np.nan)
    )
    performance_ax.errorbar(
        (0, 1), score_mean, yerr=score_sem, color="C0", marker="o", linewidth=2.2
    )
    performance_ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    performance_ax.set(
        xticks=(0, 1),
        xticklabels=("Constant baseline", "Lick + Ensure GLM"),
        ylabel="Held-out $R^2$",
        title=f"{workflow} lifetime GLM performance",
    )
    performance_fig.tight_layout()
    for suffix in ("png", "svg"):
        performance_path = destination / f"{workflow}_{signal}_glm_performance.{suffix}"
        performance_fig.savefig(performance_path, dpi=300)
        figure_paths.append(performance_path)
    plt.close(performance_fig)
    metadata_path = destination / "glm_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "workflow": workflow,
                "signal": signal,
                "units": response_units,
                "ylabel": ylabel,
                "sessions": labels,
                "final_alpha": final_alpha,
                "selection": (
                    "nested leave-one-session-out cross-validation; final alpha is "
                    "the median outer-fold selection"
                ),
                "interpretation": "predictive association; not causal",
                "kernel_summary": (
                    "sessions averaged within mouse; individual mice and mean plus SEM are shown"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [
        score_path,
        mouse_score_path,
        kernel_path,
        session_kernel_path,
        mouse_kernel_path,
        group_kernel_path,
        metadata_path,
        *figure_paths,
    ]


def _summarize_lifetime_kernel_rows(rows):
    mouse_groups = defaultdict(list)
    for row in rows:
        mouse_groups[(row["mouse"], row["kernel"], row["lag_s"])].append(
            row["coefficient"]
        )
    mouse_rows = []
    for (mouse, kernel, lag_s), values in mouse_groups.items():
        mouse_rows.append(
            {
                "mouse": mouse,
                "kernel": kernel,
                "lag_s": lag_s,
                "coefficient": float(np.nanmean(values)),
                "n_sessions": len(values),
            }
        )
    group_groups = defaultdict(list)
    for row in mouse_rows:
        group_groups[(row["kernel"], row["lag_s"])].append(row["coefficient"])
    group_rows = []
    for (kernel, lag_s), values in group_groups.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        group_rows.append(
            {
                "kernel": kernel,
                "lag_s": lag_s,
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "sem": (
                    float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                    if len(finite) > 1
                    else np.nan
                ),
                "n_mice": len(finite),
            }
        )
    return mouse_rows, group_rows


def _plot_lifetime_kernel_summary(ax, mouse_rows, group_rows, kernel, label, color):
    for mouse in sorted({row["mouse"] for row in mouse_rows}):
        selected = sorted(
            (
                row
                for row in mouse_rows
                if row["mouse"] == mouse and row["kernel"] == kernel
            ),
            key=lambda row: row["lag_s"],
        )
        if selected:
            ax.plot(
                [row["lag_s"] for row in selected],
                [row["coefficient"] for row in selected],
                color=color,
                linewidth=0.7,
                alpha=0.2,
            )
    selected = sorted(
        (row for row in group_rows if row["kernel"] == kernel),
        key=lambda row: row["lag_s"],
    )
    if not selected:
        return
    lag = np.asarray([row["lag_s"] for row in selected], dtype=float)
    mean = np.asarray([row["mean"] for row in selected], dtype=float)
    sem = np.asarray([row["sem"] for row in selected], dtype=float)
    ax.plot(lag, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(lag, mean - sem, mean + sem, color=color, alpha=0.2)


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
