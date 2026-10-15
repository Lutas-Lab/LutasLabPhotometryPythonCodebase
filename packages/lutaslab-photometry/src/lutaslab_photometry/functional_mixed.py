"""Cross-acquisition preparation and execution of non-concurrent FLMMs."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from lutaslab_core.functional_trials import FunctionalTrialDataset
from lutaslab_core.perievent import extract_perievent_trials, normalize_trials

from .lifetime_workflows import (
    LIFETIME_EVENTS,
    LIFETIME_SIGNALS,
    _lifetime_event_times,
    load_processed_lifetime_session,
)
from .save_sessiondata import load_session
from .session_manifest import processed_session_path, resolve_session_channel

FLMM_EFFECTS = ("condition", "group", "progress")
CONVENTIONAL_FLMM_SIGNALS = ("dff", "raw465")
CONVENTIONAL_FLMM_EVENTS = (
    "cue_onset",
    "solenoid_onset",
    "lick_times",
    "lick_bout_onset",
    "lick_bout_offset",
)


def _zscore(values: list[int]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    scale = float(np.std(array))
    return np.zeros_like(array) if scale == 0 else (array - np.mean(array)) / scale


def _zscore_within(values: list[int], groups: list[str]) -> np.ndarray:
    """Standardize trial order within session so session size is not a confound."""

    result = np.zeros(len(values), dtype=float)
    group_array = np.asarray(groups)
    value_array = np.asarray(values, dtype=float)
    for group in np.unique(group_array):
        selected = group_array == group
        result[selected] = _zscore(value_array[selected].tolist())
    return result


def _session_numbers(rows: list[dict]) -> dict[tuple[str, str, int], int]:
    identities = {
        (str(row["mouse"]), str(row["date"]), int(row["run"])) for row in rows
    }
    result = {}
    for mouse in sorted({identity[0] for identity in identities}):
        ordered = sorted(
            (identity for identity in identities if identity[0] == mouse),
            key=lambda identity: (identity[1], identity[2]),
        )
        result.update({identity: index for index, identity in enumerate(ordered, start=1)})
    return result


def _conventional_trial_source(row, data_root, signal, event, channel):
    selected_channel = resolve_session_channel(row, channel)
    session = load_session(processed_session_path(data_root, row))
    if event == "lick_bout_offset" and event not in session:
        if "lick_bout_onset" in session and "lick_bout_duration" in session:
            session[event] = np.asarray(session["lick_bout_onset"], dtype=float) + np.asarray(
                session["lick_bout_duration"], dtype=float
            )
    signal_key = {
        "dff": f"dff_ch{selected_channel}",
        "raw465": f"photometry_465_ch{selected_channel}",
    }[signal]
    time_key = f"photo_time_465_ch{selected_channel}"
    missing = {signal_key, time_key, event}.difference(session)
    if missing:
        raise ValueError(
            f"{processed_session_path(data_root, row)} is missing FLMM keys: "
            f"{sorted(missing)}"
        )
    return (
        np.asarray(session[time_key], dtype=float),
        np.asarray(session[signal_key], dtype=float),
        np.asarray(session[event], dtype=float),
        signal_key,
    )


def _lifetime_trial_source(workflow, row, data_root, signal, event):
    session, _ = load_processed_lifetime_session(workflow, row, data_root)
    continuous = session.continuous[signal]
    return (
        continuous.timestamps,
        continuous.values,
        _lifetime_event_times(session, event),
        signal,
    )


def build_flmm_dataset(
    workflow: str,
    rows: list[dict],
    data_root: str | Path,
    *,
    signal: str,
    event: str,
    window: tuple[float, float] = (-5.0, 20.0),
    dt: float = 0.1,
    normalization: str = "zscore",
    baseline: tuple[float, float] = (-5.0, 0.0),
    channel: str | int = "manifest",
) -> FunctionalTrialDataset:
    """Build one complete trial-by-time dataset across any maintained workflow."""

    if workflow not in {"conventional", "fluopulse", "iflip3"}:
        raise ValueError("workflow must be conventional, fluopulse, or iflip3")
    if workflow == "conventional":
        if signal not in CONVENTIONAL_FLMM_SIGNALS:
            raise ValueError(f"Unsupported conventional FLMM signal: {signal}")
        if event not in CONVENTIONAL_FLMM_EVENTS:
            raise ValueError(f"Unsupported conventional FLMM event: {event}")
    else:
        if signal not in LIFETIME_SIGNALS[workflow]:
            raise ValueError(f"Unsupported {workflow} FLMM signal: {signal}")
        if event not in LIFETIME_EVENTS:
            raise ValueError(f"Unsupported lifetime FLMM event: {event}")
    if not rows:
        raise ValueError("At least one manifest row is required")

    session_numbers = _session_numbers(rows)
    matrices: list[np.ndarray] = []
    mice: list[str] = []
    sessions: list[str] = []
    trial_indices: list[int] = []
    alignment_times: list[float] = []
    groups: list[str] = []
    conditions: list[str] = []
    session_index: list[int] = []
    exclusions: list[dict[str, Any]] = []
    time_grid = None
    resolved_signals = set()

    for row in rows:
        mouse = str(row["mouse"])
        date = str(row["date"])
        run = int(row["run"])
        session_id = f"{mouse}_{date}_run{run:03d}"
        if workflow == "conventional":
            source_time, source_values, events, resolved_signal = _conventional_trial_source(
                row, data_root, signal, event, channel
            )
        else:
            source_time, source_values, events, resolved_signal = _lifetime_trial_source(
                workflow, row, data_root, signal, event
            )
        resolved_signals.add(resolved_signal)
        peri_time, trials, valid = extract_perievent_trials(
            source_time,
            source_values,
            events,
            window=window,
            dt=dt,
            require_complete=True,
        )
        if time_grid is None:
            time_grid = peri_time
        elif not np.array_equal(time_grid, peri_time):
            raise ValueError("Sessions produced inconsistent FLMM time grids")
        valid_set = set(int(index) for index in valid)
        for event_index in range(events.size):
            if event_index not in valid_set:
                exclusions.append(
                    {
                        "mouse": mouse,
                        "session": session_id,
                        "event_index": event_index + 1,
                        "reason": "incomplete_window",
                    }
                )
        if trials.shape[0] == 0:
            continue
        normalized = normalize_trials(peri_time, trials, normalization, baseline)
        for local_index, event_index in enumerate(valid):
            trial = normalized[local_index]
            if not np.all(np.isfinite(trial)):
                exclusions.append(
                    {
                        "mouse": mouse,
                        "session": session_id,
                        "event_index": int(event_index) + 1,
                        "reason": "nonfinite_after_normalization",
                    }
                )
                continue
            matrices.append(trial)
            mice.append(mouse)
            sessions.append(session_id)
            trial_indices.append(int(event_index) + 1)
            alignment_times.append(float(events[event_index]))
            groups.append(str(row.get("group", "") or "unspecified"))
            conditions.append(str(row.get("condition", "") or "unspecified"))
            session_index.append(session_numbers[(mouse, date, run)])

    if not matrices or time_grid is None:
        raise ValueError("No complete, finite trials were available for FLMM analysis")

    return FunctionalTrialDataset(
        time=time_grid,
        outcome=np.vstack(matrices),
        mouse=np.asarray(mice),
        session=np.asarray(sessions),
        trial_index=np.asarray(trial_indices),
        alignment_time=np.asarray(alignment_times),
        scalar_covariates={
            "group": np.asarray(groups),
            "condition": np.asarray(conditions),
            "trial_z": _zscore_within(trial_indices, sessions),
            "session_z": _zscore(session_index),
        },
        exclusions=exclusions,
        metadata={
            "workflow": workflow,
            "signal": signal,
            "resolved_signals": sorted(resolved_signals),
            "event": event,
            "window": list(window),
            "dt": float(dt),
            "normalization": normalization,
            "baseline": list(baseline),
            "complete_windows_required": True,
        },
    )


def flmm_formula(dataset: FunctionalTrialDataset, effect: str) -> str:
    """Return a conservative, inspectable model formula for one guided preset."""

    if effect not in FLMM_EFFECTS:
        raise ValueError(f"effect must be one of {FLMM_EFFECTS}")
    if dataset.n_mice < 2:
        raise ValueError("FLMM requires at least two mice for a mouse random intercept")
    if effect in {"condition", "group"}:
        values = dataset.scalar_covariates[effect].astype(str)
        levels = np.unique(values)
        if "unspecified" in levels:
            raise ValueError(
                f"The {effect} preset requires a populated {effect} value for every "
                "retained session"
            )
        if levels.size < 2:
            raise ValueError(f"The {effect} preset requires at least two levels")
        mice_per_level = {
            level: np.unique(dataset.mouse[values == level]).size for level in levels
        }
        under_replicated = [
            level for level, count in mice_per_level.items() if count < 2
        ]
        if under_replicated:
            raise ValueError(
                f"Every {effect} level requires at least two mice; under-replicated "
                f"levels: {under_replicated}"
            )
    fixed_effects = [effect] if effect in {"condition", "group"} else []
    trial_z = np.asarray(dataset.scalar_covariates["trial_z"], dtype=float)
    session_z = np.asarray(dataset.scalar_covariates["session_z"], dtype=float)
    if np.ptp(trial_z) > 0:
        fixed_effects.append("trial_z")
    elif effect == "progress":
        raise ValueError(
            "The progress preset requires variation in within-session trial order"
        )
    if np.ptp(session_z) > 0:
        fixed_effects.append("session_z")
    return f"outcome ~ {' + '.join(fixed_effects)} + (1 | mouse)"


def _write_dataset_csv(dataset: FunctionalTrialDataset, path: Path) -> None:
    outcome_names = [f"outcome.{index + 1}" for index in range(dataset.n_timepoints)]
    fieldnames = [
        "mouse",
        "session",
        "trial_index",
        "alignment_time",
        *dataset.scalar_covariates,
        *outcome_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index in range(dataset.n_trials):
            row: dict[str, Any] = {
                "mouse": dataset.mouse[index],
                "session": dataset.session[index],
                "trial_index": int(dataset.trial_index[index]),
                "alignment_time": float(dataset.alignment_time[index]),
            }
            row.update(
                {
                    name: values[index]
                    for name, values in dataset.scalar_covariates.items()
                }
            )
            row.update(
                dict(zip(outcome_names, dataset.outcome[index], strict=True))
            )
            writer.writerow(row)


def prepare_flmm_analysis(
    dataset: FunctionalTrialDataset,
    output_dir: str | Path,
    *,
    effect: str,
) -> list[Path]:
    """Freeze the model input, exclusions, time grid, formula, and preflight report."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    formula = flmm_formula(dataset, effect)
    _clear_previous_flmm_results(destination)
    input_path = destination / "flmm_input.csv"
    time_path = destination / "flmm_time.csv"
    exclusions_path = destination / "flmm_exclusions.csv"
    specification_path = destination / "flmm_specification.json"
    preflight_path = destination / "flmm_preflight.json"
    _write_dataset_csv(dataset, input_path)
    with time_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time"])
        writer.writerows([[float(value)] for value in dataset.time])
    exclusion_fields = ("mouse", "session", "event_index", "reason")
    with exclusions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=exclusion_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(dataset.exclusions)
    categorical_levels = {
        name: sorted({str(value) for value in values})
        for name, values in dataset.scalar_covariates.items()
        if values.dtype.kind in "OUSb"
    }
    specification_path.write_text(
        json.dumps(
            {
                **dataset.metadata,
                "effect": effect,
                "formula": formula,
                "levels": categorical_levels,
                "reference_levels": {
                    name: levels[0]
                    for name, levels in categorical_levels.items()
                    if levels
                },
                "inference": "analytic pointwise and joint 95% confidence bands",
                "interpretation": "associational; mouse is the random-effect unit",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    preflight_path.write_text(json.dumps(dataset.summary(), indent=2), encoding="utf-8")
    return [input_path, time_path, exclusions_path, specification_path, preflight_path]


def _find_rscript(rscript: str | Path | None = None) -> str | None:
    """Find Rscript from an override, PATH, or standard Windows installations."""

    if rscript is not None:
        return str(rscript)
    executable = shutil.which("Rscript")
    if executable:
        return executable

    candidates: list[Path] = []
    r_home = os.environ.get("R_HOME")
    if r_home:
        candidates.append(Path(r_home))
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if not base:
            continue
        candidates.append(
            Path(base) / ("Programs" if variable == "LOCALAPPDATA" else "") / "R"
        )
    if os.name == "nt":
        try:
            import winreg

            for hive, key in (
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\R-core\R"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\R-core\R"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\R-core\R"),
            ):
                try:
                    with winreg.OpenKey(hive, key) as handle:
                        candidates.append(Path(winreg.QueryValueEx(handle, "InstallPath")[0]))
                except OSError:
                    continue
        except ImportError:
            pass

    checked: set[Path] = set()
    for root in candidates:
        possible_roots = [root]
        if root.is_dir():
            possible_roots.extend(sorted(root.glob("R-*"), reverse=True))
        for possible_root in possible_roots:
            for relative in (Path("bin/Rscript.exe"), Path("bin/x64/Rscript.exe")):
                path = possible_root / relative
                if path not in checked and path.is_file():
                    return str(path)
                checked.add(path)
    return None


def flmm_runtime_status(rscript: str | Path | None = None) -> dict[str, Any]:
    """Report whether Rscript and the fastFMM package can be invoked."""

    executable = _find_rscript(rscript)
    if not executable:
        return {"ready": False, "rscript": None, "fastfmm_version": None}
    process = subprocess.run(
        [
            executable,
            "--vanilla",
            "-e",
            "cat(if (requireNamespace('fastFMM', quietly=TRUE)) "
            "as.character(packageVersion('fastFMM')) else '')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    version = process.stdout.strip() if process.returncode == 0 else ""
    return {
        "ready": bool(version),
        "rscript": executable,
        "fastfmm_version": version or None,
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "coefficient"


def _clear_previous_flmm_results(destination: Path) -> None:
    """Remove only generated fit products that would be stale after preparation."""

    generated = {
        "flmm_coefficients.csv",
        "flmm_fit.rds",
        "flmm_backend_info.csv",
        "flmm_backend.log",
        "flmm_mouse_random_effects.csv",
        "flmm_fitted_responses.csv",
        "flmm_fitted_responses.png",
        "flmm_fitted_responses.svg",
        "flmm_mouse_deviations.png",
        "flmm_mouse_deviations.svg",
    }
    previous_coefficients = destination / "flmm_coefficients.csv"
    if previous_coefficients.is_file():
        try:
            terms = {row["term"] for row in _coefficient_rows(previous_coefficients)}
        except (csv.Error, KeyError, OSError):
            terms = set()
        for term in terms:
            generated.update(
                {f"flmm_{_safe_name(term)}.png", f"flmm_{_safe_name(term)}.svg"}
            )
    for name in generated:
        path = destination / name
        if path.is_file():
            path.unlink()


def _coefficient_rows(results_path: str | Path) -> list[dict[str, str]]:
    with Path(results_path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _term_title(term: str, specification: dict[str, Any]) -> str:
    if term.casefold() in {"(intercept)", "intercept"}:
        effect = specification.get("effect", "")
        reference = specification.get("reference_levels", {}).get(effect)
        suffix = f": {reference}" if reference else ""
        return f"Reference response{suffix}"
    if term == "trial_z":
        return "Within-session trial progression effect"
    if term == "session_z":
        return "Across-session progression adjustment"
    effect = str(specification.get("effect", ""))
    reference = specification.get("reference_levels", {}).get(effect)
    if effect and term.startswith(effect):
        comparison = term[len(effect) :]
        suffix = f" vs {reference}" if reference else ""
        return f"{effect.title()} difference: {comparison}{suffix}"
    return term


def plot_flmm_coefficients(results_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Plot exported coefficient curves without requiring R plotting packages."""

    import matplotlib.pyplot as plt

    rows = _coefficient_rows(results_path)
    specification_path = Path(output_dir) / "flmm_specification.json"
    specification = (
        json.loads(specification_path.read_text(encoding="utf-8"))
        if specification_path.is_file()
        else {}
    )
    terms = list(dict.fromkeys(row["term"] for row in rows))
    outputs = []
    for term in terms:
        selected = [row for row in rows if row["term"] == term]
        time = np.asarray([float(row["time"]) for row in selected])
        estimate = np.asarray([float(row["estimate"]) for row in selected])
        lower_point = np.asarray([float(row["pointwise_lower"]) for row in selected])
        upper_point = np.asarray([float(row["pointwise_upper"]) for row in selected])
        lower_joint = np.asarray([float(row["joint_lower"]) for row in selected])
        upper_joint = np.asarray([float(row["joint_upper"]) for row in selected])
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.fill_between(time, lower_joint, upper_joint, color="0.75", label="Joint 95% CI")
        ax.fill_between(
            time, lower_point, upper_point, color="0.45", alpha=0.45, label="Pointwise 95% CI"
        )
        ax.plot(time, estimate, color="black", linewidth=1.5, label="Estimate")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set(
            xlabel="Time from alignment (s)",
            ylabel="Coefficient",
            title=_term_title(term, specification),
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = Path(output_dir) / f"flmm_{_safe_name(term)}.{suffix}"
            fig.savefig(path, dpi=300)
            outputs.append(path)
        plt.close(fig)
    return outputs


def plot_flmm_fitted_responses(
    results_path: str | Path,
    specification_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Reconstruct understandable level or progression curves from fixed effects."""

    import matplotlib.pyplot as plt

    rows = _coefficient_rows(results_path)
    specification = json.loads(Path(specification_path).read_text(encoding="utf-8"))
    by_term = {
        term: [row for row in rows if row["term"] == term]
        for term in dict.fromkeys(row["term"] for row in rows)
    }
    intercept_term = next(
        (
            term
            for term in by_term
            if term.casefold() in {"(intercept)", "intercept"}
        ),
        None,
    )
    if intercept_term is None:
        return []
    intercept_rows = by_term[intercept_term]
    time = np.asarray([float(row["time"]) for row in intercept_rows])
    intercept = np.asarray([float(row["estimate"]) for row in intercept_rows])
    effect = str(specification["effect"])
    curves: list[tuple[str, np.ndarray]] = []
    if effect in {"condition", "group"}:
        levels = list(specification.get("levels", {}).get(effect, []))
        if not levels:
            return []
        curves.append((f"{effect.title()}: {levels[0]} (reference)", intercept))
        contrast_terms = [term for term in by_term if term.startswith(effect)]
        for level, term in zip(levels[1:], contrast_terms, strict=False):
            contrast = np.asarray(
                [float(row["estimate"]) for row in by_term[term]], dtype=float
            )
            curves.append((f"{effect.title()}: {level}", intercept + contrast))
    else:
        if "trial_z" not in by_term:
            return []
        slope = np.asarray(
            [float(row["estimate"]) for row in by_term["trial_z"]], dtype=float
        )
        curves = [
            ("Earlier trials (−1 SD)", intercept - slope),
            ("Middle trial", intercept),
            ("Later trials (+1 SD)", intercept + slope),
        ]

    destination = Path(output_dir)
    table_path = destination / "flmm_fitted_responses.csv"
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["curve", "time", "estimate"])
        for label, values in curves:
            writer.writerows(
                [label, float(current_time), float(value)]
                for current_time, value in zip(time, values, strict=True)
            )

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for label, values in curves:
        axis.plot(time, values, linewidth=2, label=label)
    axis.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    axis.axvline(0, color="black", linestyle=":", linewidth=0.8)
    axis.set(
        xlabel="Time from alignment (s)",
        ylabel="Modeled response",
        title="Modeled response curves",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    outputs = [table_path]
    for suffix in ("png", "svg"):
        path = destination / f"flmm_fitted_responses.{suffix}"
        figure.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(figure)
    return outputs


def plot_flmm_mouse_deviations(
    random_effects_path: str | Path, output_dir: str | Path
) -> list[Path]:
    """Plot each mouse's estimated deviation from the population response."""

    import matplotlib.pyplot as plt

    with Path(random_effects_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return []
    mice = list(dict.fromkeys(row["mouse"] for row in rows))
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for mouse in mice:
        selected = [row for row in rows if row["mouse"] == mouse]
        axis.plot(
            [float(row["time"]) for row in selected],
            [float(row["deviation"]) for row in selected],
            linewidth=1,
            alpha=0.7,
            label=mouse,
        )
    axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
    axis.axvline(0, color="black", linestyle=":", linewidth=0.8)
    axis.set(
        xlabel="Time from alignment (s)",
        ylabel="Mouse deviation",
        title="Mouse-specific deviations from the population response",
    )
    if len(mice) <= 20:
        axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    outputs = []
    for suffix in ("png", "svg"):
        path = Path(output_dir) / f"flmm_mouse_deviations.{suffix}"
        figure.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(figure)
    return outputs


def run_prepared_flmm(
    output_dir: str | Path,
    *,
    rscript: str | Path | None = None,
    cores: int = 1,
) -> list[Path]:
    """Fit a prepared FLMM through the isolated Rscript backend."""

    destination = Path(output_dir)
    status = flmm_runtime_status(rscript)
    if not status["ready"]:
        raise RuntimeError(
            "FLMM fitting requires Rscript and the R package fastFMM. "
            "The prepared CSV and specification remain available for inspection."
        )
    specification = json.loads((destination / "flmm_specification.json").read_text())
    script = Path(__file__).with_name("r") / "fit_flmm.R"
    log_path = destination / "flmm_backend.log"
    process = subprocess.run(
        [
            str(status["rscript"]),
            "--vanilla",
            str(script),
            str(destination / "flmm_input.csv"),
            str(destination / "flmm_time.csv"),
            str(destination),
            specification["formula"],
            str(max(1, int(cores))),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"fastFMM failed; inspect {log_path}")
    results_path = destination / "flmm_coefficients.csv"
    if not results_path.is_file():
        raise RuntimeError("fastFMM completed without exporting coefficient results")
    outputs = [
        results_path,
        destination / "flmm_fit.rds",
        destination / "flmm_backend_info.csv",
        log_path,
        *plot_flmm_coefficients(results_path, destination),
        *plot_flmm_fitted_responses(
            results_path,
            destination / "flmm_specification.json",
            destination,
        ),
    ]
    random_effects_path = destination / "flmm_mouse_random_effects.csv"
    if random_effects_path.is_file():
        outputs.extend(plot_flmm_mouse_deviations(random_effects_path, destination))
        outputs.append(random_effects_path)
    return outputs


__all__ = [
    "CONVENTIONAL_FLMM_EVENTS",
    "CONVENTIONAL_FLMM_SIGNALS",
    "FLMM_EFFECTS",
    "build_flmm_dataset",
    "flmm_formula",
    "flmm_runtime_status",
    "plot_flmm_coefficients",
    "plot_flmm_fitted_responses",
    "plot_flmm_mouse_deviations",
    "prepare_flmm_analysis",
    "run_prepared_flmm",
]
