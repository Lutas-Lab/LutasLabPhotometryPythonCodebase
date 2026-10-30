"""Browser interface for the Lutas Lab photometry workflows."""

from __future__ import annotations

import csv
import io
import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lutaslab_photometry.gui_workflows import (
    MANIFEST_COLUMNS,
    build_behavior_glm_command,
    build_lifetime_command,
    build_preprocess_command,
    build_psth_command,
    build_saved_psth_statistics_command,
    command_output_directories,
    discover_manifest_files,
    discover_psth_result_directories,
    display_command,
    manifest_csv_text,
    manifest_display_label,
    next_analysis_run_directory,
    next_manifest_path,
    normalize_manifest_rows,
    replace_command_manifest,
    run_command,
    safe_analysis_run_label,
    snapshot_manifest,
    suggest_psth_run_label,
    workflow_analysis_directory,
    write_manifest,
    write_manifest_metadata,
)
from lutaslab_photometry.heatmap_ordering import (
    HEATMAP_SORT_LABELS,
    default_heatmap_sort_window,
)
from lutaslab_photometry.lifetime_workflows import (
    IFLIP3_FIT_PARAMETER_DEFAULTS,
    LIFETIME_EVENTS,
    LIFETIME_SIGNALS,
    apply_iflip3_calibration,
    calibrate_iflip3_sessions,
    discover_lifetime_paths,
    estimate_iflip3_binning_sensitivity,
    iflip3_residual_summary,
    lifetime_manifest_columns,
    lifetime_preprocessing_parameters,
    lifetime_processed_path,
    load_iflip3_calibration,
    next_iflip3_calibration_path,
    normalize_lifetime_manifest_rows,
    preview_iflip3_fit,
    save_iflip3_calibration,
    write_lifetime_manifest,
)
from lutaslab_photometry.processed_provenance import classify_processed_provenance
from lutaslab_photometry.psth_results import load_saved_psth_timecourses
from lutaslab_photometry.psth_statistics import metrics_for_signal_type
from lutaslab_photometry.qc_reporting import parse_qc_warning_line
from lutaslab_photometry.save_sessiondata import conventional_preprocessing_parameters
from lutaslab_photometry.session_manifest import processed_session_path
from lutaslab_photometry.session_trace import (
    load_conventional_session_trace,
    load_lifetime_session_trace,
    plot_session_trace,
    save_session_trace,
)
from lutaslab_photometry.update_check import (
    GITHUB_REPOSITORY_URL,
    UpdateStatus,
    check_for_update,
    local_git_commit,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRIAL_CLASS_LABELS = {
    "all": "All cue trials",
    "cue_lick": "Licked during cue",
    "post_cue_lick": "Licked after cue",
    "cue_only": "During cue only",
    "post_only": "After cue only",
    "cue_and_post": "During and after cue",
    "cue_miss": "No lick during or after cue",
}
CONVENTIONAL_PHOTOMETRY_SIGNAL_LABELS = {
    "dff": "IRLS ΔF/F",
    "raw465": "465 fluorescence",
    "raw405": "405 isosbestic fluorescence (QC)",
    "fitted405": "Fitted 405 control (QC)",
}
LIFETIME_SIGNAL_LABELS = {
    "mpet": "MPET — standard correction (background if provided + afterpulse)",
    "mpet_afterpulse_only": "MPET — afterpulse corrected only",
    "mpet_uncorrected": "MPET — uncorrected",
    "long_lifetime_fraction": "Long-lifetime fraction — standard correction",
    "short_component_counts": "Short-component counts — standard correction",
    "long_component_counts": "Long-component counts — standard correction",
    "long_lifetime_fraction_afterpulse_only": (
        "Long-lifetime fraction — afterpulse corrected only"
    ),
    "short_component_counts_afterpulse_only": (
        "Short-component counts — afterpulse corrected only"
    ),
    "long_component_counts_afterpulse_only": ("Long-component counts — afterpulse corrected only"),
    "raw_intensity": "Raw photon intensity",
}
LIFETIME_GLM_RESPONSE_HELP = {
    "fluopulse": (
        "Response predicted by the GLM. tau is fluorescence lifetime (ps); "
        "amplitude is fitted fluorescence amplitude; fit_r_square is the vendor-fit "
        "quality trace and is primarily a QC response."
    ),
    "iflip3": (
        "Response predicted by the GLM. Corrected signals use the selected measured "
        "background plus the configured afterpulse correction. Afterpulse-only signals "
        "omit measured-background subtraction, and uncorrected MPET omits both. Fitted "
        "component comparisons reuse the same decay-shape calibration."
    ),
}
DEFAULT_ROWS = [
    {
        "mouse": "DK21",
        "date": "230704",
        "run": 1,
        "group": "control",
        "condition": "naive",
        "channel": 1,
    }
]


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _cached_update_status(project_root: str) -> UpdateStatus:
    """Limit the public GitHub check to once every six hours."""

    return check_for_update(project_root)


def _show_update_status() -> None:
    status = _cached_update_status(str(PROJECT_ROOT))
    st.divider()
    st.caption("Software version")
    if status.local_commit:
        st.code(status.local_commit[:7], language=None)
    if status.state == "up_to_date":
        st.success("Up to date with GitHub main.")
    elif status.state == "update_available":
        count = status.commits_behind
        detail = f" ({count} new commit{'s' if count != 1 else ''})" if count else ""
        st.warning(f"Update available{detail}.")
        st.caption(
            "Close the GUI, open this repository in GitHub Desktop, select main, "
            "click Pull origin, and then run install_gui.bat again."
        )
    elif status.state == "local_ahead":
        st.info("This checkout is newer than GitHub main.")
    elif status.state == "diverged":
        st.warning(
            "This checkout differs from GitHub main. Ask the repository maintainer before updating."
        )
    elif status.state == "not_a_clone":
        st.info(
            "Version checking is unavailable because this appears to be a ZIP copy. "
            "Clone the repository with GitHub Desktop to enable it."
        )
    else:
        st.caption("Could not check GitHub. The GUI can still be used normally.")
    st.link_button("View repository on GitHub", GITHUB_REPOSITORY_URL, width="stretch")


def _uploaded_rows(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _manifest_text(workflow, rows):
    """Serialize one workflow's current editor rows for dirty-state checks."""

    if workflow == "conventional":
        return manifest_csv_text(rows)
    validated = normalize_lifetime_manifest_rows(workflow, rows)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=lifetime_manifest_columns(workflow),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(validated)
    return output.getvalue()


def _write_workflow_manifest(workflow, path, rows):
    if workflow == "conventional":
        return write_manifest(path, rows)
    return write_lifetime_manifest(workflow, path, rows)


def _normalized_manifest_row_count(workflow, rows):
    if workflow == "conventional":
        return len(normalize_manifest_rows(rows))
    return len(normalize_lifetime_manifest_rows(workflow, rows))


def _save_manifest_version(
    workflow,
    rows,
    manifest_dir,
    display_name,
    *,
    parent_manifest=None,
):
    """Create a timestamped, descriptive manifest and provenance sidecar."""

    path = next_manifest_path(manifest_dir, display_name)
    saved = _write_workflow_manifest(workflow, path, rows)
    write_manifest_metadata(
        saved,
        workflow=workflow,
        display_name=display_name,
        row_count=_normalized_manifest_row_count(workflow, rows),
        code_commit=local_git_commit(PROJECT_ROOT),
        parent_manifest=parent_manifest,
    )
    return saved


def _manifest_matches_rows(workflow, manifest_path, rows):
    path = Path(manifest_path).expanduser() if manifest_path else None
    if path is None or not path.is_file():
        return False
    try:
        return path.read_text(encoding="utf-8-sig") == _manifest_text(workflow, rows)
    except (OSError, ValueError):
        return False


def _save_manifest_for_run(
    workflow,
    rows,
    manifest_dir,
    display_name,
    active_manifest,
):
    """Reuse an unchanged active manifest or save edited rows as a new version."""

    if _manifest_matches_rows(workflow, active_manifest, rows):
        return Path(active_manifest).expanduser().resolve(), False
    saved = _save_manifest_version(
        workflow,
        rows,
        manifest_dir,
        display_name,
        parent_manifest=active_manifest or None,
    )
    return saved, True


def _read_manifest_file(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _manifest_workspace_sidebar(workflow, label, *, legacy_paths=()):
    """Render common workspace settings and return manifest-library state."""

    workspace = st.text_input(
        "Analysis workspace",
        value=str(workflow_analysis_directory(PROJECT_ROOT, workflow)),
        key=f"{workflow}_analysis_workspace",
        help="Contains manifests and downstream analysis folders for this workflow.",
    )
    manifest_dir = Path(workspace).expanduser() / "manifests"
    active_key = f"{workflow}_active_manifest"
    if active_key not in st.session_state:
        first_legacy = next((str(Path(path)) for path in legacy_paths if Path(path).is_file()), "")
        st.session_state[active_key] = first_legacy
    manifest_name = st.text_input(
        "Manifest name",
        value="sessions",
        key=f"{workflow}_manifest_name",
        help=(
            "Use a descriptive cohort or experiment name. New versions receive a "
            "timestamp and never silently replace an existing manifest."
        ),
    )
    active = st.session_state.get(active_key, "")
    st.caption(
        f"Active manifest: {active}" if active else f"No saved {label} manifest is active yet."
    )
    return Path(workspace).expanduser(), manifest_dir, manifest_name, active_key


def _manifest_library_panel(
    workflow,
    rows_state_key,
    manifest_dir,
    active_key,
    *,
    legacy_paths=(),
):
    """Render saved-manifest discovery and load the selected CSV on request."""

    candidates = discover_manifest_files(manifest_dir, legacy_paths=legacy_paths)
    if not candidates:
        st.caption("No saved manifests were found in this workspace yet.")
        return
    selected = st.selectbox(
        "Saved manifests",
        candidates,
        format_func=manifest_display_label,
        key=f"{workflow}_saved_manifest_selection",
        help="Legacy sessions.csv files remain available and are never moved or deleted.",
    )
    if st.button(
        "Load selected manifest",
        key=f"{workflow}_load_saved_manifest",
        width="stretch",
    ):
        try:
            rows = _read_manifest_file(selected)
            if workflow == "conventional":
                rows = normalize_manifest_rows(rows)
            else:
                rows = normalize_lifetime_manifest_rows(workflow, rows)
        except (OSError, ValueError) as error:
            st.error(f"Could not load manifest: {error}")
        else:
            st.session_state[rows_state_key] = rows
            st.session_state[active_key] = str(Path(selected).resolve())
            editor_key = (
                "manifest_editor" if workflow == "conventional" else f"{workflow}_manifest_editor"
            )
            st.session_state.pop(editor_key, None)
            st.rerun()


def _records(editor_value):
    if hasattr(editor_value, "to_dict"):
        return editor_value.to_dict(orient="records")
    return list(editor_value)


def _manifest_label_options(rows, field):
    """Return distinct nonblank manifest labels with stable display casing."""

    values = {}
    for row in rows:
        raw = row.get(field, "")
        if raw is None or pd.isna(raw):
            continue
        value = str(raw).strip()
        if value:
            values.setdefault(value.casefold(), value)
    return [values[key] for key in sorted(values)]


def _analysis_subset_controls(rows, key_prefix):
    """Render optional group and condition filters derived from manifest rows."""

    group_options = _manifest_label_options(rows, "group")
    condition_options = _manifest_label_options(rows, "condition")
    left, right = st.columns(2)
    with left:
        group_filter = st.selectbox(
            "Group to analyze",
            [None, *group_options],
            format_func=lambda value: "All groups" if value is None else value,
            key=f"{key_prefix}_group_filter",
            help=(
                "Restrict this analysis to one manifest group. All groups includes every "
                "row that also passes the condition filter."
            ),
        )
    with right:
        condition_filter = st.selectbox(
            "Condition to analyze",
            [None, *condition_options],
            format_func=lambda value: "All conditions" if value is None else value,
            key=f"{key_prefix}_condition_filter",
            help=(
                "Restrict this analysis to one manifest condition. All conditions includes "
                "every row that also passes the group filter."
            ),
        )
    return group_filter, condition_filter


def _relative_path_text(path: Path, data_root: str | Path) -> str:
    """Prefer portable manifest paths relative to the selected data root."""

    root = Path(data_root).expanduser()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _path_picker(workflow, field, label, candidates, data_root, *, optional=False):
    choices = [_relative_path_text(path, data_root) for path in candidates]
    manual_label = "Enter a different path..."
    options = choices + (["(none)"] if optional else []) + [manual_label]
    selected = st.selectbox(
        label,
        options,
        key=f"{workflow}_discovery_{field}_choice",
    )
    if selected == manual_label:
        return st.text_input(
            f"{label} (editable path)",
            key=f"{workflow}_discovery_{field}_manual",
            placeholder="Path relative to the data root, or an absolute path",
        ).strip()
    return "" if selected == "(none)" else selected


def _preview_command(command):
    st.code(display_command(command), language="powershell")
    st.info("Command preview only; no files were changed and no analysis was run.")


def _iflip3_fit_controls() -> dict[str, dict[str, float | str]]:
    """Render aggregate-decay fitting controls and return backend-ready settings."""

    labels = {
        "tau1": "Short lifetime τ1 (ns)",
        "tau2": "Long lifetime τ2 (ns)",
        "t0": "Decay start t0 (ns)",
        "sigma": "IRF width σ (ns)",
        "background": "Residual background (counts/bin)",
    }
    settings = {}
    headings = st.columns((1.8, 1, 1, 1))
    fit_headings = ("Parameter", "Mode", "Value/start", "Bounds")
    for column, heading in zip(headings, fit_headings, strict=True):
        column.caption(heading)
    for name, defaults in IFLIP3_FIT_PARAMETER_DEFAULTS.items():
        parameter, mode_column, value_column, bounds_column = st.columns((1.8, 1, 1, 1))
        parameter.write(labels[name])
        mode = mode_column.selectbox(
            f"{labels[name]} mode",
            ("Auto", "Fixed", "Bounded"),
            index=("auto", "fixed", "bounded").index(str(defaults["mode"])),
            key=f"iflip3_fit_{name}_mode",
            label_visibility="collapsed",
        ).lower()
        value = value_column.number_input(
            f"{labels[name]} value",
            value=float(defaults["value"]),
            format="%.5g",
            key=f"iflip3_fit_{name}_value",
            label_visibility="collapsed",
        )
        lower = float(defaults["lower"])
        upper = float(defaults["upper"])
        if mode == "bounded":
            bound_columns = bounds_column.columns(2)
            lower = bound_columns[0].number_input(
                f"{labels[name]} minimum",
                value=lower,
                format="%.5g",
                key=f"iflip3_fit_{name}_lower",
                label_visibility="collapsed",
            )
            upper = bound_columns[1].number_input(
                f"{labels[name]} maximum",
                value=upper,
                format="%.5g",
                key=f"iflip3_fit_{name}_upper",
                label_visibility="collapsed",
            )
        else:
            bounds_column.caption("—")
        settings[name] = {
            "mode": mode,
            "value": float(value),
            "lower": float(lower),
            "upper": float(upper),
        }
    st.caption(
        "Auto fits from the entered starting value. Fixed holds the value constant. "
        "Bounded fits only within the entered minimum and maximum."
    )
    return settings


def _show_iflip3_fit_preview(preview) -> None:
    """Display fit curves, structured residuals, parameters, and quality metrics."""

    import matplotlib.pyplot as plt

    figure, (fit_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
        constrained_layout=True,
    )
    fit_axis.plot(preview.lifetime_time, preview.aggregate_decay, color="black", label="Data")
    fit_axis.plot(preview.lifetime_time, preview.fitted_decay, color="#d62728", label="Total fit")
    fit_axis.plot(
        preview.lifetime_time,
        preview.short_component + preview.background,
        label="Short component",
    )
    fit_axis.plot(
        preview.lifetime_time,
        preview.long_component + preview.background,
        label="Long component",
    )
    fit_axis.set_ylabel("Aggregate counts")
    fit_axis.legend(frameon=False, ncols=2)
    residual_axis.axhline(0.0, color="0.5", linewidth=1)
    residual_axis.plot(preview.lifetime_time, preview.residuals, color="#4c78a8")
    residual_axis.set_xlabel("Time after excitation pulse (ns)")
    residual_axis.set_ylabel("Residual")
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    metrics = st.columns(6)
    metric_values = (
        ("τ1", f"{preview.lifetimes[0]:.4g} ns"),
        ("τ2", f"{preview.lifetimes[1]:.4g} ns"),
        ("t0", f"{preview.t0:.4g} ns"),
        ("IRF σ", f"{preview.irf_sigma:.4g} ns"),
        ("R²", f"{preview.r_squared:.5f}"),
        ("RMSE", f"{preview.rmse:.4g}"),
    )
    for column, (name, value) in zip(metrics, metric_values, strict=True):
        column.metric(name, value)
    residual_summary = iflip3_residual_summary(preview.aggregate_decay, preview.residuals)
    residual_metrics = st.columns(3)
    residual_values = (
        (
            "Maximum |residual| / peak",
            f"{residual_summary['max_absolute_residual_percent_peak']:.3g}%",
        ),
        (
            "Residual RMSE / peak",
            f"{residual_summary['rmse_percent_peak']:.3g}%",
        ),
        (
            "Absolute residual area",
            f"{residual_summary['absolute_residual_area_percent']:.3g}%",
        ),
    )
    for column, (name, value) in zip(residual_metrics, residual_values, strict=True):
        column.metric(name, value)
    st.caption(f"Residual background: {preview.background:.5g} aggregate counts/bin")


def _show_iflip3_binning_sensitivity(sensitivity) -> None:
    """Display the photon shot-noise tradeoff for a calibrated session."""

    import matplotlib.pyplot as plt

    durations = sensitivity.bin_durations_seconds
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), constrained_layout=True)
    axes[0].plot(
        durations,
        sensitivity.mpet_95_half_width_ps,
        color="#4c78a8",
        marker="o",
    )
    axes[0].set(
        xlabel="Temporal bin (s)",
        ylabel="MPET 95% half-width (ps)",
        title="MPET shot-noise precision",
    )
    axes[1].plot(
        durations,
        sensitivity.long_fraction_95_half_width,
        color="#f58518",
        marker="o",
    )
    axes[1].set(
        xlabel="Temporal bin (s)",
        ylabel="Long-fraction 95% half-width",
        title="Component-fraction precision",
    )
    for axis in axes:
        axis.set_xticks(durations)
        axis.grid(alpha=0.2)
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    photon_summary = ", ".join(
        f"{duration:g} s: {photons:,.0f}"
        for duration, photons in zip(durations, sensitivity.median_sensor_photons, strict=True)
    )
    st.caption(f"Median fitted sensor photons per bin — {photon_summary}.")
    st.caption(
        f"Estimated from {sensitivity.n_bootstrap} Poisson resamples of this "
        "session's calibrated decay. This isolates photon shot noise; it does not "
        "include biological variability, calibration mismatch, or uncertainty in "
        "the background correction."
    )


def _show_iflip3_calibration(calibration) -> None:
    """Display shared parameters and session-level calibration diagnostics."""

    import matplotlib.pyplot as plt
    import numpy as np

    metrics = st.columns(5)
    session_specific_t0 = calibration.schema_version != "1.0"
    values = (
        ("τ1", f"{calibration.lifetimes[0]:.4g} ns"),
        ("τ2", f"{calibration.lifetimes[1]:.4g} ns"),
        (
            "Median t0" if session_specific_t0 else "Shared t0",
            f"{calibration.t0:.4g} ns",
        ),
        ("IRF σ", f"{calibration.irf_sigma:.4g} ns"),
        ("Sessions", str(len(calibration.session_qc))),
    )
    for column, (name, value) in zip(metrics, values, strict=True):
        column.metric(name, value)
    if session_specific_t0:
        st.caption(
            "τ1, τ2, and IRF σ are shared. Each session has its own fixed t0; "
            "residual fit background is fixed at zero."
        )
    else:
        st.caption(
            "Legacy calibration: τ1, τ2, t0, and IRF σ are shared; residual fit "
            "background is fixed at zero."
        )
    flagged_sessions = [
        item["session_id"] for item in calibration.session_qc if item.get("t0_offset_flag")
    ]
    if flagged_sessions:
        st.warning("Large t0 offsets require review: " + ", ".join(flagged_sessions))

    figure, (fit_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
        constrained_layout=True,
    )
    residual_rows = []
    for item in calibration.session_qc:
        observed = np.asarray(item["normalized_decay"], dtype=float)
        fitted = np.asarray(item["fitted_decay"], dtype=float)
        observed_color = "#ff7f0e" if item.get("t0_offset_flag") else "0.7"
        fit_axis.plot(calibration.lifetime_time, observed, color=observed_color, alpha=0.7)
        fit_axis.plot(calibration.lifetime_time, fitted, color="#d62728", alpha=0.25)
        residual_rows.append(np.asarray(item["residuals"], dtype=float))
    fit_axis.set_ylabel("Normalized aggregate decay")
    residual_image = residual_axis.imshow(
        np.asarray(residual_rows),
        aspect="auto",
        interpolation="nearest",
        extent=(
            calibration.lifetime_time[0],
            calibration.lifetime_time[-1],
            len(residual_rows) - 0.5,
            -0.5,
        ),
        cmap="coolwarm",
    )
    residual_axis.set_xlabel("Time after excitation pulse (ns)")
    residual_axis.set_ylabel("Session")
    if len(calibration.session_qc) <= 20:
        residual_axis.set_yticks(range(len(calibration.session_qc)))
        residual_axis.set_yticklabels(
            [item["session_id"] for item in calibration.session_qc], fontsize=7
        )
    figure.colorbar(residual_image, ax=residual_axis, label="Residual")
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    table_rows = []
    for item in calibration.session_qc:
        residual_summary = iflip3_residual_summary(item["normalized_decay"], item["residuals"])
        table_rows.append(
            {
                "session": item["session_id"],
                "measured background": item["measured_background_applied"],
                "R²": item["r_squared"],
                "RMSE": item["rmse_normalized"],
                "max |residual| / peak (%)": residual_summary["max_absolute_residual_percent_peak"],
                "RMSE / peak (%)": residual_summary["rmse_percent_peak"],
                "absolute residual area (%)": residual_summary["absolute_residual_area_percent"],
                "peak time (ns)": item.get("peak_time_ns"),
                "t0 (ns)": item.get("t0_ns", calibration.t0),
                "t0 offset (ns)": item.get("t0_offset_from_median_ns", 0.0),
                "t0 review": item.get("t0_offset_flag", False),
                "short fraction": item["short_fraction_aggregate"],
                "long fraction": item["long_fraction_aggregate"],
            }
        )
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")


def _processed_provenance_panel(
    workflow,
    rows,
    data_root,
    expected_parameters,
) -> None:
    """Let the user inspect one processed session's saved code and settings."""

    with st.expander("Check an existing processed file"):
        try:
            valid_rows = (
                normalize_manifest_rows(rows)
                if workflow == "conventional"
                else normalize_lifetime_manifest_rows(workflow, rows)
            )
        except ValueError:
            valid_rows = []
        if not valid_rows:
            st.caption("Add a complete session row to check its processed file.")
            return
        labels = [
            f"{row['mouse']} · {row['date']} · run {int(row['run']):03d}"
            + (f" · {row['condition']}" if row.get("condition") else "")
            for row in valid_rows
        ]
        selected = st.selectbox(
            "Session",
            range(len(valid_rows)),
            format_func=lambda index: labels[index],
            key=f"{workflow}_provenance_session",
        )
        if not st.button("Check processed-file version", key=f"{workflow}_check_provenance"):
            return
        row = valid_rows[selected]
        try:
            path = (
                processed_session_path(data_root, row)
                if workflow == "conventional"
                else lifetime_processed_path(workflow, row, data_root)
            )
            status = classify_processed_provenance(
                path,
                current_commit=local_git_commit(PROJECT_ROOT),
                expected_schema_version="1.0",
                expected_parameters=expected_parameters,
            )
        except (OSError, ValueError) as error:
            st.error(f"The processed file could not be checked: {error}")
            return
        message = f"{status.summary}. {status.detail}"
        if status.state == "current":
            st.success(message)
        elif status.state in {"missing", "different_commit"}:
            st.info(message)
        elif status.state == "older_schema":
            st.error(message)
        else:
            st.warning(message)
        st.caption(f"Processed file: {path}")
        if status.provenance:
            code = status.provenance.get("code") or {}
            commit = code.get("commit")
            if commit:
                st.code(str(commit)[:12], language=None)
            st.caption(f"Processed: {status.provenance.get('processing_utc', 'unknown time')}")
            st.json(status.provenance, expanded=False)


def _run_command_batch(command, workflow_name, command_labels=None):
    """Run one or more commands with streamed logs and session progress."""

    commands = list(command) if command and isinstance(command[0], (list, tuple)) else [command]
    labels = command_labels or [workflow_name] * len(commands)
    output_lines = deque(maxlen=500)
    output_panel = st.empty()
    progress_panel = st.progress(0.0, text=f"Starting {workflow_name}...")
    qc_records = []

    def show_qc_summary():
        unique = {(record["path"], record["reason"]): record for record in qc_records}
        if not unique:
            return
        records = list(unique.values())
        count = len(records)
        st.warning(
            f"Running data were excluded from {count} "
            f"{'session' if count == 1 else 'sessions'} because the running "
            "files failed quality control. Other photometry and lifetime signals "
            "were processed normally; only running-speed analyses are unavailable "
            "for these sessions."
        )
        with st.expander(f"Affected running files ({count})"):
            for record in records:
                location = record["path"] or "Path unavailable"
                st.markdown(f"- `{location}` — {record['reason']}")

    for command_index, (current_command, label) in enumerate(zip(commands, labels, strict=True)):
        if len(commands) > 1:
            output_lines.append(f"--- {label} ---")

        def show_output(line, _command_index=command_index, _label=label):
            stripped = line.rstrip("\r\n")
            qc_record = parse_qc_warning_line(stripped)
            if qc_record is not None:
                qc_records.append(qc_record)
                return
            fields = stripped.split("\t", maxsplit=3)
            if len(fields) == 4 and fields[0] == "PROGRESS":
                try:
                    current = int(fields[1])
                    total = int(fields[2])
                except ValueError:
                    pass
                else:
                    fraction = current / max(total, 1)
                    overall = (_command_index + fraction) / len(commands)
                    progress_panel.progress(
                        min(overall, 1.0),
                        text=f"{_label}: {current} of {total} — {fields[3]}",
                    )
                    return
            output_lines.append(stripped)
            output_panel.code(
                "\n".join(output_lines) or "(Waiting for console output...)",
                language="text",
            )

        try:
            return_code, output = run_command(current_command, PROJECT_ROOT, show_output)
        except OSError as error:
            st.error(f"{label} could not start: {error}")
            return False
        if not output_lines:
            visible_output = "\n".join(
                line for line in output.splitlines() if parse_qc_warning_line(line) is None
            )
            output_panel.code(visible_output or "(No console output)", language="text")
        if return_code != 0:
            show_qc_summary()
            st.error(f"{label} exited with code {return_code}.")
            return False
        progress_panel.progress(
            (command_index + 1) / len(commands),
            text=f"Finished {label} ({command_index + 1} of {len(commands)} outputs)",
        )
    show_qc_summary()
    st.success(f"{workflow_name} finished successfully.")
    return True


def _run_workflow(
    command,
    editor_rows,
    manifest_dir,
    manifest_name,
    active_manifest_key,
    data_root,
    workflow_name,
    command_labels=None,
):
    data_root_path = Path(data_root).expanduser()
    if not data_root_path.is_dir():
        st.error(f"Raw-data root does not exist or is not a directory: {data_root_path}")
        return False

    active_manifest = st.session_state.get(active_manifest_key, "")
    try:
        saved_path, created = _save_manifest_for_run(
            "conventional",
            editor_rows,
            manifest_dir,
            manifest_name,
            active_manifest,
        )
    except (OSError, ValueError) as error:
        st.error(f"The session manifest was not saved: {error}")
        return False
    st.session_state[active_manifest_key] = str(saved_path)
    command = replace_command_manifest(command, saved_path)
    action = "Saved a new manifest version" if created else "Using active manifest"
    st.success(f"{action}: {saved_path}")
    with st.spinner("Running workflow. Keep this browser tab open..."):
        completed = _run_command_batch(command, workflow_name, command_labels)
    if completed:
        for output_dir in command_output_directories(command):
            snapshot_manifest(saved_path, output_dir)
    return completed


def _run_lifetime_workflow(
    command,
    workflow,
    editor_rows,
    manifest_dir,
    manifest_name,
    active_manifest_key,
    data_root,
    workflow_name,
    command_labels=None,
):
    data_root_path = Path(data_root).expanduser()
    if not data_root_path.is_dir():
        st.error(f"Data root does not exist or is not a directory: {data_root_path}")
        return False
    active_manifest = st.session_state.get(active_manifest_key, "")
    try:
        saved_path, created = _save_manifest_for_run(
            workflow,
            editor_rows,
            manifest_dir,
            manifest_name,
            active_manifest,
        )
    except (OSError, ValueError) as error:
        st.error(f"The session manifest was not saved: {error}")
        return False
    st.session_state[active_manifest_key] = str(saved_path)
    command = replace_command_manifest(command, saved_path)
    action = "Saved a new manifest version" if created else "Using active manifest"
    st.success(f"{action}: {saved_path}")
    with st.spinner("Running workflow. Keep this browser tab open..."):
        completed = _run_command_batch(command, workflow_name, command_labels)
    if completed:
        for output_dir in command_output_directories(command):
            snapshot_manifest(saved_path, output_dir)
    return completed


def _show_glm_results(output_directory, key_prefix):
    """Render saved GLM performance, kernels, and held-out diagnostics."""

    output_directory = Path(output_directory)
    if not output_directory.is_dir():
        st.warning(f"The latest GLM output folder is unavailable: {output_directory}")
        return
    conventional = (output_directory / "forecast_metadata.json").is_file()
    metadata_path = output_directory / (
        "forecast_metadata.json" if conventional else "glm_metadata.json"
    )
    if not metadata_path.is_file():
        st.warning("The selected folder does not contain completed GLM metadata.")
        return
    st.divider()
    st.subheader("Latest GLM results")
    st.code(str(output_directory), language=None)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    st.caption(
        "Performance and traces below are held out from model fitting. Final pooled "
        "kernels use all selected data and describe predictive association, not causality."
    )
    performance_tab, kernels_tab, predictions_tab, tables_tab = st.tabs(
        ["Performance", "Kernels", "Held-out predictions", "Tables"]
    )
    with performance_tab:
        performance_images = sorted(output_directory.glob("*performance.png"))
        for image_path in performance_images:
            st.image(image_path.read_bytes(), width="stretch")
        score_path = output_directory / (
            "forecast_mouse_metrics.csv" if conventional else "glm_mouse_cross_validation.csv"
        )
        if score_path.is_file():
            st.dataframe(pd.read_csv(score_path), hide_index=True, width="stretch")
        if not performance_images and not score_path.is_file():
            st.info("No performance output was found.")
    with kernels_tab:
        kernel_images = sorted(output_directory.glob("*kernels.png"))
        if kernel_images:
            for image_path in kernel_images:
                st.image(image_path.read_bytes(), width="stretch")
        else:
            st.info("No kernel figure was found.")
    with predictions_tab:
        prediction_path = output_directory / (
            "forecast_heldout_predictions.csv" if conventional else "glm_heldout_predictions.csv"
        )
        if not prediction_path.is_file():
            st.info(
                "No held-out prediction table was found. Rerun the GLM with this "
                "software version to create prediction and residual diagnostics."
            )
        else:
            predictions = pd.read_csv(prediction_path)
            if conventional and "model" in predictions:
                models = list(dict.fromkeys(predictions["model"].astype(str)))
                default_model = models.index("combined") if "combined" in models else 0
                selected_model = st.selectbox(
                    "Model",
                    models,
                    index=default_model,
                    key=f"{key_prefix}_prediction_model",
                )
                predictions = predictions[predictions["model"] == selected_model]
            if conventional and "horizon" in predictions:
                horizons = sorted(predictions["horizon"].dropna().unique())
                selected_horizon = st.selectbox(
                    "Forecast horizon (s)",
                    horizons,
                    key=f"{key_prefix}_prediction_horizon",
                )
                predictions = predictions[predictions["horizon"] == selected_horizon]
            sessions = list(dict.fromkeys(predictions["session_id"].astype(str)))
            selected_session = st.selectbox(
                "Session",
                sessions,
                key=f"{key_prefix}_prediction_session",
            )
            selected = predictions[
                predictions["session_id"].astype(str) == selected_session
            ].sort_values("time_s")
            if not selected.empty:
                score = pd.to_numeric(selected["heldout_r2"], errors="coerce").iloc[0]
                residual = pd.to_numeric(selected["residual"], errors="coerce")
                first, second = st.columns(2)
                first.metric("Held-out R²", f"{score:.3f}" if np.isfinite(score) else "n/a")
                second.metric("Residual RMSE", f"{np.sqrt(np.nanmean(residual**2)):.4g}")
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(
                    2,
                    1,
                    figsize=(9, 5.5),
                    sharex=True,
                    gridspec_kw={"height_ratios": (2.2, 1)},
                )
                axes[0].plot(selected["time_s"], selected["actual"], color="black", label="Actual")
                axes[0].plot(
                    selected["time_s"],
                    selected["predicted"],
                    color="C3",
                    label="Held-out prediction",
                )
                axes[0].set_ylabel(metadata.get("ylabel", "Response"))
                axes[0].legend()
                axes[1].plot(selected["time_s"], selected["residual"], color="C0")
                axes[1].axhline(0, color="black", linewidth=0.8)
                axes[1].set(xlabel="Time (s)", ylabel="Residual")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
    with tables_tab:
        table_names = (
            (
                ("Session metrics", "forecast_session_metrics.csv"),
                ("Group metrics", "forecast_group_metrics.csv"),
                ("Session kernels", "forecast_session_kernels.csv"),
            )
            if conventional
            else (
                ("Session cross-validation", "glm_cross_validation.csv"),
                ("Session kernels", "glm_session_kernels.csv"),
                ("Group kernels", "glm_group_kernels.csv"),
            )
        )
        for label, filename in table_names:
            path = output_directory / filename
            if path.is_file():
                with st.expander(label):
                    st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


def _show_psth_summary_results(output_directory, key_prefix):
    """Render completed saved-PSTH statistics and downloads in the GUI."""

    output_directory = Path(output_directory)
    if not output_directory.is_dir():
        st.warning(f"The latest summary output folder is unavailable: {output_directory}")
        return
    st.divider()
    st.subheader("Latest summary results")
    st.caption(
        "Mouse-level values are shown; mice are the inferential sample size. "
        "Average-PSTH metrics and averages of per-trial metrics are kept separate."
    )
    st.code(str(output_directory), language=None)
    metadata_path = output_directory / "psth_statistics_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if metadata is not None:
            response_window = metadata.get("response_window", ["?", "?"])
            st.caption(
                f"{metadata.get('signal', 'signal')} aligned to "
                f"{metadata.get('event', 'event')} · response window "
                f"{response_window[0]} to {response_window[1]} s · metrics: "
                f"{', '.join(metadata.get('metrics', []))}"
            )

    figures_tab, group_tab, tests_tab, trial_tab = st.tabs(
        [
            "Metric plots",
            "Group summaries",
            "Statistical tests",
            "Per-trial metrics",
        ]
    )
    with figures_tab:
        figure_paths = sorted(output_directory.glob("psth_*.png"))
        if not figure_paths:
            st.info("No PNG metric figures were created for this result.")
        else:
            columns = st.columns(2)
            for index, path in enumerate(figure_paths):
                with columns[index % 2]:
                    st.image(
                        path.read_bytes(),
                        caption=path.stem.replace("psth_", "").replace("_", " "),
                        width="stretch",
                    )
    with group_tab:
        group_path = output_directory / "psth_group_summary.csv"
        if group_path.is_file():
            st.dataframe(pd.read_csv(group_path), hide_index=True, width="stretch")
        else:
            st.info("No group summary table was created.")
    with tests_tab:
        tests_path = output_directory / "psth_statistical_tests.csv"
        if tests_path.is_file():
            st.dataframe(pd.read_csv(tests_path), hide_index=True, width="stretch")
        else:
            st.info(
                "No pairwise statistical tests were created. This is expected when "
                "only one group/condition is available."
            )
    with trial_tab:
        trial_group_path = output_directory / "psth_trial_metric_group_summary.csv"
        trial_mouse_path = output_directory / "psth_mouse_trial_metric_summary.csv"
        trial_tests_path = output_directory / "psth_trial_metric_statistical_tests.csv"
        if trial_group_path.is_file():
            st.caption(
                "Each metric was calculated per trial, averaged within session, "
                "then sessions were weighted equally within mouse."
            )
            st.dataframe(pd.read_csv(trial_group_path), hide_index=True, width="stretch")
            if trial_mouse_path.is_file():
                with st.expander("Mouse values derived from individual trials"):
                    st.dataframe(
                        pd.read_csv(trial_mouse_path),
                        hide_index=True,
                        width="stretch",
                    )
            if trial_tests_path.is_file():
                with st.expander("Statistical tests for trial-derived mouse values"):
                    st.dataframe(
                        pd.read_csv(trial_tests_path),
                        hide_index=True,
                        width="stretch",
                    )
        else:
            st.info(
                "This source PSTH did not contain individual-trial exports. Rerun "
                "the PSTH with the current software to create trial-level metrics."
            )


def _psth_summary_controls(default_base_directory, key_prefix, *, summary_output_base=None):
    """Render saved-PSTH metric controls shared by every acquisition system."""

    st.subheader("PSTH summary statistics")
    st.write(
        "Load standardized values from a completed PSTH run, select a response "
        "window, and export mouse-level metrics and Prism-ready tables."
    )
    st.caption(
        "Sessions are averaged within each mouse. Mice are the inferential sample "
        "size; session-level tables are saved for QC."
    )
    st.info(
        "Saved numeric timecourses are created automatically by PSTH runs made with "
        "this software version. Older figure-only runs must be rerun once."
    )
    results_base = st.text_input(
        "PSTH results folder",
        value=str(Path(default_base_directory).expanduser()),
        key=f"{key_prefix}_summary_browser_base",
        help=(
            "Folder containing the timestamped PSTH run folders. Change this when "
            "the PSTHs were saved to a different base directory."
        ),
    )
    st.button(
        "Refresh saved-run list",
        key=f"{key_prefix}_summary_refresh_runs",
        help="Rescan the PSTH results folder for newly completed runs.",
    )
    run_directories = discover_psth_result_directories(results_base)
    custom_option = "__custom_psth_result_path__"
    run_options = [str(path) for path in run_directories] + [custom_option]

    def run_label(option):
        if option == custom_option:
            return "Enter a custom result path…"
        path = Path(option)
        try:
            relative = path.relative_to(Path(results_base).expanduser())
        except ValueError:
            relative = path
        return "Base folder (legacy direct output)" if relative == Path(".") else str(relative)

    selected_run = st.selectbox(
        "Saved PSTH run folder",
        run_options,
        format_func=run_label,
        key=f"{key_prefix}_summary_saved_run",
        help=(
            "Completed PSTH runs found below the results folder, newest first. "
            "Choose the parent run folder; compatible condition subfolders are "
            "combined automatically."
        ),
    )
    if not run_directories:
        st.warning(
            "No saved PSTH runs with numeric timecourse exports were found in this "
            "folder. Choose a different results folder or enter a custom path."
        )
    if selected_run == custom_option:
        input_directory = st.text_input(
            "Custom PSTH result directory",
            value=str(Path(results_base).expanduser()),
            key=f"{key_prefix}_summary_custom_input",
            help=(
                "Enter one completed PSTH run folder, including a parent folder "
                "containing compatible group/condition result subfolders."
            ),
        )
    else:
        input_directory = selected_run
        st.caption(f"Selected PSTH run: {input_directory}")
    detected_bundle = None
    detected_signal_type = "photometry"
    try:
        detected_bundle = load_saved_psth_timecourses(input_directory)
    except (OSError, ValueError):
        pass
    else:
        detected_signal_type = detected_bundle["identity"].get("signal_type", "photometry")
        st.caption(
            f"Detected response type: {detected_signal_type} · "
            f"{detected_bundle['identity'].get('signal', 'signal')}"
        )
        if not detected_bundle["trial_data_available"]:
            st.warning(
                "This older PSTH run has average timecourses but no individual-trial "
                "export. Average-PSTH metrics will work; rerun the PSTH for per-trial "
                "metrics."
            )
    first, second, third = st.columns(3)
    with first:
        response_start = st.number_input(
            "Response-window start (s)",
            value=0.0,
            key=f"{key_prefix}_summary_start",
            help="First time included when calculating every selected response metric.",
        )
        response_end = st.number_input(
            "Response-window end (s)",
            value=2.0,
            key=f"{key_prefix}_summary_end",
            help="Last time included when calculating every selected response metric.",
        )
    with second:
        peak_smoothing = st.number_input(
            "Peak smoothing (s)",
            min_value=0.0,
            value=0.1,
            step=0.05,
            key=f"{key_prefix}_summary_smoothing",
            help=(
                "Centered moving-average duration used only to locate peaks and "
                "troughs. Mean and AUC use the unsmoothed mouse trace."
            ),
        )
        test = st.selectbox(
            "Pairwise statistical test",
            ("auto", "paired_t", "wilcoxon", "welch", "mannwhitney"),
            key=f"{key_prefix}_summary_test",
            help=(
                "Auto uses paired t-tests for repeated conditions in the same mice "
                "and Welch tests for independent groups. Holm correction is applied "
                "across comparisons."
            ),
        )
    with third:
        bootstrap_resamples = st.number_input(
            "Bootstrap resamples",
            min_value=100,
            value=5000,
            step=500,
            key=f"{key_prefix}_summary_bootstrap",
            help="Mouse-level resamples used for difference and effect-size intervals.",
        )
        bootstrap_seed = st.number_input(
            "Bootstrap seed",
            min_value=0,
            value=0,
            step=1,
            key=f"{key_prefix}_summary_seed",
            help="Keeps bootstrap confidence intervals reproducible.",
        )
    metric_labels = {
        "mean": "Mean response",
        "auc": "Signed AUC",
        "positive_auc": "Positive AUC",
        "negative_auc": "Negative AUC",
        "peak": "Positive peak",
        "peak_latency": "Peak latency",
        "trough": "Negative trough",
        "trough_latency": "Trough latency",
        "lick_count": "Lick count",
        "any_lick": "Any lick (trial fraction)",
        "first_lick_latency": "First-lick latency",
        "distance": "Distance traveled",
        "moving_fraction": "Fraction of time moving",
        "movement_onset_latency": "Movement-onset latency",
    }
    available_metrics = metrics_for_signal_type(detected_signal_type)
    default_metrics = {
        "licking": ("mean", "lick_count", "any_lick", "first_lick_latency"),
        "running": (
            "mean",
            "distance",
            "moving_fraction",
            "movement_onset_latency",
            "peak",
        ),
    }.get(detected_signal_type, ("mean", "auc", "peak", "peak_latency"))
    metric_key = f"{key_prefix}_summary_metrics"
    signal_type_key = f"{key_prefix}_summary_metric_signal_type"
    if st.session_state.get(signal_type_key) != detected_signal_type:
        st.session_state[metric_key] = list(default_metrics)
        st.session_state[signal_type_key] = detected_signal_type
    metrics = st.multiselect(
        "Metrics",
        available_metrics,
        default=default_metrics,
        format_func=metric_labels.get,
        key=metric_key,
        help=(
            "Photometry metrics describe signal magnitude and timing. Licking metrics "
            "include count, response probability, and first-lick latency. Running "
            "metrics include distance, time moving, and movement onset. Each is saved "
            "both from the averaged PSTH and from individual trials."
        ),
    )
    movement_threshold = 1.0
    if detected_signal_type == "running":
        movement_threshold = st.number_input(
            "Movement threshold (cm/s)",
            min_value=0.0,
            value=1.0,
            step=0.25,
            key=f"{key_prefix}_summary_movement_threshold",
            help=(
                "Running speed above this value counts as movement when calculating "
                "moving fraction and movement-onset latency."
            ),
        )
    window_valid = response_start < response_end
    if not window_valid:
        st.error("Response-window start must be less than its end.")
    if not metrics:
        st.error("Select at least one response metric.")
    output_key = f"{key_prefix}_summary_output"
    last_input_key = f"{key_prefix}_summary_last_input"
    result_key = f"{key_prefix}_summary_result_directory"
    default_output = str(
        Path(summary_output_base).expanduser()
        if summary_output_base is not None
        else Path(input_directory).expanduser() / "summary_statistics"
    )
    if st.session_state.get(last_input_key) != str(input_directory):
        st.session_state[output_key] = default_output
        st.session_state[last_input_key] = str(input_directory)
        st.session_state.pop(result_key, None)
    output_base = st.text_input(
        "Summary-statistics output base directory",
        key=output_key,
        help="Each run is saved in a new timestamped folder below this directory.",
    )
    window_label = safe_analysis_run_label(f"{response_start:g}to{response_end:g}s")
    output_directory = next_analysis_run_directory(output_base, window_label)
    st.caption(f"This summary run will be saved to: {output_directory}")
    left, right = st.columns(2)
    with left:
        preview = st.button(
            "Preview saved PSTH",
            width="stretch",
            key=f"{key_prefix}_summary_preview",
        )
    with right:
        run_summary = st.button(
            "Calculate metrics and Prism exports",
            type="primary",
            width="stretch",
            disabled=not window_valid or not metrics,
            key=f"{key_prefix}_summary_run",
        )
    if preview:
        try:
            bundle = load_saved_psth_timecourses(input_directory)
        except (OSError, ValueError) as error:
            st.error(f"Could not load saved PSTH values: {error}")
        else:
            import matplotlib.pyplot as plt

            grouped = {}
            for record, trace in zip(bundle["mouse_records"], bundle["mouse_matrix"], strict=True):
                label = f"{record['group']} / {record['condition']}"
                grouped.setdefault(label, []).append(trace)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for label, traces in grouped.items():
                ax.plot(
                    bundle["time"],
                    pd.DataFrame(traces).mean(axis=0).to_numpy(),
                    label=label,
                )
            ax.axvspan(response_start, response_end, color="C1", alpha=0.18)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set(
                xlabel=f"Time from {bundle['identity']['event']} (s)",
                ylabel=bundle["identity"]["units"],
                title="Saved mouse-level PSTH preview",
            )
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            identity = bundle["identity"]
            st.caption(
                f"{identity['workflow']} · {identity['signal']} aligned to "
                f"{identity['event']} · {len(bundle['mouse_records'])} mouse rows · "
                f"{len(bundle['session_records'])} sessions"
            )
    command = build_saved_psth_statistics_command(
        PROJECT_ROOT,
        input_directory,
        output_directory,
        response_window=(response_start, response_end),
        peak_smoothing=peak_smoothing,
        metrics=metrics,
        test=test,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        movement_threshold=movement_threshold,
    )
    if run_summary:
        if not Path(input_directory).expanduser().exists():
            st.error(f"PSTH result directory does not exist: {input_directory}")
        else:
            with st.spinner("Calculating PSTH summary statistics..."):
                completed = _run_command_batch(command, "PSTH summary statistics")
            if completed:
                source_manifest = Path(input_directory) / "input_manifest.csv"
                if source_manifest.is_file():
                    Path(output_directory).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        source_manifest,
                        Path(output_directory) / "input_manifest.csv",
                    )
                    source_metadata = Path(input_directory) / "input_manifest_metadata.json"
                    if source_metadata.is_file():
                        shutil.copy2(
                            source_metadata,
                            Path(output_directory) / "input_manifest_metadata.json",
                        )
                st.session_state[result_key] = str(output_directory)
    result_directory = st.session_state.get(result_key)
    if result_directory:
        _show_psth_summary_results(result_directory, key_prefix)
    with st.expander("Advanced: inspect or copy summary command"):
        if st.button("Preview summary command", key=f"{key_prefix}_summary_command"):
            _preview_command(command)


def _session_trace_controls(
    workflow,
    rows,
    data_root,
    manifest_dir,
    manifest_name,
    active_manifest_key,
):
    """Render a shared continuous-session example-trace viewer and exporter."""

    st.subheader("Single-session example trace")
    st.write(
        "Inspect one continuous processed signal with measured Cue and Ensure "
        "durations shown as transparent patches and individual licks shown as ticks."
    )
    st.caption(
        "Ensure patches use the recorded onset and offset, so 0.5-s and 2-s "
        "deliveries retain their true durations. Older files without an offset use "
        "an onset line rather than an assumed duration."
    )
    try:
        valid_rows = (
            normalize_manifest_rows(rows)
            if workflow == "conventional"
            else normalize_lifetime_manifest_rows(workflow, rows)
        )
    except ValueError as error:
        st.warning(f"Complete the session table before loading a trace: {error}")
        return
    if not valid_rows:
        st.info("Add at least one session to the session table.")
        return

    def row_label(index):
        row = valid_rows[index]
        details = f"{row['mouse']} · {row['date']} · run {int(row['run']):03d}"
        if row.get("condition"):
            details += f" · {row['condition']}"
        return details

    first, second, third = st.columns(3)
    with first:
        selected_index = st.selectbox(
            "Session",
            range(len(valid_rows)),
            format_func=row_label,
            key=f"{workflow}_trace_session",
        )
    selected_row = valid_rows[selected_index]
    with second:
        if workflow == "conventional":
            signal = st.selectbox(
                "Signal",
                tuple(CONVENTIONAL_PHOTOMETRY_SIGNAL_LABELS),
                format_func=CONVENTIONAL_PHOTOMETRY_SIGNAL_LABELS.get,
                key="conventional_trace_signal",
            )
        else:
            signal = st.selectbox(
                "Signal",
                LIFETIME_SIGNALS[workflow],
                format_func=LIFETIME_SIGNAL_LABELS.get,
                key=f"{workflow}_trace_signal",
            )
    with third:
        channel = "manifest"
        if workflow == "conventional":
            channel = st.selectbox(
                "Photoreceiver channel",
                ("manifest", "1", "2"),
                key="conventional_trace_channel",
                help="Use the selected session's manifest channel or override it.",
            )
        load_trace = st.button(
            "Load session trace",
            type="primary",
            width="stretch",
            key=f"{workflow}_load_session_trace",
        )

    data_key = f"{workflow}_session_trace_data"
    identity_key = f"{workflow}_session_trace_identity"
    selection_identity = (
        selected_row["mouse"],
        selected_row["date"],
        int(selected_row["run"]),
        signal,
        channel,
        str(Path(data_root).expanduser()),
    )
    if load_trace:
        try:
            with st.spinner("Loading processed session trace..."):
                data = (
                    load_conventional_session_trace(
                        selected_row, data_root, signal=signal, channel=channel
                    )
                    if workflow == "conventional"
                    else load_lifetime_session_trace(workflow, selected_row, data_root, signal)
                )
        except (OSError, KeyError, ValueError) as error:
            st.error(f"Could not load session trace: {error}")
        else:
            st.session_state[data_key] = data
            st.session_state[identity_key] = selection_identity
            start = float(data.time[0])
            end = float(data.time[-1])
            if data.ensure_starts.size:
                center = float(data.ensure_starts[0])
                zoom_start = max(start, center - 5.0)
                zoom_end = min(end, center + 30.0)
            else:
                zoom_start = start
                zoom_end = min(end, start + 60.0)
            st.session_state[f"{workflow}_trace_overview_start"] = start
            st.session_state[f"{workflow}_trace_overview_end"] = end
            st.session_state[f"{workflow}_trace_zoom_start"] = zoom_start
            st.session_state[f"{workflow}_trace_zoom_end"] = zoom_end

    data = st.session_state.get(data_key)
    loaded_identity = st.session_state.get(identity_key)
    if data is None:
        st.info("Choose a session and signal, then click Load session trace.")
        return
    if loaded_identity != selection_identity:
        st.warning(
            "The controls now select a different session or signal. Click Load "
            "session trace to update the displayed data."
        )
    duration_caption = []
    cue_durations = data.cue_ends - data.cue_starts
    cue_durations = cue_durations[np.isfinite(cue_durations)]
    ensure_durations = data.ensure_ends - data.ensure_starts
    ensure_durations = ensure_durations[np.isfinite(ensure_durations)]
    if cue_durations.size:
        duration_caption.append(f"median Cue duration {np.median(cue_durations):g} s")
    if ensure_durations.size:
        duration_caption.append(f"median Ensure duration {np.median(ensure_durations):g} s")
    st.caption(
        f"Loaded {data.session_id} · {data.signal} · {len(data.lick_times)} licks"
        + (f" · {' · '.join(duration_caption)}" if duration_caption else "")
    )

    window_left, window_right = st.columns(2)
    with window_left:
        st.markdown("**Overview window**")
        overview_start = st.number_input(
            "Overview start (s)", key=f"{workflow}_trace_overview_start"
        )
        overview_end = st.number_input("Overview end (s)", key=f"{workflow}_trace_overview_end")
    with window_right:
        st.markdown("**Zoom window**")
        zoom_start = st.number_input("Zoom start (s)", key=f"{workflow}_trace_zoom_start")
        zoom_end = st.number_input("Zoom end (s)", key=f"{workflow}_trace_zoom_end")
    transform_labels = {
        "native": "Native processed units",
        "subtract_mean": "Subtract full-session mean",
        "zscore": "Full-session z-score",
    }
    option_columns = st.columns(4)
    with option_columns[0]:
        transform = st.selectbox(
            "Display transform",
            tuple(transform_labels),
            format_func=transform_labels.get,
            key=f"{workflow}_trace_transform",
            help=(
                "The transform is calculated from the full session, so changing the "
                "overview or zoom window cannot change the trace scale."
            ),
        )
    with option_columns[1]:
        show_cues = st.checkbox("Show Cue", value=True, key=f"{workflow}_trace_cues")
    with option_columns[2]:
        show_ensures = st.checkbox("Show Ensure", value=True, key=f"{workflow}_trace_ensures")
    with option_columns[3]:
        show_licks = st.checkbox("Show licks", value=True, key=f"{workflow}_trace_licks")
    windows_valid = (
        overview_start < overview_end
        and overview_start <= zoom_start < zoom_end <= overview_end
        and overview_start >= data.time[0]
        and overview_end <= data.time[-1]
    )
    if not windows_valid:
        st.error(
            f"Windows must increase and remain within {data.time[0]:g}–"
            f"{data.time[-1]:g} s, and the zoom must lie within the overview."
        )
        return
    try:
        figure, _, _ = plot_session_trace(
            data,
            overview_window=(overview_start, overview_end),
            zoom_window=(zoom_start, zoom_end),
            transform=transform,
            show_cues=show_cues,
            show_ensures=show_ensures,
            show_licks=show_licks,
        )
    except ValueError as error:
        st.error(f"Could not draw session trace: {error}")
        return
    import matplotlib.pyplot as plt

    st.pyplot(figure, width="stretch")
    plt.close(figure)
    output_base = st.text_input(
        "Session-trace output base directory",
        value=str(workflow_analysis_directory(PROJECT_ROOT, workflow) / "session_traces"),
        key=f"{workflow}_trace_output_base",
        help="Each saved example is written to a new timestamped folder.",
    )
    output_directory = next_analysis_run_directory(output_base, f"{data.session_id}__{data.signal}")
    st.caption(f"This example trace will be saved to: {output_directory}")
    if st.button(
        "Save example trace and data",
        type="primary",
        width="stretch",
        key=f"{workflow}_save_session_trace",
    ):
        try:
            saved_manifest, created = _save_manifest_for_run(
                workflow,
                rows,
                manifest_dir,
                manifest_name,
                st.session_state.get(active_manifest_key, ""),
            )
            st.session_state[active_manifest_key] = str(saved_manifest)
            outputs = save_session_trace(
                data,
                output_directory,
                overview_window=(overview_start, overview_end),
                zoom_window=(zoom_start, zoom_end),
                transform=transform,
                show_cues=show_cues,
                show_ensures=show_ensures,
                show_licks=show_licks,
            )
            snapshot_manifest(saved_manifest, output_directory)
        except (OSError, ValueError) as error:
            st.error(f"Could not save session trace: {error}")
        else:
            action = " and a new manifest version" if created else ""
            st.success(f"Saved {len(outputs)} trace files{action} to {output_directory}")


def _lifetime_gui(workflow: str) -> None:
    label = "FluoPulse" if workflow == "fluopulse" else "iFLIP3"
    state_key = f"{workflow}_manifest_rows"
    columns = lifetime_manifest_columns(workflow)
    if state_key not in st.session_state:
        row = {
            "mouse": "",
            "date": "",
            "run": 1,
            "group": "",
            "condition": "",
            **{name: "" for name in columns[5:]},
        }
        st.session_state[state_key] = [row]

    with st.sidebar:
        st.subheader(f"{label} workspace")
        st.text_input("Repository", value=str(PROJECT_ROOT), disabled=True)
        data_root = st.text_input("Data root", value="Z:\\", key=f"{workflow}_root")
        legacy_manifest = PROJECT_ROOT / "analysis" / workflow / "sessions.csv"
        analysis_workspace, manifest_dir, manifest_name, active_manifest_key = (
            _manifest_workspace_sidebar(
                workflow,
                label,
                legacy_paths=(legacy_manifest,),
            )
        )
        manifest_path = st.session_state.get(active_manifest_key, "") or (
            manifest_dir / "unsaved_sessions.csv"
        )
        st.caption(
            "Blank recording paths use the laboratory mouse/date/run conventions "
            "under the data root. NI-DAQ is optional; when its path is blank, the "
            "workflow uses a conventionally named file if one exists and otherwise "
            "keeps the recording on its native sensor clock. Explicit paths override "
            "discovery."
        )

    sessions_tab, align_tab, trace_tab, psth_tab, summary_tab, glm_tab = st.tabs(
        [
            "1. Sessions",
            "2. Preprocess and align",
            "3. Session trace",
            "4. Event-aligned PSTH",
            "5. Summary statistics",
            "6. Lifetime GLM",
        ]
    )
    with sessions_tab:
        st.subheader(f"{label} session manifest")
        _manifest_library_panel(
            workflow,
            state_key,
            manifest_dir,
            active_manifest_key,
            legacy_paths=(legacy_manifest,),
        )
        if workflow == "iflip3":
            st.info(
                "The path assistant finds session and background candidates from mouse, "
                "date, and run. A matched iFLIP3 background improves background "
                "correction when available, but older recordings can be processed "
                "without one."
            )
        else:
            st.info(
                "The path assistant finds Doric and NI-DAQ files from mouse, date, and "
                "run, including standard names with descriptive suffix text."
            )

        with st.expander("Add a session and find its files", expanded=True):
            identity_columns = st.columns((1.2, 1.2, 0.8, 1, 1))
            with identity_columns[0]:
                discovery_mouse = st.text_input("Mouse", key=f"{workflow}_discovery_mouse")
            with identity_columns[1]:
                discovery_date = st.text_input("Date (YYMMDD)", key=f"{workflow}_discovery_date")
            with identity_columns[2]:
                discovery_run = st.number_input(
                    "Run",
                    min_value=0,
                    step=1,
                    value=1,
                    key=f"{workflow}_discovery_run",
                )
            with identity_columns[3]:
                discovery_group = st.text_input("Group", key=f"{workflow}_discovery_group")
            with identity_columns[4]:
                discovery_condition = st.text_input(
                    "Condition", key=f"{workflow}_discovery_condition"
                )

            if st.button(
                "Find matching files",
                type="primary",
                key=f"{workflow}_discover_files",
            ):
                try:
                    choices = discover_lifetime_paths(
                        workflow,
                        discovery_mouse,
                        discovery_date,
                        discovery_run,
                        data_root,
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.session_state[f"{workflow}_discovery_results"] = choices
                    st.session_state[f"{workflow}_discovery_identity"] = (
                        discovery_mouse.strip(),
                        discovery_date.strip(),
                        int(discovery_run),
                        str(Path(data_root).expanduser()),
                    )

            choices = st.session_state.get(f"{workflow}_discovery_results")
            identity = st.session_state.get(f"{workflow}_discovery_identity")
            current_identity = (
                discovery_mouse.strip(),
                discovery_date.strip(),
                int(discovery_run),
                str(Path(data_root).expanduser()),
            )
            if choices is not None and identity == current_identity:
                recording_field = "doric_path" if workflow == "fluopulse" else "iflip_path"
                recording_label = (
                    "Doric recording" if workflow == "fluopulse" else "iFLIP3 recording"
                )
                if not choices[recording_field]:
                    st.warning(
                        f"No {recording_label} candidate was found. Enter its filename or "
                        "path below."
                    )
                recording_path = _path_picker(
                    workflow,
                    recording_field,
                    recording_label,
                    choices[recording_field],
                    data_root,
                )
                nidaq_path = _path_picker(
                    workflow,
                    "nidaq_path",
                    "NI-DAQ file (optional)",
                    choices["nidaq_path"],
                    data_root,
                    optional=True,
                )
                running_path = _path_picker(
                    workflow,
                    "running_path",
                    "Running file (optional)",
                    choices["running_path"],
                    data_root,
                    optional=True,
                )
                background_path = ""
                if workflow == "iflip3":
                    background_candidates = [
                        path
                        for path in choices["background_path"]
                        if _relative_path_text(path, data_root) != recording_path
                    ]
                    if not background_candidates:
                        st.info(
                            "No likely background was found. You can leave the background "
                            "blank or enter one manually."
                        )
                    background_path = _path_picker(
                        workflow,
                        "background_path",
                        "Matched background (optional)",
                        background_candidates,
                        data_root,
                        optional=True,
                    )

                if st.button(
                    "Add or update manifest row",
                    width="stretch",
                    key=f"{workflow}_add_discovered",
                ):
                    if not recording_path:
                        st.error(f"Choose or enter the {recording_label} path.")
                    else:
                        new_row = {
                            "mouse": discovery_mouse.strip(),
                            "date": discovery_date.strip(),
                            "run": int(discovery_run),
                            "group": discovery_group.strip(),
                            "condition": discovery_condition.strip(),
                            recording_field: recording_path,
                            "nidaq_path": nidaq_path,
                            "running_path": running_path,
                        }
                        if workflow == "iflip3":
                            new_row["background_path"] = background_path
                        existing_rows = [
                            row
                            for row in st.session_state[state_key]
                            if str(row.get("mouse", "") or "").strip()
                            or str(row.get("date", "") or "").strip()
                        ]
                        new_identity = (
                            new_row["mouse"].casefold(),
                            new_row["date"],
                            new_row["run"],
                        )
                        updated_rows = []
                        replaced = False
                        for row in existing_rows:
                            try:
                                row_run = int(row.get("run", -1))
                            except (TypeError, ValueError):
                                row_run = -1
                            row_identity = (
                                str(row.get("mouse", "")).casefold(),
                                str(row.get("date", "")),
                                row_run,
                            )
                            if row_identity == new_identity:
                                updated_rows.append(new_row)
                                replaced = True
                            else:
                                updated_rows.append(row)
                        if not replaced:
                            updated_rows.append(new_row)
                        st.session_state[state_key] = updated_rows
                        st.session_state.pop(f"{workflow}_manifest_editor", None)
                        st.rerun()

        uploaded = st.file_uploader("Load an existing CSV", type="csv", key=f"{workflow}_upload")
        if uploaded is not None and st.button("Use uploaded CSV", key=f"{workflow}_use_upload"):
            st.session_state[state_key] = _uploaded_rows(uploaded)
            st.rerun()
        edited = st.data_editor(
            pd.DataFrame(st.session_state[state_key], columns=columns),
            num_rows="dynamic",
            column_order=columns,
            column_config={
                "mouse": st.column_config.TextColumn("Mouse", required=True),
                "date": st.column_config.TextColumn("Date (YYMMDD)", required=True),
                "run": st.column_config.NumberColumn("Run", min_value=0, step=1),
                "group": st.column_config.TextColumn("Group"),
                "condition": st.column_config.TextColumn("Condition"),
                "background_path": st.column_config.TextColumn(
                    "Matched background path (optional)"
                ),
            },
            width="stretch",
            key=f"{workflow}_manifest_editor",
        )
        rows = _records(edited)
        st.session_state[state_key] = rows
        left, middle, update, right = st.columns(4)
        with left:
            if st.button("Validate sessions", width="stretch", key=f"{workflow}_validate"):
                try:
                    sessions = normalize_lifetime_manifest_rows(workflow, rows)
                    st.success(f"Validated {len(sessions)} unique sessions.")
                except ValueError as error:
                    st.error(str(error))
        with middle:
            if st.button(
                "Save new version",
                type="primary",
                width="stretch",
                key=f"{workflow}_save",
            ):
                try:
                    saved = _save_manifest_version(
                        workflow,
                        rows,
                        manifest_dir,
                        manifest_name,
                        parent_manifest=st.session_state.get(active_manifest_key) or None,
                    )
                    st.session_state[active_manifest_key] = str(saved)
                    st.success(f"Saved new manifest version: {saved}")
                except (OSError, ValueError) as error:
                    st.error(str(error))
        with update:
            active_manifest = st.session_state.get(active_manifest_key, "")
            if st.button(
                "Update active",
                width="stretch",
                key=f"{workflow}_update_manifest",
                disabled=not active_manifest,
                help="Explicitly replace the active manifest instead of creating a version.",
            ):
                try:
                    saved = _write_workflow_manifest(workflow, active_manifest, rows)
                    write_manifest_metadata(
                        saved,
                        workflow=workflow,
                        display_name=manifest_name,
                        row_count=_normalized_manifest_row_count(workflow, rows),
                        code_commit=local_git_commit(PROJECT_ROOT),
                    )
                    st.success(f"Updated active manifest: {saved}")
                except (OSError, ValueError) as error:
                    st.error(str(error))
        with right:
            try:
                validated = normalize_lifetime_manifest_rows(workflow, rows)
                buffer = io.StringIO(newline="")
                writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(validated)
                csv_text = buffer.getvalue()
            except ValueError:
                csv_text = ""
            st.download_button(
                "Download CSV",
                csv_text,
                file_name=f"{safe_analysis_run_label(manifest_name)}.csv",
                mime="text/csv",
                disabled=not csv_text,
                width="stretch",
                key=f"{workflow}_download",
            )

    with align_tab:
        st.subheader("Preprocess and align lifetime recordings")
        st.write(
            "Loads each lifetime recording and saves an analysis-ready compressed NPZ "
            "file beside the primary raw recording. When NI-DAQ is available, clocks "
            "are synchronized and NI-DAQ behavior is included; otherwise the native "
            "sensor clock and available embedded events are retained. PSTH, heatmap, "
            "and GLM analyses use this processed file."
        )
        iflip3_fit_settings = None
        iflip3_calibration_path = None
        active_iflip3_calibration = None
        iflip3_temporal_bin_seconds = None
        if workflow == "iflip3":
            st.markdown("#### Lifetime temporal resolution")
            iflip3_temporal_bin_seconds = st.selectbox(
                "Pool photon histograms before lifetime calculation",
                (None, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
                format_func=lambda value: (
                    "Native acquisition interval" if value is None else f"{value:g} seconds"
                ),
                key="iflip3_temporal_bin_seconds",
                help=(
                    "Background, dead-time, and afterpulse corrections are applied at the "
                    "native rate. Corrected photon histograms are then summed before MPET "
                    "and component-fraction calculation."
                ),
            )
            st.markdown("#### Multi-session calibration")
            st.write(
                "Estimate shared τ1, τ2, and IRF σ plus one t0 per recording from "
                "multiple compatible recordings. Residual fit background is fixed at zero."
            )
            try:
                preview_rows = normalize_lifetime_manifest_rows(workflow, rows)
            except ValueError:
                preview_rows = []
            preview_labels = [
                f"{row['mouse']} · {row['date']} · run {int(row['run']):03d}"
                + (f" · {row['condition']}" if row.get("condition") else "")
                for row in preview_rows
            ]
            selected_calibration_indices = st.multiselect(
                "Sessions used to estimate the calibration",
                range(len(preview_rows)),
                default=range(len(preview_rows)),
                format_func=lambda index: preview_labels[index],
                key="iflip3_calibration_sessions",
                help=(
                    "Use all technically compatible sessions by default. Choose a custom "
                    "subset when acquisition hardware/settings changed or QC identifies an outlier."
                ),
            )
            pending_calibration_path = st.session_state.pop("iflip3_pending_calibration_path", None)
            if pending_calibration_path is not None:
                st.session_state["iflip3_calibration_path"] = pending_calibration_path
            calibration_path_text = st.text_input(
                "Calibration JSON",
                value=str(analysis_workspace / "calibrations" / "shared.json"),
                key="iflip3_calibration_path",
            ).strip()
            calibration_path = Path(calibration_path_text).expanduser()
            if not calibration_path.is_absolute():
                calibration_path = PROJECT_ROOT / calibration_path
            run_column, load_column = st.columns(2)
            with run_column:
                if st.button(
                    "Run multi-session calibration",
                    type="primary",
                    disabled=len(selected_calibration_indices) < 2,
                    key="iflip3_run_calibration",
                ):
                    try:
                        with st.spinner(
                            "Loading selected sessions and fitting shared parameters..."
                        ):
                            calibration = calibrate_iflip3_sessions(
                                [preview_rows[index] for index in selected_calibration_indices],
                                data_root,
                            )
                    except (OSError, ValueError, RuntimeError) as error:
                        st.error(f"Multi-session calibration failed: {error}")
                    else:
                        st.session_state["iflip3_calibration_result"] = calibration
                        st.success("Calibration converged. Review QC, then save it.")
            with load_column:
                if st.button("Load saved calibration", key="iflip3_load_calibration"):
                    try:
                        calibration = load_iflip3_calibration(calibration_path)
                    except (OSError, ValueError) as error:
                        st.error(f"Calibration could not be loaded: {error}")
                    else:
                        st.session_state["iflip3_calibration_result"] = calibration
                        st.success(f"Loaded {calibration_path}")
            calibration = st.session_state.get("iflip3_calibration_result")
            if calibration is not None:
                _show_iflip3_calibration(calibration)
                save_new_version = st.checkbox(
                    "Save as a new versioned calibration file",
                    value=True,
                    key="iflip3_version_calibration_save",
                    help=(
                        "Recommended. Creates a timestamped JSON and never replaces an "
                        "existing calibration."
                    ),
                )
                save_target = (
                    next_iflip3_calibration_path(calibration_path, calibration)
                    if save_new_version
                    else calibration_path
                )
                confirm_calibration_replace = False
                if save_new_version:
                    st.caption(f"New calibration file: {save_target}")
                elif calibration_path.exists():
                    st.warning(f"The existing calibration will be replaced: {calibration_path}")
                    confirm_calibration_replace = st.checkbox(
                        "I understand that saving will replace this calibration JSON",
                        value=False,
                        key="iflip3_confirm_calibration_replace",
                    )
                if st.button("Save calibration", key="iflip3_save_calibration"):
                    if (
                        not save_new_version
                        and calibration_path.exists()
                        and not confirm_calibration_replace
                    ):
                        st.error(
                            "Confirm replacement, choose a different path, or save as a "
                            "new versioned file."
                        )
                    else:
                        try:
                            saved_calibration = save_iflip3_calibration(calibration, save_target)
                        except OSError as error:
                            st.error(f"Calibration could not be saved: {error}")
                        else:
                            st.session_state["iflip3_pending_calibration_path"] = str(
                                saved_calibration
                            )
                            st.session_state["iflip3_calibration_saved_message"] = (
                                f"Saved {saved_calibration}"
                            )
                            st.rerun()
            saved_calibration_message = st.session_state.pop(
                "iflip3_calibration_saved_message", None
            )
            if saved_calibration_message:
                st.success(saved_calibration_message)
            calibration_file_exists = calibration_path.is_file()
            use_calibration = st.checkbox(
                "Apply this saved calibration during preprocessing",
                value=False,
                disabled=not calibration_file_exists,
                key="iflip3_use_calibration",
            )
            if use_calibration:
                try:
                    active_iflip3_calibration = load_iflip3_calibration(calibration_path)
                except (OSError, ValueError) as error:
                    st.error(f"Calibration cannot be used: {error}")
                else:
                    iflip3_calibration_path = calibration_path
                    st.success(
                        "Preprocessing will use shared τ1, τ2, and IRF σ, estimate or "
                        "reuse one fixed t0 per session, and vary only component counts "
                        "over time."
                    )
            st.caption(
                "Create separate calibration files for different pulse intervals, lifetime-bin "
                "axes, detectors, optical paths, or acquisition configurations."
            )
            st.divider()
            st.markdown("#### Inspect the calibration on one session")
            st.write(
                "Apply the current shared calibration to one recording and inspect the exact "
                "fixed-basis decomposition used during calibrated preprocessing."
            )
            diagnostic_calibration = active_iflip3_calibration or calibration
            if preview_rows and diagnostic_calibration is not None:
                calibrated_preview_index = st.selectbox(
                    "Session to inspect with the calibration",
                    range(len(preview_rows)),
                    format_func=lambda index: preview_labels[index],
                    key="iflip3_calibrated_preview_session",
                )
                if active_iflip3_calibration is None:
                    st.info(
                        "This previews the calibration result shown above. Save and enable it "
                        "before preprocessing to use the same values in the batch."
                    )
                if st.button(
                    "Inspect calibrated session",
                    type="primary",
                    key="iflip3_preview_calibrated_session",
                ):
                    try:
                        with st.spinner("Applying the shared calibration to this recording..."):
                            calibrated_preview = apply_iflip3_calibration(
                                preview_rows[calibrated_preview_index],
                                data_root,
                                diagnostic_calibration,
                            )
                            try:
                                binning_sensitivity = estimate_iflip3_binning_sensitivity(
                                    calibrated_preview
                                )
                                binning_sensitivity_error = None
                            except ValueError as sensitivity_error:
                                binning_sensitivity = None
                                binning_sensitivity_error = str(sensitivity_error)
                    except (OSError, ValueError, RuntimeError) as error:
                        st.error(f"The calibrated session preview failed: {error}")
                    else:
                        st.session_state["iflip3_calibrated_preview_result"] = calibrated_preview
                        st.session_state["iflip3_calibrated_preview_identity"] = preview_labels[
                            calibrated_preview_index
                        ]
                        st.session_state["iflip3_calibrated_preview_created_utc"] = (
                            diagnostic_calibration.created_utc
                        )
                        st.session_state["iflip3_binning_sensitivity"] = binning_sensitivity
                        st.session_state["iflip3_binning_sensitivity_error"] = (
                            binning_sensitivity_error
                        )
                calibrated_preview = st.session_state.get("iflip3_calibrated_preview_result")
                calibrated_identity = st.session_state.get("iflip3_calibrated_preview_identity")
                if calibrated_preview is not None:
                    st.caption(f"Calibrated session: {calibrated_identity}")
                    if (
                        calibrated_identity != preview_labels[calibrated_preview_index]
                        or st.session_state.get("iflip3_calibrated_preview_created_utc")
                        != diagnostic_calibration.created_utc
                    ):
                        st.warning(
                            "The selected session or calibration changed after this preview. "
                            "Inspect the calibrated session again."
                        )
                    _show_iflip3_fit_preview(calibrated_preview)
                    st.success(
                        "This view uses fixed τ1, τ2, t0, and IRF σ with residual "
                        "background fixed at zero."
                    )
                    binning_sensitivity = st.session_state.get("iflip3_binning_sensitivity")
                    if binning_sensitivity is not None:
                        st.markdown("##### Photon-count sensitivity by temporal bin")
                        _show_iflip3_binning_sensitivity(binning_sensitivity)
                    elif st.session_state.get("iflip3_binning_sensitivity_error"):
                        st.info(
                            "Photon-count sensitivity could not be estimated: "
                            + st.session_state["iflip3_binning_sensitivity_error"]
                        )
            elif not preview_rows:
                st.info("Add a complete session row before inspecting an iFLIP3 fit.")
            else:
                st.info("Run or load a multi-session calibration to inspect a session.")

            with st.expander("Advanced: unconstrained single-session fit"):
                st.write(
                    "Use this only to diagnose an outlier or support a legacy dataset without "
                    "a multi-session calibration. Its fitted values are not used when a saved "
                    "calibration is enabled."
                )
                iflip3_fit_settings = _iflip3_fit_controls()
                if preview_rows:
                    preview_index = st.selectbox(
                        "Session for unconstrained comparison",
                        range(len(preview_rows)),
                        format_func=lambda index: preview_labels[index],
                        key="iflip3_fit_preview_session",
                    )
                    if st.button(
                        "Run unconstrained diagnostic fit",
                        key="iflip3_preview_fit",
                    ):
                        try:
                            with st.spinner("Loading the recording and fitting its decay..."):
                                preview = preview_iflip3_fit(
                                    preview_rows[preview_index],
                                    data_root,
                                    iflip3_fit_settings,
                                )
                        except (OSError, ValueError, RuntimeError) as error:
                            st.error(f"The unconstrained fit could not be generated: {error}")
                        else:
                            st.session_state["iflip3_fit_preview_result"] = preview
                            st.session_state["iflip3_fit_preview_identity"] = preview_labels[
                                preview_index
                            ]
                            st.session_state["iflip3_fit_preview_settings"] = iflip3_fit_settings
                    preview = st.session_state.get("iflip3_fit_preview_result")
                    preview_identity = st.session_state.get("iflip3_fit_preview_identity")
                    if preview is not None:
                        st.caption(f"Unconstrained session: {preview_identity}")
                        if (
                            preview_identity != preview_labels[preview_index]
                            or st.session_state.get("iflip3_fit_preview_settings")
                            != iflip3_fit_settings
                        ):
                            st.warning(
                                "The selected session or fit controls changed after this preview. "
                                "Run the unconstrained fit again."
                            )
                        _show_iflip3_fit_preview(preview)
                        if preview.success:
                            st.success(
                                "The local optimizer converged. Differences from the shared "
                                "calibration are diagnostic and do not override it."
                            )
                        else:
                            st.warning(f"The optimizer reported: {preview.message}")
                else:
                    st.info("Add a complete session row before previewing an iFLIP3 fit.")
            st.divider()
            st.caption(
                "When a saved calibration is enabled, preprocessing ignores the advanced "
                "single-session controls."
            )
        _processed_provenance_panel(
            workflow,
            rows,
            data_root,
            lifetime_preprocessing_parameters(
                workflow,
                None if active_iflip3_calibration is not None else iflip3_fit_settings,
                active_iflip3_calibration,
                iflip3_temporal_bin_seconds,
            ),
        )
        overwrite_processed = st.checkbox(
            "Overwrite existing processed files",
            value=False,
            key=f"{workflow}_overwrite_processed",
        )
        if workflow == "iflip3":
            st.caption(
                "If this session was processed previously, select overwrite to apply changed "
                "fit parameters and regenerate its component signals."
            )
        command = build_lifetime_command(
            PROJECT_ROOT,
            "preprocess",
            workflow,
            manifest_path,
            data_root,
            overwrite=overwrite_processed,
            iflip3_fit_settings=(
                None if active_iflip3_calibration is not None else iflip3_fit_settings
            ),
            iflip3_calibration=iflip3_calibration_path,
            iflip3_temporal_bin_seconds=iflip3_temporal_bin_seconds,
        )
        if st.button(
            "Run preprocessing",
            type="primary",
            width="stretch",
            key=f"{workflow}_run_align",
        ):
            _run_lifetime_workflow(
                command,
                workflow,
                rows,
                manifest_dir,
                manifest_name,
                active_manifest_key,
                data_root,
                "Preprocessing",
            )
        with st.expander("Advanced: inspect or copy command"):
            if st.button("Preview preprocessing command", key=f"{workflow}_preview_align"):
                _preview_command(command)

    with trace_tab:
        _session_trace_controls(
            workflow,
            rows,
            data_root,
            manifest_dir,
            manifest_name,
            active_manifest_key,
        )

    with psth_tab:
        st.subheader("Event-aligned lifetime timecourses")
        psth_group_filter, psth_condition_filter = _analysis_subset_controls(
            rows, f"{workflow}_psth"
        )
        first, second, third = st.columns(3)
        with first:
            signal = st.selectbox(
                "Lifetime/QC signal",
                LIFETIME_SIGNALS[workflow],
                format_func=lambda value: LIFETIME_SIGNAL_LABELS.get(value, value),
                key=f"{workflow}_signal",
                help=(
                    "Processed signal to align and average around the selected event. "
                    + (
                        "Calibrated iFLIP3 short and long tau values are fixed parameters, "
                        "not time-varying signals, so they are recorded in metadata and "
                        "GLM baseline tables rather than offered as flat PSTHs."
                        if workflow == "iflip3"
                        else ""
                    )
                ),
            )
            event = st.selectbox(
                "Alignment event",
                LIFETIME_EVENTS,
                key=f"{workflow}_event",
                help="Each occurrence becomes time zero for one peri-event trial.",
            )
            if event in {
                "first_lick_after_ensure",
                "first_lick_bout_after_ensure",
            }:
                derived_match_start = st.number_input(
                    "Post-Ensure match start (s)",
                    min_value=0.0,
                    value=0.0,
                    key=f"{workflow}_derived_match_start",
                    help="Earliest allowed latency after Ensure for the matched event.",
                )
                derived_match_end = st.number_input(
                    "Post-Ensure match end (s)",
                    min_value=0.01,
                    value=20.0,
                    key=f"{workflow}_derived_match_end",
                    help=(
                        "Latest allowed latency. Matching stops at the next Ensure so "
                        "events are not assigned across adjacent deliveries."
                    ),
                )
                derived_event_window = (derived_match_start, derived_match_end)
                derived_event_window_valid = derived_match_start < derived_match_end
                if not derived_event_window_valid:
                    st.error("Match-window start must be less than its end.")
            else:
                derived_event_window = (0.0, 20.0)
                derived_event_window_valid = True
        with second:
            window_start = st.number_input(
                "Window start (s)",
                value=-5.0,
                key=f"{workflow}_window_start",
                help=(
                    "Earliest time retained relative to the alignment event; "
                    "negative is before the event."
                ),
            )
            window_end = st.number_input(
                "Window end (s)",
                value=20.0,
                key=f"{workflow}_window_end",
                help=(
                    "Latest time retained relative to the alignment event; "
                    "positive is after the event."
                ),
            )
            dt = st.number_input(
                "Time bin (s)",
                min_value=0.01,
                value=0.1,
                key=f"{workflow}_dt",
                help=(
                    "Spacing of the common time grid used to resample and average "
                    "trials. Smaller bins preserve more temporal detail."
                ),
            )
        with third:
            lifetime_label = {
                "tau": "τ",
                "mpet": "MPET",
                "mpet_afterpulse_only": "MPET",
                "mpet_uncorrected": "MPET",
            }.get(signal)
            normalization_labels = (
                {
                    "subtract": f"Baseline-subtracted (Δ{lifetime_label}, ps)",
                    "none": f"Unsubtracted ({lifetime_label}, ps)",
                }
                if lifetime_label is not None
                else {
                    "subtract": "Baseline-subtracted (Δ from baseline)",
                    "none": "Unsubtracted (native values)",
                }
            )
            normalization = st.selectbox(
                "Normalization",
                ("subtract", "none"),
                format_func=normalization_labels.get,
                key=f"{workflow}_normalization",
                help=(
                    "Baseline subtraction expresses each trial relative to its own "
                    "pre-event baseline; none preserves native values."
                ),
            )
            baseline_start = st.number_input(
                "Baseline start (s)",
                value=-5.0,
                disabled=normalization == "none",
                key=f"{workflow}_baseline_start",
                help=(
                    "Start of the pre-event interval used to calculate each trial's "
                    "baseline. Must fall within the plotted window and precede "
                    "Baseline end."
                ),
            )
            baseline_end = st.number_input(
                "Baseline end (s)",
                value=0.0,
                disabled=normalization == "none",
                key=f"{workflow}_baseline_end",
                help=(
                    "End of the baseline interval (exclusive). Zero uses samples "
                    "strictly before the alignment event."
                ),
            )
        fixed_y_axis = st.checkbox(
            "Use fixed y-axis limits for the primary PSTH",
            value=False,
            key=f"{workflow}_fixed_y_axis",
            help=(
                "Uses the exact same vertical range on this signal's PSTH, making "
                "figures directly comparable across analysis runs. Licking and running "
                "plots retain automatic scaling because they use different units."
            ),
        )
        y_limits = None
        y_limits_valid = True
        if fixed_y_axis:
            y_left, y_right = st.columns(2)
            with y_left:
                y_min = st.number_input(
                    "Y-axis minimum",
                    value=-100.0,
                    key=f"{workflow}_y_min",
                    help="Lowest value displayed on the primary PSTH y-axis.",
                )
            with y_right:
                y_max = st.number_input(
                    "Y-axis maximum",
                    value=100.0,
                    key=f"{workflow}_y_max",
                    help="Highest value displayed on the primary PSTH y-axis.",
                )
            y_limits_valid = y_min < y_max
            if y_limits_valid:
                y_limits = (y_min, y_max)
            else:
                st.error("Y-axis minimum must be less than the maximum.")
        selection_left, selection_right = st.columns(2)
        with selection_left:
            first_event_only = st.checkbox(
                "Align only to the first event in each session",
                value=False,
                key=f"{workflow}_first_event_only",
            )
        with selection_right:
            allow_partial_windows = st.checkbox(
                "Keep partial windows at recording boundaries",
                value=True,
                help=(
                    "Keeps the available samples and leaves the unrecorded tail blank "
                    "instead of excluding the event. The number of contributing "
                    "sessions can therefore decrease near the plot edges."
                ),
                key=f"{workflow}_allow_partial_windows",
            )
        if event == "visual_cue":
            class_left, class_right = st.columns(2)
            with class_left:
                trial_class = st.selectbox(
                    "Cue-trial class",
                    tuple(TRIAL_CLASS_LABELS),
                    format_func=TRIAL_CLASS_LABELS.get,
                    key=f"{workflow}_trial_class",
                    help="Classify visual-cue trials from cue and lick timestamps.",
                )
            with class_right:
                post_cue_window = st.number_input(
                    "Post-cue response window (seconds)",
                    min_value=0.0,
                    value=2.0,
                    step=0.5,
                    disabled=trial_class == "all",
                    key=f"{workflow}_post_cue_window",
                )
        else:
            trial_class = "all"
            post_cue_window = 2.0
        output = st.text_input(
            "Figure/output base directory",
            value=str(analysis_workspace / "psth"),
            key=f"{workflow}_psth_output",
        )
        suggested_run_label = suggest_psth_run_label(
            signal,
            event,
            (window_start, window_end),
            group_filter=psth_group_filter,
            condition_filter=psth_condition_filter,
        )
        run_label = st.text_input(
            "Run label (optional)",
            value="",
            placeholder=suggested_run_label,
            help=(
                "Used in the new run-folder name. Leave blank to use the suggested "
                "label derived from the selected subset, signal, event, and window."
            ),
            key=f"{workflow}_psth_run_label",
        )
        with st.expander("Advanced: output-folder behavior"):
            reuse_output_directory = st.checkbox(
                "Save directly into the base directory and replace matching files",
                value=False,
                key=f"{workflow}_reuse_psth_output",
                help=(
                    "Legacy behavior. Use only when you intentionally want to replace "
                    "figures from an earlier run."
                ),
            )
            if reuse_output_directory:
                st.warning(
                    "Matching files will be replaced, and unrelated files left by an "
                    "older run may remain in this directory."
                )
        resolved_run_label = run_label.strip() or suggested_run_label
        run_output = (
            Path(output).expanduser()
            if reuse_output_directory
            else next_analysis_run_directory(output, resolved_run_label)
        )
        st.caption(f"This run will be saved to: {run_output}")
        null_method = st.selectbox(
            "Random-alignment control",
            ("none", "random_onsets", "circular_shift"),
            key=f"{workflow}_null_method",
            help=(
                "Compare the response with session-local random or circularly shifted "
                "alignment times. Null controls are unavailable for licking responses."
            ),
        )
        null_left, null_middle, null_right = st.columns(3)
        with null_left:
            n_shuffles = st.number_input(
                "Shuffles",
                min_value=1,
                value=500,
                step=100,
                key=f"{workflow}_n_shuffles",
            )
        with null_middle:
            seed = st.number_input("Random seed", min_value=0, value=0, key=f"{workflow}_null_seed")
        with null_right:
            null_exclusion = st.number_input(
                "Exclude near real events (s)",
                min_value=0.0,
                value=0.0,
                step=0.5,
                disabled=null_method == "none",
                key=f"{workflow}_null_exclusion",
            )
        st.markdown("#### Additional behavioral responses")
        behavior_left, behavior_right = st.columns(2)
        with behavior_left:
            include_licking = st.checkbox(
                "Also generate licking PSTH",
                value=False,
                help="Saved in licks/s. Heatmaps are also saved when enabled below.",
                key=f"{workflow}_include_licking_psth",
            )
        with behavior_right:
            include_running = st.checkbox(
                "Also generate running PSTH",
                value=False,
                help=(
                    "Saved in cm/s. Sessions without a running signal are skipped. "
                    "Heatmaps are also saved when enabled below."
                ),
                key=f"{workflow}_include_running_psth",
            )
        save_heatmaps = st.checkbox(
            "Save session, mouse-level, and pooled-trial heatmaps",
            value=True,
            key=f"{workflow}_heatmaps",
        )
        sort_options = [
            "event_order",
            "response_mean",
            "ensure_latency",
            "first_lick_latency",
            "first_bout_latency",
            "bout_size",
            "bout_duration",
            "post_event_lick_count",
            "pre_event_lick_rate",
        ]
        if event in {"lick_bout_onset", "first_lick_bout_after_ensure"}:
            sort_options.append("cue_to_bout_latency")
        heatmap_sort = st.selectbox(
            "Heatmap trial order",
            sort_options,
            format_func=HEATMAP_SORT_LABELS.get,
            disabled=not save_heatmaps,
            key=f"{workflow}_heatmap_sort",
        )
        sort_window = default_heatmap_sort_window(heatmap_sort)
        if sort_window is not None:
            sort_left, sort_right = st.columns(2)
            with sort_left:
                sort_start = st.number_input(
                    "Sort matching-window start (s)",
                    value=float(sort_window[0]),
                    key=f"{workflow}_{heatmap_sort}_sort_start",
                )
            with sort_right:
                sort_end = st.number_input(
                    "Sort matching-window end (s)",
                    value=float(sort_window[1]),
                    key=f"{workflow}_{heatmap_sort}_sort_end",
                )
            sort_window = (sort_start, sort_end)
        heatmap_sort_direction = "auto"
        heatmap_unmatched = "bottom"
        heatmap_cmap = "coolwarm"
        with st.expander("Advanced: heatmap ordering details"):
            ordering_left, ordering_middle, ordering_right = st.columns(3)
            with ordering_left:
                heatmap_sort_direction = st.selectbox(
                    "Sort direction",
                    ("auto", "ascending", "descending"),
                    key=f"{workflow}_heatmap_sort_direction",
                    disabled=not save_heatmaps or heatmap_sort == "event_order",
                )
            with ordering_middle:
                heatmap_unmatched = st.selectbox(
                    "Trials without a match",
                    ("bottom", "exclude"),
                    key=f"{workflow}_heatmap_unmatched",
                    disabled=(
                        not save_heatmaps or heatmap_sort in {"event_order", "response_mean"}
                    ),
                )
            with ordering_right:
                heatmap_cmap = st.selectbox(
                    "Heatmap color map",
                    ("coolwarm", "RdBu_r", "seismic", "viridis"),
                    key=f"{workflow}_heatmap_cmap",
                    disabled=not save_heatmaps,
                )
        command = build_lifetime_command(
            PROJECT_ROOT,
            "psth",
            workflow,
            manifest_path,
            data_root,
            run_output,
            signal=signal,
            event=event,
            window=(window_start, window_end),
            dt=dt,
            normalization=normalization,
            baseline=(baseline_start, baseline_end),
            y_limits=y_limits,
            heatmaps=save_heatmaps,
            heatmap_sort=heatmap_sort,
            heatmap_sort_window=sort_window,
            heatmap_sort_direction=heatmap_sort_direction,
            heatmap_unmatched=heatmap_unmatched,
            heatmap_cmap=heatmap_cmap,
            null_method=("none" if signal == "licking" else null_method),
            n_shuffles=n_shuffles,
            seed=seed,
            null_exclusion=null_exclusion,
            trial_class=trial_class,
            post_cue_window=post_cue_window,
            derived_event_window=derived_event_window,
            first_event_only=first_event_only,
            allow_partial_windows=allow_partial_windows,
            group_filter=psth_group_filter,
            condition_filter=psth_condition_filter,
        )
        commands = [command]
        command_labels = [f"{signal} PSTH"]
        for behavior_signal, include, behavior_label in (
            ("licking", include_licking, "Licking PSTH"),
            ("running_speed", include_running, "Running PSTH"),
        ):
            if not include:
                continue
            commands.append(
                build_lifetime_command(
                    PROJECT_ROOT,
                    "psth",
                    workflow,
                    manifest_path,
                    data_root,
                    run_output / behavior_signal,
                    signal=behavior_signal,
                    event=event,
                    window=(window_start, window_end),
                    dt=dt,
                    normalization="none",
                    baseline=(baseline_start, baseline_end),
                    heatmaps=save_heatmaps,
                    heatmap_sort=heatmap_sort,
                    heatmap_sort_window=sort_window,
                    heatmap_sort_direction=heatmap_sort_direction,
                    heatmap_unmatched=heatmap_unmatched,
                    heatmap_cmap=heatmap_cmap,
                    null_method=("none" if behavior_signal == "licking" else null_method),
                    n_shuffles=n_shuffles,
                    seed=seed,
                    null_exclusion=null_exclusion,
                    trial_class=trial_class,
                    post_cue_window=post_cue_window,
                    derived_event_window=derived_event_window,
                    first_event_only=first_event_only,
                    allow_partial_windows=allow_partial_windows,
                    group_filter=psth_group_filter,
                    condition_filter=psth_condition_filter,
                )
            )
            command_labels.append(behavior_label)
        if st.button(
            "Run PSTH analysis",
            type="primary",
            width="stretch",
            key=f"{workflow}_run_psth",
            disabled=not y_limits_valid or not derived_event_window_valid,
        ):
            _run_lifetime_workflow(
                commands,
                workflow,
                rows,
                manifest_dir,
                manifest_name,
                active_manifest_key,
                data_root,
                "PSTH analysis",
                command_labels,
            )
        with st.expander("Advanced: inspect or copy command"):
            if st.button("Preview PSTH command", key=f"{workflow}_preview_psth"):
                for label, preview_command in zip(command_labels, commands, strict=True):
                    st.caption(label)
                    _preview_command(preview_command)

    with summary_tab:
        _psth_summary_controls(
            analysis_workspace / "psth",
            f"{workflow}_psth",
            summary_output_base=analysis_workspace / "summary_statistics",
        )

    with glm_tab:
        st.subheader("Lick and Ensure lifetime GLM")
        st.write(
            "Fits pre- and post-event kernels for licking and Ensure delivery, including "
            "one shared Ensure response across all deliveries."
        )
        st.caption(
            "Each session is median-centered so mouse, placement, and condition baseline "
            "differences do not shift the event kernels. Ridge strength is selected with "
            "leave-one-session-out validation; at least three sessions are required. "
            "Outputs include modulation performance, absolute prediction offsets, and a "
            "separate session-baseline analysis by mouse, group/placement, and condition. "
            "Lifetime kernels use picoseconds. Results describe predictive association, "
            "not causality."
        )
        glm_group_filter, glm_condition_filter = _analysis_subset_controls(rows, f"{workflow}_glm")
        glm_signal = st.selectbox(
            "GLM response",
            LIFETIME_SIGNALS[workflow],
            format_func=lambda value: LIFETIME_SIGNAL_LABELS.get(value, value),
            key=f"{workflow}_glm_signal",
            help=LIFETIME_GLM_RESPONSE_HELP[workflow],
        )
        glm_left, glm_middle, glm_right = st.columns(3)
        with glm_left:
            pre_event_kernel = st.number_input(
                "Pre-event kernel duration (s)",
                min_value=0.0,
                value=5.0,
                key=f"{workflow}_pre_event_kernel",
                help=(
                    "How far before each lick or Ensure delivery to estimate an association. "
                    "Negative-time effects are predictive associations, not effects caused "
                    "by the future event. Pre- and post-event basis functions are separated "
                    "at zero, and post-event kernels are constrained to zero at onset."
                ),
            )
        with glm_middle:
            lick_kernel = st.number_input(
                "Post-lick kernel duration (s)",
                min_value=1.0,
                value=10.0,
                key=f"{workflow}_lick_kernel",
                help=(
                    "How long after each lick the model estimates a possible association "
                    "with the selected lifetime response."
                ),
            )
        with glm_right:
            ensure_kernel = st.number_input(
                "Post-Ensure kernel duration (s)",
                min_value=1.0,
                value=20.0,
                key=f"{workflow}_ensure_kernel",
                help=(
                    "How long after each Ensure delivery the model estimates its response. "
                    "All Ensure deliveries contribute to one shared kernel."
                ),
            )
        glm_output = st.text_input(
            "GLM results base directory",
            value=str(analysis_workspace / "glm"),
            key=f"{workflow}_glm_output",
            help=("Each run is saved in a new timestamped folder below this directory."),
        )
        glm_run_output = next_analysis_run_directory(glm_output, f"{glm_signal}__lifetime_glm")
        st.caption(f"This run will be saved to: {glm_run_output}")
        command = build_lifetime_command(
            PROJECT_ROOT,
            "glm",
            workflow,
            manifest_path,
            data_root,
            glm_run_output,
            signal=glm_signal,
            pre_event_seconds=pre_event_kernel,
            lick_kernel_seconds=lick_kernel,
            ensure_kernel_seconds=ensure_kernel,
            group_filter=glm_group_filter,
            condition_filter=glm_condition_filter,
        )
        if st.button(
            "Run lifetime GLM",
            type="primary",
            width="stretch",
            key=f"{workflow}_run_glm",
        ):
            succeeded = _run_lifetime_workflow(
                command,
                workflow,
                rows,
                manifest_dir,
                manifest_name,
                active_manifest_key,
                data_root,
                "Lifetime GLM",
            )
            if succeeded:
                st.session_state[f"{workflow}_glm_result_directory"] = str(glm_run_output)
        with st.expander("Advanced: inspect or copy command"):
            if st.button("Preview GLM command", key=f"{workflow}_preview_glm"):
                _preview_command(command)
        glm_result_directory = st.session_state.get(f"{workflow}_glm_result_directory")
        if glm_result_directory:
            _show_glm_results(glm_result_directory, f"{workflow}_glm")


st.set_page_config(page_title="Lutas Lab Photometry", page_icon="📈", layout="wide")
st.title("Lutas Lab Photometry")
st.caption("One browser interface for the repository's maintained acquisition systems")

with st.sidebar:
    selected_workflow = st.radio(
        "Analysis system",
        ("conventional", "fluopulse", "iflip3"),
        format_func=lambda value: {
            "conventional": "Conventional photometry",
            "fluopulse": "FluoPulse lifetime",
            "iflip3": "iFLIP3 lifetime",
        }[value],
        help="Switching systems preserves the current table and controls for each mode.",
    )
    _show_update_status()

if selected_workflow != "conventional":
    _lifetime_gui(selected_workflow)
    st.stop()

with st.sidebar:
    st.subheader("Workspace")
    st.text_input("Repository", value=str(PROJECT_ROOT), disabled=True)
    data_root = st.text_input("Raw-data root", value=r"Z:\Photometry")
    conventional_legacy_manifest = PROJECT_ROOT / "analysis" / "sessions.csv"
    (
        conventional_analysis_workspace,
        conventional_manifest_dir,
        conventional_manifest_name,
        conventional_active_manifest_key,
    ) = _manifest_workspace_sidebar(
        "conventional",
        "conventional",
        legacy_paths=(conventional_legacy_manifest,),
    )
    manifest_path_text = st.session_state.get(conventional_active_manifest_key, "") or (
        conventional_manifest_dir / "unsaved_sessions.csv"
    )
    st.caption(
        "Preview and Run are separate actions. Run validates and saves the current "
        "session table first. Existing processed files are skipped unless overwrite "
        "is enabled."
    )

if "manifest_rows" not in st.session_state:
    st.session_state.manifest_rows = DEFAULT_ROWS

sessions_tab, preprocess_tab, trace_tab, psth_tab, summary_tab, glm_tab = st.tabs(
    [
        "1. Sessions",
        "2. Batch preprocessing",
        "3. Session trace",
        "4. Event-aligned PSTH",
        "5. Summary statistics",
        "6. Behavioral GLM",
    ]
)

with sessions_tab:
    st.subheader("Session manifest")
    _manifest_library_panel(
        "conventional",
        "manifest_rows",
        conventional_manifest_dir,
        conventional_active_manifest_key,
        legacy_paths=(conventional_legacy_manifest,),
    )
    st.write(
        "Enter one row per session. Group and condition are used for stratified "
        "figures; channel selects photoreceiver 1 or 2."
    )
    uploaded = st.file_uploader("Load an existing CSV", type="csv")
    if uploaded is not None and st.button("Use uploaded CSV"):
        st.session_state.manifest_rows = _uploaded_rows(uploaded)
        st.rerun()

    editor_frame = pd.DataFrame(
        st.session_state.manifest_rows,
        columns=MANIFEST_COLUMNS,
    )
    edited = st.data_editor(
        editor_frame,
        num_rows="dynamic",
        column_order=MANIFEST_COLUMNS,
        column_config={
            "mouse": st.column_config.TextColumn("Mouse", required=True),
            "date": st.column_config.TextColumn("Date (YYMMDD)", required=True),
            "run": st.column_config.NumberColumn("Run", min_value=0, step=1),
            "group": st.column_config.TextColumn("Group"),
            "condition": st.column_config.TextColumn("Condition"),
            "channel": st.column_config.SelectboxColumn(
                "Channel",
                options=[1, 2],
                required=True,
            ),
        },
        width="stretch",
        key="manifest_editor",
    )
    editor_rows = _records(edited)
    st.session_state.manifest_rows = editor_rows

    left, middle, update, right = st.columns(4)
    with left:
        if st.button("Validate sessions", width="stretch"):
            try:
                sessions = normalize_manifest_rows(editor_rows)
                st.success(f"Validated {len(sessions)} unique sessions.")
            except ValueError as error:
                st.error(str(error))
    with middle:
        if st.button("Save new version", type="primary", width="stretch"):
            try:
                saved_path = _save_manifest_version(
                    "conventional",
                    editor_rows,
                    conventional_manifest_dir,
                    conventional_manifest_name,
                    parent_manifest=(
                        st.session_state.get(conventional_active_manifest_key) or None
                    ),
                )
                st.session_state[conventional_active_manifest_key] = str(saved_path)
                st.success(f"Saved new manifest version: {saved_path}")
            except (OSError, ValueError) as error:
                st.error(str(error))
    with update:
        active_manifest = st.session_state.get(conventional_active_manifest_key, "")
        if st.button(
            "Update active",
            width="stretch",
            disabled=not active_manifest,
            help="Explicitly replace the active manifest instead of creating a version.",
        ):
            try:
                saved_path = write_manifest(active_manifest, editor_rows)
                write_manifest_metadata(
                    saved_path,
                    workflow="conventional",
                    display_name=conventional_manifest_name,
                    row_count=len(normalize_manifest_rows(editor_rows)),
                    code_commit=local_git_commit(PROJECT_ROOT),
                )
                st.success(f"Updated active manifest: {saved_path}")
            except (OSError, ValueError) as error:
                st.error(str(error))
    with right:
        try:
            csv_text = manifest_csv_text(editor_rows)
        except ValueError:
            csv_text = ""
        st.download_button(
            "Download CSV",
            data=csv_text,
            file_name=f"{safe_analysis_run_label(conventional_manifest_name)}.csv",
            mime="text/csv",
            disabled=not csv_text,
            width="stretch",
        )

with preprocess_tab:
    st.subheader("Batch preprocessing")
    st.write("Creates each processed `.npz` beside its original raw session files.")
    overwrite = st.checkbox("Overwrite existing processed files", value=False)
    continue_on_error = st.checkbox("Continue after a failed session", value=True)
    _processed_provenance_panel(
        "conventional",
        editor_rows,
        data_root,
        conventional_preprocessing_parameters(),
    )
    preprocess_command = build_preprocess_command(
        PROJECT_ROOT,
        manifest_path_text,
        data_root,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )
    if st.button("Run preprocessing", type="primary", width="stretch"):
        _run_workflow(
            preprocess_command,
            editor_rows,
            conventional_manifest_dir,
            conventional_manifest_name,
            conventional_active_manifest_key,
            data_root,
            "Preprocessing",
        )
    with st.expander("Advanced: inspect or copy command"):
        if st.button("Preview preprocessing command", width="stretch"):
            _preview_command(preprocess_command)

with trace_tab:
    _session_trace_controls(
        "conventional",
        editor_rows,
        data_root,
        conventional_manifest_dir,
        conventional_manifest_name,
        conventional_active_manifest_key,
    )

with psth_tab:
    st.subheader("Event-aligned timecourses")
    psth_group_filter, psth_condition_filter = _analysis_subset_controls(
        editor_rows, "conventional_psth"
    )
    first, second, third = st.columns(3)
    with first:
        event_key = st.selectbox(
            "Alignment event",
            (
                "cue_onset",
                "solenoid_onset",
                "lick_bout_onset",
                "lick_bout_offset",
                "lick_times",
                "first_lick_after_ensure",
                "first_lick_bout_after_ensure",
            ),
            help="Each occurrence becomes time zero for one peri-event trial.",
        )
        photometry_signal = st.selectbox(
            "Photometry signal",
            tuple(CONVENTIONAL_PHOTOMETRY_SIGNAL_LABELS),
            format_func=CONVENTIONAL_PHOTOMETRY_SIGNAL_LABELS.get,
            key="conventional_photometry_signal",
            help=(
                "IRLS ΔF/F is the standard corrected response. The 405 and fitted "
                "405 traces are controls for inspecting artifacts."
            ),
        )
        channel = st.selectbox(
            "Photoreceiver channel",
            ("manifest", "1", "2"),
            help=(
                "Use the channel recorded in each manifest row or override all "
                "sessions with channel 1 or 2."
            ),
        )
        if event_key in {
            "first_lick_after_ensure",
            "first_lick_bout_after_ensure",
        }:
            match_start = st.number_input(
                "Post-Ensure match start (s)",
                min_value=0.0,
                value=0.0,
                help="Earliest allowed latency after Ensure for the matched event.",
            )
            match_end = st.number_input(
                "Post-Ensure match end (s)",
                min_value=0.01,
                value=20.0,
                help=(
                    "Latest allowed latency. Matching stops at the next Ensure so one "
                    "lick or bout cannot be assigned across adjacent deliveries."
                ),
            )
            derived_event_window = (match_start, match_end)
            derived_event_window_valid = match_start < match_end
            if not derived_event_window_valid:
                st.error("Match-window start must be less than its end.")
        else:
            derived_event_window = (0.0, 20.0)
            derived_event_window_valid = True
    with second:
        window_start = st.number_input(
            "Window start (s)",
            value=-5.0,
            help=(
                "Earliest time retained relative to the alignment event; negative "
                "is before the event."
            ),
        )
        window_end = st.number_input(
            "Window end (s)",
            value=20.0,
            help=(
                "Latest time retained relative to the alignment event; positive is after the event."
            ),
        )
        dt = st.number_input(
            "Time bin (s)",
            min_value=0.001,
            value=0.02,
            format="%.3f",
            help=(
                "Spacing of the common time grid used to resample and average trials. "
                "Smaller bins preserve more temporal detail."
            ),
        )
    with third:
        normalization = st.selectbox(
            "Normalization",
            ("zscore", "subtract", "none"),
            help=(
                "Z-score and subtraction use each trial's selected pre-event baseline; "
                "none preserves the processed signal values."
            ),
        )
        baseline_start = st.number_input(
            "Baseline start (s)",
            value=-5.0,
            disabled=normalization == "none",
            help=(
                "Start of the pre-event interval used to calculate each trial's "
                "baseline. Must fall within the plotted window and precede Baseline end."
            ),
        )
        baseline_end = st.number_input(
            "Baseline end (s)",
            value=0.0,
            disabled=normalization == "none",
            help=(
                "End of the baseline interval (exclusive). Zero uses samples strictly "
                "before the alignment event."
            ),
        )

    fixed_y_axis = st.checkbox(
        "Use fixed y-axis limits for the primary PSTH",
        value=False,
        key="conventional_fixed_y_axis",
        help=(
            "Uses the exact same vertical range on photometry PSTHs and their "
            "condition-comparison panels. Licking and running plots retain automatic "
            "scaling because they use different units."
        ),
    )
    y_limits = None
    y_limits_valid = True
    if fixed_y_axis:
        y_left, y_right = st.columns(2)
        with y_left:
            y_min = st.number_input(
                "Y-axis minimum",
                value=-2.0,
                key="conventional_y_min",
                help="Lowest value displayed on the primary photometry PSTH y-axis.",
            )
        with y_right:
            y_max = st.number_input(
                "Y-axis maximum",
                value=2.0,
                key="conventional_y_max",
                help="Highest value displayed on the primary photometry PSTH y-axis.",
            )
        y_limits_valid = y_min < y_max
        if y_limits_valid:
            y_limits = (y_min, y_max)
        else:
            st.error("Y-axis minimum must be less than the maximum.")

    selection_left, selection_right = st.columns(2)
    with selection_left:
        first_event_only = st.checkbox("Align only to the first event in each session", value=False)
    with selection_right:
        allow_partial_windows = st.checkbox(
            "Keep partial windows at recording boundaries",
            value=True,
            help=(
                "Keeps the available samples and leaves the unrecorded tail blank "
                "instead of excluding the event. The number of contributing "
                "sessions can therefore decrease near the plot edges."
            ),
        )

    if event_key == "cue_onset":
        class_left, class_right = st.columns(2)
        with class_left:
            trial_class = st.selectbox(
                "Cue-trial class",
                tuple(TRIAL_CLASS_LABELS),
                format_func=TRIAL_CLASS_LABELS.get,
                help=(
                    "Recomputed from the saved cue and lick timestamps each time the analysis runs."
                ),
            )
        with class_right:
            post_cue_window = st.number_input(
                "Post-cue response window (seconds)",
                min_value=0.0,
                value=2.0,
                step=0.5,
                disabled=trial_class == "all",
                help="Time after cue offset used to classify post-cue licking.",
            )
    else:
        trial_class = "all"
        post_cue_window = 2.0
        st.caption("Cue-trial classification is available for cue-onset alignment.")

    output_dir = st.text_input(
        "Figure/output base directory",
        value=str(conventional_analysis_workspace / "psth"),
    )
    suggested_run_label = suggest_psth_run_label(
        photometry_signal,
        event_key,
        (window_start, window_end),
        group_filter=psth_group_filter,
        condition_filter=psth_condition_filter,
    )
    run_label = st.text_input(
        "Run label (optional)",
        value="",
        placeholder=suggested_run_label,
        help=(
            "Used in the new run-folder name. Leave blank to use the suggested "
            "label derived from the selected subset, signal, event, and window."
        ),
        key="conventional_psth_run_label",
    )
    with st.expander("Advanced: output-folder behavior"):
        reuse_output_directory = st.checkbox(
            "Save directly into the base directory and replace matching files",
            value=False,
            key="conventional_reuse_psth_output",
            help=(
                "Legacy behavior. Use only when you intentionally want to replace "
                "figures from an earlier run."
            ),
        )
        if reuse_output_directory:
            st.warning(
                "Matching files will be replaced, and unrelated files left by an "
                "older run may remain in this directory."
            )
    resolved_run_label = run_label.strip() or suggested_run_label
    run_output_dir = (
        Path(output_dir).expanduser()
        if reuse_output_directory
        else next_analysis_run_directory(output_dir, resolved_run_label)
    )
    st.caption(f"This run will be saved to: {run_output_dir}")
    stratify = st.checkbox("Separate group and condition", value=True)
    null_method = st.selectbox(
        "Random-alignment control",
        ("none", "random_onsets", "circular_shift"),
    )
    null_left, null_middle, null_right = st.columns(3)
    with null_left:
        n_shuffles = st.number_input("Shuffles", min_value=1, value=500, step=100)
    with null_middle:
        seed = st.number_input("Random seed", min_value=0, value=0, step=1)
    with null_right:
        null_exclusion = st.number_input(
            "Exclude near real events (s)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            disabled=null_method == "none",
            help=(
                "Minimum center-to-center distance between null and real alignment "
                "events. Large values may be impossible in dense sessions."
            ),
        )

    st.markdown("#### Additional behavioral responses")
    behavior_left, behavior_right = st.columns(2)
    with behavior_left:
        include_licking = st.checkbox(
            "Also generate licking PSTH",
            value=False,
            help="Saved in licks/s. Heatmaps are also saved when enabled below.",
            key="conventional_include_licking_psth",
        )
    with behavior_right:
        include_running = st.checkbox(
            "Also generate running PSTH",
            value=False,
            help=(
                "Saved in cm/s. Sessions without running data are skipped. "
                "Heatmaps are also saved when enabled below."
            ),
            key="conventional_include_running_psth",
        )
    st.markdown("#### Trial heatmaps")
    save_heatmaps = st.checkbox(
        "Save session, mouse-level, and pooled-trial heatmaps",
        value=True,
        help="Uses the same aligned trials as the corresponding PSTH.",
    )
    heatmap_left, heatmap_right = st.columns(2)
    with heatmap_left:
        heatmap_sort_options = [
            "event_order",
            "response_mean",
            "ensure_latency",
            "first_lick_latency",
            "first_bout_latency",
            "bout_size",
            "bout_duration",
            "post_event_lick_count",
            "pre_event_lick_rate",
        ]
        if event_key in {"lick_bout_onset", "first_lick_bout_after_ensure"}:
            heatmap_sort_options.append("cue_to_bout_latency")
        heatmap_sort = st.selectbox(
            "Heatmap trial order",
            heatmap_sort_options,
            format_func=HEATMAP_SORT_LABELS.get,
            disabled=not save_heatmaps,
        )
    with heatmap_right:
        heatmap_cmap = st.selectbox(
            "Heatmap color map",
            ("coolwarm", "RdBu_r", "seismic"),
            disabled=not save_heatmaps,
        )
    heatmap_sort_window = default_heatmap_sort_window(heatmap_sort)
    heatmap_sort_direction = "auto"
    heatmap_unmatched = "bottom"
    with st.expander("Advanced: heatmap ordering details"):
        if heatmap_sort_window is None:
            st.caption("This ordering does not require behavioral event matching.")
        else:
            st.caption(
                "Behavioral events are matched relative to each alignment time. "
                "The first eligible future event—or most recent cue for cue-to-bout "
                "latency—is used."
            )
            sort_window_left, sort_window_right = st.columns(2)
            with sort_window_left:
                sort_window_start = st.number_input(
                    "Matching window start (s)",
                    value=float(heatmap_sort_window[0]),
                    key=f"heatmap_sort_start_{heatmap_sort}",
                )
            with sort_window_right:
                sort_window_end = st.number_input(
                    "Matching window end (s)",
                    value=float(heatmap_sort_window[1]),
                    key=f"heatmap_sort_end_{heatmap_sort}",
                )
            heatmap_sort_window = (sort_window_start, sort_window_end)
        ordering_left, ordering_right = st.columns(2)
        with ordering_left:
            heatmap_sort_direction = st.selectbox(
                "Sort direction",
                ("auto", "ascending", "descending"),
                format_func=lambda value: {
                    "auto": "Automatic for measurement",
                    "ascending": "Low to high",
                    "descending": "High to low",
                }[value],
                disabled=not save_heatmaps or heatmap_sort == "event_order",
            )
        with ordering_right:
            heatmap_unmatched = st.selectbox(
                "Trials without a match",
                ("bottom", "exclude"),
                format_func=lambda value: {
                    "bottom": "Keep at bottom",
                    "exclude": "Exclude from heatmap",
                }[value],
                disabled=(not save_heatmaps or heatmap_sort in {"event_order", "response_mean"}),
            )

    psth_command = build_psth_command(
        PROJECT_ROOT,
        manifest_path_text,
        data_root,
        run_output_dir,
        event_key=event_key,
        signal="photometry",
        photometry_signal=photometry_signal,
        channel=channel,
        window=(window_start, window_end),
        dt=dt,
        normalization=normalization,
        baseline=(baseline_start, baseline_end),
        y_limits=y_limits,
        stratify=stratify,
        null_method=null_method,
        n_shuffles=n_shuffles,
        seed=seed,
        null_exclusion=null_exclusion,
        trial_class=trial_class,
        post_cue_window=post_cue_window,
        derived_event_window=derived_event_window,
        heatmaps=save_heatmaps,
        heatmap_sort=heatmap_sort,
        heatmap_sort_window=heatmap_sort_window,
        heatmap_sort_direction=heatmap_sort_direction,
        heatmap_unmatched=heatmap_unmatched,
        heatmap_cmap=heatmap_cmap,
        first_event_only=first_event_only,
        allow_partial_windows=allow_partial_windows,
        group_filter=psth_group_filter,
        condition_filter=psth_condition_filter,
    )
    psth_commands = [psth_command]
    psth_command_labels = [f"{CONVENTIONAL_PHOTOMETRY_SIGNAL_LABELS[photometry_signal]} PSTH"]
    for behavior_signal, include, behavior_label in (
        ("licking", include_licking, "Licking PSTH"),
        ("running", include_running, "Running PSTH"),
    ):
        if not include:
            continue
        psth_commands.append(
            build_psth_command(
                PROJECT_ROOT,
                manifest_path_text,
                data_root,
                run_output_dir / behavior_signal,
                event_key=event_key,
                signal=behavior_signal,
                channel=channel,
                window=(window_start, window_end),
                dt=dt,
                normalization="none",
                baseline=(baseline_start, baseline_end),
                derived_event_window=derived_event_window,
                stratify=stratify,
                null_method="none" if behavior_signal == "licking" else null_method,
                n_shuffles=n_shuffles,
                seed=seed,
                null_exclusion=null_exclusion,
                trial_class=trial_class,
                post_cue_window=post_cue_window,
                heatmaps=save_heatmaps,
                heatmap_sort=heatmap_sort,
                heatmap_sort_window=heatmap_sort_window,
                heatmap_sort_direction=heatmap_sort_direction,
                heatmap_unmatched=heatmap_unmatched,
                heatmap_cmap=heatmap_cmap,
                first_event_only=first_event_only,
                allow_partial_windows=allow_partial_windows,
                group_filter=psth_group_filter,
                condition_filter=psth_condition_filter,
            )
        )
        psth_command_labels.append(behavior_label)
    if st.button(
        "Run PSTH and plots",
        type="primary",
        width="stretch",
        disabled=not y_limits_valid or not derived_event_window_valid,
    ):
        _run_workflow(
            psth_commands,
            editor_rows,
            conventional_manifest_dir,
            conventional_manifest_name,
            conventional_active_manifest_key,
            data_root,
            "PSTH analysis",
            psth_command_labels,
        )
    with st.expander("Advanced: inspect or copy command"):
        if st.button("Preview PSTH command", width="stretch"):
            for label, preview_command in zip(psth_command_labels, psth_commands, strict=True):
                st.caption(label)
                _preview_command(preview_command)

with summary_tab:
    _psth_summary_controls(
        conventional_analysis_workspace / "psth",
        "conventional_psth",
        summary_output_base=conventional_analysis_workspace / "summary_statistics",
    )

with glm_tab:
    st.subheader("Behavioral GLM")
    st.write(
        "Fit contemporaneous photometry from locomotion, licking, cue, and reward "
        "signals. The workflow compares a constant baseline, photometry history, "
        "behavior-only, and combined ridge models using nested session-held-out "
        "validation."
    )
    st.caption(
        "Outputs include held-out performance and combined-model predictor kernels "
        "for photometry history, running, licking, cue onset, and Ensure onset. "
        "Kernel figures show individual-mouse traces and mouse mean ± SEM. These are "
        "predictive associations and do not establish causality."
    )
    glm_group_filter, glm_condition_filter = _analysis_subset_controls(
        editor_rows, "conventional_glm"
    )
    glm_first, glm_second, glm_third = st.columns(3)
    with glm_first:
        glm_channel = st.selectbox(
            "GLM photoreceiver channel",
            ("manifest", "1", "2"),
            help=(
                "Use the photoreceiver channel listed for each session in the manifest, "
                "or force channel 1 or 2 for every included session."
            ),
        )
        photometry_source = st.selectbox(
            "Photometry response",
            ("raw465", "dff"),
            format_func=lambda value: {
                "raw465": "Raw 465 fluorescence",
                "dff": "Processed dF/F",
            }[value],
            help=(
                "Signal predicted by the GLM. Processed dF/F is the artifact-corrected "
                "response used for standard analysis; raw 465 fluorescence retains "
                "brightness, bleaching, and shared motion effects."
            ),
        )
    with glm_second:
        glm_history = st.number_input(
            "Predictor kernel duration (s)",
            min_value=0.1,
            value=5.0,
            step=0.5,
            help=(
                "How far after each predictor the GLM estimates its association with "
                "photometry. For example, 5 s plots coefficients from 0 to 5 s after "
                "licks, cues, Ensure deliveries, and running measurements."
            ),
        )
        glm_lag_step = st.number_input(
            "Kernel point spacing (s)",
            min_value=0.01,
            value=0.5,
            step=0.1,
            help="Time spacing between adjacent points in each predictor kernel.",
        )
        glm_dt = st.number_input(
            "GLM time bin (s)",
            min_value=0.01,
            value=0.1,
            step=0.05,
            help="Common time grid used to bin events and resample continuous signals.",
        )
    with glm_third:
        st.info(
            "Validation holds out one complete session at a time. Ridge strength is "
            "selected using only the remaining sessions, matching the lifetime GLMs. "
            "At least three sessions are required."
        )

    glm_output_dir = st.text_input(
        "GLM results base directory",
        value=str(conventional_analysis_workspace / "glm"),
        help=("Each run is saved in a new timestamped folder below this directory."),
    )
    glm_run_output_dir = next_analysis_run_directory(
        glm_output_dir, f"{photometry_source}__behavior_glm"
    )
    st.caption(f"This run will be saved to: {glm_run_output_dir}")
    glm_command = build_behavior_glm_command(
        PROJECT_ROOT,
        manifest_path_text,
        data_root,
        glm_run_output_dir,
        channel=glm_channel,
        photometry_source=photometry_source,
        history=glm_history,
        lag_step=glm_lag_step,
        dt=glm_dt,
        validation="leave_one_session_out",
        group_filter=glm_group_filter,
        condition_filter=glm_condition_filter,
    )
    if st.button("Run behavioral GLM", type="primary", width="stretch"):
        succeeded = _run_workflow(
            glm_command,
            editor_rows,
            conventional_manifest_dir,
            conventional_manifest_name,
            conventional_active_manifest_key,
            data_root,
            "Behavioral GLM",
        )
        if succeeded:
            st.session_state["conventional_glm_result_directory"] = str(glm_run_output_dir)
    with st.expander("Advanced: inspect or copy command"):
        if st.button("Preview GLM command", width="stretch"):
            _preview_command(glm_command)
    conventional_glm_result = st.session_state.get("conventional_glm_result_directory")
    if conventional_glm_result:
        _show_glm_results(conventional_glm_result, "conventional_glm")
