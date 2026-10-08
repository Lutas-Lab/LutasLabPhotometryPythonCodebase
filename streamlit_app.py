"""Browser interface for the Lutas Lab photometry workflows."""

from __future__ import annotations

import csv
import io
from collections import deque
from pathlib import Path

import pandas as pd
import streamlit as st

from lutaslab_photometry.gui_workflows import (
    MANIFEST_COLUMNS,
    build_behavior_glm_command,
    build_lifetime_command,
    build_preprocess_command,
    build_psth_command,
    display_command,
    manifest_csv_text,
    normalize_manifest_rows,
    run_command,
    write_manifest,
)
from lutaslab_photometry.heatmap_ordering import (
    HEATMAP_SORT_LABELS,
    default_heatmap_sort_window,
)
from lutaslab_photometry.lifetime_workflows import (
    LIFETIME_EVENTS,
    LIFETIME_SIGNALS,
    discover_lifetime_paths,
    lifetime_manifest_columns,
    normalize_lifetime_manifest_rows,
    write_lifetime_manifest,
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


def _uploaded_rows(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _records(editor_value):
    if hasattr(editor_value, "to_dict"):
        return editor_value.to_dict(orient="records")
    return list(editor_value)


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


def _run_workflow(command, editor_rows, manifest_path, data_root, workflow_name):
    data_root_path = Path(data_root).expanduser()
    if not data_root_path.is_dir():
        st.error(f"Raw-data root does not exist or is not a directory: {data_root_path}")
        return

    try:
        saved_path = write_manifest(manifest_path, editor_rows)
    except (OSError, ValueError) as error:
        st.error(f"The session manifest was not saved: {error}")
        return

    st.success(f"Validated and saved the current session table to {saved_path}")
    output_lines = deque(maxlen=500)
    output_panel = st.empty()

    def show_output(line):
        output_lines.append(line.rstrip("\r\n"))
        output_panel.code(
            "\n".join(output_lines) or "(Waiting for console output...)",
            language="text",
        )

    with st.spinner("Running workflow. Keep this browser tab open..."):
        try:
            return_code, output = run_command(command, PROJECT_ROOT, show_output)
        except OSError as error:
            st.error(f"{workflow_name} could not start: {error}")
            return
    if not output_lines:
        output_panel.code(output or "(No console output)", language="text")
    if return_code == 0:
        st.success(f"{workflow_name} finished successfully.")
    else:
        st.error(f"{workflow_name} exited with code {return_code}.")


def _run_lifetime_workflow(
    command,
    workflow,
    editor_rows,
    manifest_path,
    data_root,
    workflow_name,
):
    data_root_path = Path(data_root).expanduser()
    if not data_root_path.is_dir():
        st.error(f"Data root does not exist or is not a directory: {data_root_path}")
        return
    try:
        saved_path = write_lifetime_manifest(workflow, manifest_path, editor_rows)
    except (OSError, ValueError) as error:
        st.error(f"The session manifest was not saved: {error}")
        return
    st.success(f"Validated and saved the current session table to {saved_path}")
    output_lines = deque(maxlen=500)
    output_panel = st.empty()

    def show_output(line):
        output_lines.append(line.rstrip("\r\n"))
        output_panel.code("\n".join(output_lines), language="text")

    with st.spinner("Running workflow. Keep this browser tab open..."):
        try:
            return_code, output = run_command(command, PROJECT_ROOT, show_output)
        except OSError as error:
            st.error(f"{workflow_name} could not start: {error}")
            return
    if not output_lines:
        output_panel.code(output or "(No console output)", language="text")
    if return_code == 0:
        st.success(f"{workflow_name} finished successfully.")
    else:
        st.error(f"{workflow_name} exited with code {return_code}.")


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
        manifest_path = st.text_input(
            "Session manifest",
            value=str(PROJECT_ROOT / "analysis" / workflow / "sessions.csv"),
            key=f"{workflow}_manifest_path",
        )
        st.caption(
            "Blank recording and NI-DAQ paths use the laboratory mouse/date/run "
            "conventions under the data root. Explicit paths override discovery."
        )

    sessions_tab, align_tab, psth_tab, glm_tab = st.tabs(
        [
            "1. Sessions",
            "2. Preprocess and align",
            "3. Event-aligned PSTH",
            "4. Lifetime GLM",
        ]
    )
    with sessions_tab:
        st.subheader(f"{label} session manifest")
        if workflow == "iflip3":
            st.info(
                "The path assistant finds session and background candidates from mouse, "
                "date, and run. You still choose the matched iFLIP3 background used for "
                "background and afterpulse correction."
            )
        else:
            st.info(
                "The path assistant finds Doric and NI-DAQ files from mouse, date, and "
                "run, including standard names with descriptive suffix text."
            )

        with st.expander("Add a session and find its files", expanded=True):
            identity_columns = st.columns((1.2, 1.2, 0.8, 1, 1))
            with identity_columns[0]:
                discovery_mouse = st.text_input(
                    "Mouse", key=f"{workflow}_discovery_mouse"
                )
            with identity_columns[1]:
                discovery_date = st.text_input(
                    "Date (YYMMDD)", key=f"{workflow}_discovery_date"
                )
            with identity_columns[2]:
                discovery_run = st.number_input(
                    "Run",
                    min_value=0,
                    step=1,
                    value=1,
                    key=f"{workflow}_discovery_run",
                )
            with identity_columns[3]:
                discovery_group = st.text_input(
                    "Group", key=f"{workflow}_discovery_group"
                )
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
                    "NI-DAQ file",
                    choices["nidaq_path"],
                    data_root,
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
                        st.warning(
                            "No likely background was found. Enter the matched background "
                            "filename or path below."
                        )
                    background_path = _path_picker(
                        workflow,
                        "background_path",
                        "Matched background",
                        background_candidates,
                        data_root,
                    )

                if st.button(
                    "Add or update manifest row",
                    width="stretch",
                    key=f"{workflow}_add_discovered",
                ):
                    if not recording_path:
                        st.error(f"Choose or enter the {recording_label} path.")
                    elif not nidaq_path:
                        st.error("Choose or enter the NI-DAQ path.")
                    elif workflow == "iflip3" and not background_path:
                        st.error("Choose or enter the matched background path.")
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

        uploaded = st.file_uploader(
            "Load an existing CSV", type="csv", key=f"{workflow}_upload"
        )
        if uploaded is not None and st.button(
            "Use uploaded CSV", key=f"{workflow}_use_upload"
        ):
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
                    "Matched background path", required=workflow == "iflip3"
                ),
            },
            width="stretch",
            key=f"{workflow}_manifest_editor",
        )
        rows = _records(edited)
        st.session_state[state_key] = rows
        left, middle, right = st.columns(3)
        with left:
            if st.button("Validate sessions", width="stretch", key=f"{workflow}_validate"):
                try:
                    sessions = normalize_lifetime_manifest_rows(workflow, rows)
                    st.success(f"Validated {len(sessions)} unique sessions.")
                except ValueError as error:
                    st.error(str(error))
        with middle:
            if st.button(
                "Save manifest",
                type="primary",
                width="stretch",
                key=f"{workflow}_save",
            ):
                try:
                    saved = write_lifetime_manifest(workflow, manifest_path, rows)
                    st.success(f"Saved {saved}")
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
                file_name=f"{workflow}_sessions.csv",
                mime="text/csv",
                disabled=not csv_text,
                width="stretch",
                key=f"{workflow}_download",
            )

    with align_tab:
        st.subheader("Preprocess and align lifetime recordings")
        st.write(
            "Loads each lifetime recording, aligns it to NI-DAQ synchronization pulses, "
            "and saves an analysis-ready compressed NPZ file beside the primary raw "
            "recording. PSTH, heatmap, and GLM analyses use this processed file."
        )
        overwrite_processed = st.checkbox(
            "Overwrite existing processed files",
            value=False,
            key=f"{workflow}_overwrite_processed",
        )
        command = build_lifetime_command(
            PROJECT_ROOT,
            "preprocess",
            workflow,
            manifest_path,
            data_root,
            overwrite=overwrite_processed,
        )
        if st.button(
            "Run preprocessing",
            type="primary",
            width="stretch",
            key=f"{workflow}_run_align",
        ):
            _run_lifetime_workflow(
                command, workflow, rows, manifest_path, data_root, "Preprocessing"
            )
        with st.expander("Advanced: inspect or copy command"):
            if st.button("Preview preprocessing command", key=f"{workflow}_preview_align"):
                _preview_command(command)

    with psth_tab:
        st.subheader("Event-aligned lifetime timecourses")
        first, second, third = st.columns(3)
        with first:
            signal = st.selectbox(
                "Lifetime/QC signal", LIFETIME_SIGNALS[workflow], key=f"{workflow}_signal"
            )
            event = st.selectbox(
                "Alignment event", LIFETIME_EVENTS, key=f"{workflow}_event"
            )
        with second:
            window_start = st.number_input(
                "Window start (s)", value=-5.0, key=f"{workflow}_window_start"
            )
            window_end = st.number_input(
                "Window end (s)", value=20.0, key=f"{workflow}_window_end"
            )
            dt = st.number_input(
                "Time bin (s)", min_value=0.01, value=0.1, key=f"{workflow}_dt"
            )
        with third:
            normalization = st.selectbox(
                "Normalization",
                ("zscore", "subtract", "none"),
                key=f"{workflow}_normalization",
            )
            baseline_start = st.number_input(
                "Baseline start (s)", value=-5.0, key=f"{workflow}_baseline_start"
            )
            baseline_end = st.number_input(
                "Baseline end (s)", value=0.0, key=f"{workflow}_baseline_end"
            )
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
        output = st.text_input(
            "Figure/output directory",
            value=str(PROJECT_ROOT / "analysis" / workflow / "psth"),
            key=f"{workflow}_psth_output",
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
            "post_event_lick_count",
            "pre_event_lick_rate",
        ]
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
        command = build_lifetime_command(
            PROJECT_ROOT,
            "psth",
            workflow,
            manifest_path,
            data_root,
            output,
            signal=signal,
            event=event,
            window=(window_start, window_end),
            dt=dt,
            normalization=normalization,
            baseline=(baseline_start, baseline_end),
            heatmaps=save_heatmaps,
            heatmap_sort=heatmap_sort,
            heatmap_sort_window=sort_window,
            first_event_only=first_event_only,
            allow_partial_windows=allow_partial_windows,
        )
        if st.button(
            "Run PSTH and heatmaps",
            type="primary",
            width="stretch",
            key=f"{workflow}_run_psth",
        ):
            _run_lifetime_workflow(
                command, workflow, rows, manifest_path, data_root, "PSTH analysis"
            )
        with st.expander("Advanced: inspect or copy command"):
            if st.button("Preview PSTH command", key=f"{workflow}_preview_psth"):
                _preview_command(command)

    with glm_tab:
        st.subheader("Lick and Ensure lifetime GLM")
        st.write(
            "Fits forward-time event kernels for licking and Ensure delivery, including a "
            "separate first-Ensure effect and adaptation across later deliveries."
        )
        st.caption(
            "Ridge strength is selected with leave-one-session-out validation. At least "
            "three sessions are required. Results describe predictive association, not causality."
        )
        glm_signal = st.selectbox(
            "GLM response", LIFETIME_SIGNALS[workflow], key=f"{workflow}_glm_signal"
        )
        glm_left, glm_right = st.columns(2)
        with glm_left:
            lick_kernel = st.number_input(
                "Lick kernel duration (s)",
                min_value=1.0,
                value=10.0,
                key=f"{workflow}_lick_kernel",
            )
        with glm_right:
            ensure_kernel = st.number_input(
                "Ensure kernel duration (s)",
                min_value=1.0,
                value=20.0,
                key=f"{workflow}_ensure_kernel",
            )
        glm_output = st.text_input(
            "GLM output directory",
            value=str(PROJECT_ROOT / "analysis" / workflow / "glm"),
            key=f"{workflow}_glm_output",
        )
        command = build_lifetime_command(
            PROJECT_ROOT,
            "glm",
            workflow,
            manifest_path,
            data_root,
            glm_output,
            signal=glm_signal,
            lick_kernel_seconds=lick_kernel,
            ensure_kernel_seconds=ensure_kernel,
        )
        if st.button(
            "Run lifetime GLM",
            type="primary",
            width="stretch",
            key=f"{workflow}_run_glm",
        ):
            _run_lifetime_workflow(
                command, workflow, rows, manifest_path, data_root, "Lifetime GLM"
            )
        with st.expander("Advanced: inspect or copy command"):
            if st.button("Preview GLM command", key=f"{workflow}_preview_glm"):
                _preview_command(command)


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

if selected_workflow != "conventional":
    _lifetime_gui(selected_workflow)
    st.stop()

with st.sidebar:
    st.subheader("Workspace")
    st.text_input("Repository", value=str(PROJECT_ROOT), disabled=True)
    data_root = st.text_input("Raw-data root", value=r"Z:\Photometry")
    manifest_path_text = st.text_input(
        "Session manifest",
        value=str(PROJECT_ROOT / "analysis" / "sessions.csv"),
    )
    st.caption(
        "Preview and Run are separate actions. Run validates and saves the current "
        "session table first. Existing processed files are skipped unless overwrite "
        "is enabled."
    )

if "manifest_rows" not in st.session_state:
    st.session_state.manifest_rows = DEFAULT_ROWS

sessions_tab, preprocess_tab, psth_tab, glm_tab = st.tabs(
    [
        "1. Sessions",
        "2. Batch preprocessing",
        "3. Event-aligned PSTH",
        "4. Behavioral GLM",
    ]
)

with sessions_tab:
    st.subheader("Session manifest")
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

    left, middle, right = st.columns(3)
    with left:
        if st.button("Validate sessions", width="stretch"):
            try:
                sessions = normalize_manifest_rows(editor_rows)
                st.success(f"Validated {len(sessions)} unique sessions.")
            except ValueError as error:
                st.error(str(error))
    with middle:
        if st.button("Save manifest", type="primary", width="stretch"):
            try:
                saved_path = write_manifest(manifest_path_text, editor_rows)
                st.success(f"Saved {saved_path}")
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
            file_name="sessions.csv",
            mime="text/csv",
            disabled=not csv_text,
            width="stretch",
        )

with preprocess_tab:
    st.subheader("Batch preprocessing")
    st.write("Creates each processed `.npz` beside its original raw session files.")
    overwrite = st.checkbox("Overwrite existing processed files", value=False)
    continue_on_error = st.checkbox("Continue after a failed session", value=True)
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
            manifest_path_text,
            data_root,
            "Preprocessing",
        )
    with st.expander("Advanced: inspect or copy command"):
        if st.button("Preview preprocessing command", width="stretch"):
            _preview_command(preprocess_command)

with psth_tab:
    st.subheader("Event-aligned timecourses")
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
            ),
        )
        signal = st.selectbox("Signal", ("photometry", "licking"))
        channel = st.selectbox("Photoreceiver channel", ("manifest", "1", "2"))
    with second:
        window_start = st.number_input("Window start (s)", value=-5.0)
        window_end = st.number_input("Window end (s)", value=20.0)
        dt = st.number_input("Time bin (s)", min_value=0.001, value=0.02, format="%.3f")
    with third:
        normalization = st.selectbox("Normalization", ("zscore", "subtract", "none"))
        baseline_start = st.number_input("Baseline start (s)", value=-5.0)
        baseline_end = st.number_input("Baseline end (s)", value=0.0)

    selection_left, selection_right = st.columns(2)
    with selection_left:
        first_event_only = st.checkbox(
            "Align only to the first event in each session", value=False
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
        )

    if event_key == "cue_onset":
        class_left, class_right = st.columns(2)
        with class_left:
            trial_class = st.selectbox(
                "Cue-trial class",
                tuple(TRIAL_CLASS_LABELS),
                format_func=TRIAL_CLASS_LABELS.get,
                help=(
                    "Recomputed from the saved cue and lick timestamps each time "
                    "the analysis runs."
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
        "Figure/output directory",
        value=str(PROJECT_ROOT / "analysis" / "gui_psth"),
    )
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

    st.markdown("#### Trial heatmaps")
    save_heatmaps = st.checkbox(
        "Save session, mouse-level, and pooled-trial heatmaps",
        value=True,
        help="Uses the same aligned and normalized trials as the PSTH.",
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
        if event_key == "lick_bout_onset":
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
                disabled=(
                    not save_heatmaps
                    or heatmap_sort in {"event_order", "response_mean"}
                ),
            )

    psth_command = build_psth_command(
        PROJECT_ROOT,
        manifest_path_text,
        data_root,
        output_dir,
        event_key=event_key,
        signal=signal,
        channel=channel,
        window=(window_start, window_end),
        dt=dt,
        normalization=normalization,
        baseline=(baseline_start, baseline_end),
        stratify=stratify,
        null_method=null_method,
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
    )
    if st.button("Run PSTH and plots", type="primary", width="stretch"):
        _run_workflow(
            psth_command,
            editor_rows,
            manifest_path_text,
            data_root,
            "PSTH analysis",
        )
    with st.expander("Advanced: inspect or copy command"):
        if st.button("Preview PSTH command", width="stretch"):
            _preview_command(psth_command)

with glm_tab:
    st.subheader("Behavioral GLM")
    st.write(
        "Fit contemporaneous photometry from locomotion, licking, cue, and reward "
        "signals. The workflow compares a constant baseline, photometry history, "
        "behavior-only, and combined ridge models using nested forward validation."
    )
    st.caption(
        "This is a predictive model comparison. It does not establish that a "
        "behavior causes the photometry response."
    )
    glm_first, glm_second, glm_third = st.columns(3)
    with glm_first:
        glm_channel = st.selectbox(
            "GLM photoreceiver channel",
            ("manifest", "1", "2"),
        )
        photometry_source = st.selectbox(
            "Photometry response",
            ("raw465", "dff"),
            format_func=lambda value: {
                "raw465": "Raw 465 fluorescence",
                "dff": "Processed dF/F",
            }[value],
        )
    with glm_second:
        glm_history = st.number_input(
            "Predictor history (s)", min_value=0.1, value=5.0, step=0.5
        )
        glm_lag_step = st.number_input(
            "Lag spacing (s)", min_value=0.01, value=0.5, step=0.1
        )
        glm_dt = st.number_input(
            "GLM time bin (s)", min_value=0.01, value=0.1, step=0.05
        )
    with glm_third:
        glm_folds = st.number_input(
            "Outer validation folds", min_value=2, value=5, step=1
        )
        glm_inner_folds = st.number_input(
            "Inner alpha-selection folds", min_value=2, value=3, step=1
        )

    glm_output_dir = st.text_input(
        "GLM output directory",
        value=str(PROJECT_ROOT / "analysis" / "gui_glm"),
    )
    glm_command = build_behavior_glm_command(
        PROJECT_ROOT,
        manifest_path_text,
        data_root,
        glm_output_dir,
        channel=glm_channel,
        photometry_source=photometry_source,
        history=glm_history,
        lag_step=glm_lag_step,
        dt=glm_dt,
        folds=glm_folds,
        inner_folds=glm_inner_folds,
    )
    if st.button("Run behavioral GLM", type="primary", width="stretch"):
        _run_workflow(
            glm_command,
            editor_rows,
            manifest_path_text,
            data_root,
            "Behavioral GLM",
        )
    with st.expander("Advanced: inspect or copy command"):
        if st.button("Preview GLM command", width="stretch"):
            _preview_command(glm_command)
