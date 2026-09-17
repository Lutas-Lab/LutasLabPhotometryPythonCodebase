"""Streamlit prototype for the Lutas Lab photometry workflows."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.gui_workflows import (
    MANIFEST_COLUMNS,
    build_preprocess_command,
    build_psth_command,
    display_command,
    manifest_csv_text,
    normalize_manifest_rows,
    run_command,
    write_manifest,
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


def _show_command(command, preview_only):
    st.code(display_command(command), language="powershell")
    if preview_only:
        st.info("Preview only is enabled; no analysis was run.")
        return
    with st.spinner("Running workflow. Keep this browser tab open..."):
        return_code, output = run_command(command, PROJECT_ROOT)
    st.code(output or "(No console output)", language="text")
    if return_code == 0:
        st.success("Workflow finished successfully.")
    else:
        st.error(f"Workflow exited with code {return_code}.")


st.set_page_config(page_title="Lutas Lab Photometry", page_icon="📈", layout="wide")
st.title("Lutas Lab Photometry")
st.caption("Prototype interface for the repository's maintained analysis scripts")

with st.sidebar:
    st.subheader("Workspace")
    st.text_input("Repository", value=str(PROJECT_ROOT), disabled=True)
    data_root = st.text_input("Raw-data root", value=r"Z:\Photometry")
    manifest_path_text = st.text_input(
        "Session manifest",
        value=str(PROJECT_ROOT / "analysis" / "sessions.csv"),
    )
    preview_only = st.toggle(
        "Preview analysis commands only",
        value=True,
        help="Disable this to run preprocessing or PSTH analysis.",
    )
    st.caption(
        "This setting does not affect Validate, Save manifest, or Download CSV. "
        "Existing processed files are skipped unless overwrite is enabled."
    )

if "manifest_rows" not in st.session_state:
    st.session_state.manifest_rows = DEFAULT_ROWS

sessions_tab, preprocess_tab, psth_tab = st.tabs(
    ["1. Sessions", "2. Batch preprocessing", "3. Event-aligned PSTH"]
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
        use_container_width=True,
        key="manifest_editor",
    )
    editor_rows = _records(edited)
    st.session_state.manifest_rows = editor_rows

    left, middle, right = st.columns(3)
    with left:
        if st.button("Validate sessions", use_container_width=True):
            try:
                sessions = normalize_manifest_rows(editor_rows)
                st.success(f"Validated {len(sessions)} unique sessions.")
            except ValueError as error:
                st.error(str(error))
    with middle:
        if st.button("Save manifest", type="primary", use_container_width=True):
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
            use_container_width=True,
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
    if st.button("Preview / run preprocessing", type="primary"):
        _show_command(preprocess_command, preview_only)

with psth_tab:
    st.subheader("Event-aligned timecourses")
    first, second, third = st.columns(3)
    with first:
        event_key = st.selectbox(
            "Alignment event",
            ("cue_onset", "solenoid_onset", "lick_bout_onset", "lick_times"),
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
    null_left, null_right = st.columns(2)
    with null_left:
        n_shuffles = st.number_input("Shuffles", min_value=1, value=500, step=100)
    with null_right:
        seed = st.number_input("Random seed", min_value=0, value=0, step=1)

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
        trial_class=trial_class,
        post_cue_window=post_cue_window,
    )
    if st.button("Preview / run PSTH", type="primary"):
        _show_command(psth_command, preview_only)
