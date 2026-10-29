from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_streamlit_app_loads_without_exceptions():
    project_root = Path(__file__).resolve().parents[3]
    app = streamlit_testing.AppTest.from_file(
        str(project_root / "streamlit_app.py")
    ).run(timeout=45)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "1. Sessions",
        "2. Batch preprocessing",
        "3. Session trace",
        "4. Event-aligned PSTH",
        "5. Summary statistics",
        "6. Behavioral GLM",
    ]
    assert {button.label for button in app.button} >= {
        "Validate sessions",
        "Save manifest",
        "Preview preprocessing command",
        "Run preprocessing",
        "Load session trace",
        "Preview PSTH command",
        "Run PSTH and plots",
        "Preview saved PSTH",
        "Calculate metrics and Prism exports",
        "Preview summary command",
        "Preview GLM command",
        "Run behavioral GLM",
    }
    assert app.checkbox(key="conventional_include_licking_psth").value is False
    assert app.checkbox(key="conventional_include_running_psth").value is False
    assert app.checkbox(key="conventional_reuse_psth_output").value is False
    assert app.checkbox(key="conventional_fixed_y_axis").value is False
    assert app.selectbox(key="conventional_photometry_signal").value == "dff"
    assert app.multiselect(key="conventional_psth_summary_metrics").value == [
        "mean",
        "auc",
        "peak",
        "peak_latency",
    ]
    assert (
        app.selectbox(key="conventional_psth_summary_saved_run").label
        == "Saved PSTH run folder"
    )
    assert app.selectbox(key="conventional_trace_signal").label == "Signal"


def test_gui_source_summarizes_structured_running_qc():
    project_root = Path(__file__).resolve().parents[3]
    source = (project_root / "streamlit_app.py").read_text(encoding="utf-8")

    assert "parse_qc_warning_line(stripped)" in source
    assert "Running data were excluded from" in source
    assert "Affected running files" in source


def test_gui_source_renders_completed_psth_summary_results():
    project_root = Path(__file__).resolve().parents[3]
    source = (project_root / "streamlit_app.py").read_text(encoding="utf-8")

    assert 'st.subheader("Latest summary results")' in source
    assert '"Per-trial metrics"' in source
    assert "psth_group_summary.csv" in source
    assert "psth_statistical_tests.csv" in source
    assert "psth_trial_metric_group_summary.csv" in source
    assert "summary_download_" not in source


def test_gui_source_renders_persistent_glm_results_and_timestamped_runs():
    project_root = Path(__file__).resolve().parents[3]
    source = (project_root / "streamlit_app.py").read_text(encoding="utf-8")

    assert 'st.subheader("Latest GLM results")' in source
    assert '"Held-out predictions"' in source
    assert "glm_heldout_predictions.csv" in source
    assert "forecast_heldout_predictions.csv" in source
    assert "next_analysis_run_directory(" in source


@pytest.mark.parametrize(
    ("workflow", "expected_signal"),
    [("fluopulse", "tau"), ("iflip3", "mpet")],
)
def test_streamlit_lifetime_modes_load(workflow, expected_signal):
    project_root = Path(__file__).resolve().parents[3]
    app = streamlit_testing.AppTest.from_file(
        str(project_root / "streamlit_app.py")
    ).run(timeout=45)
    app.radio[0].set_value(workflow)
    app.run(timeout=45)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "1. Sessions",
        "2. Preprocess and align",
        "3. Session trace",
        "4. Event-aligned PSTH",
        "5. Summary statistics",
        "6. Lifetime GLM",
    ]
    assert app.selectbox(key=f"{workflow}_signal").value == expected_signal
    assert app.selectbox(key=f"{workflow}_normalization").value == "subtract"
    assert {button.label for button in app.button} >= {
        "Validate sessions",
        "Save manifest",
        "Run preprocessing",
        "Run PSTH analysis",
        "Calculate metrics and Prism exports",
        "Run lifetime GLM",
    }
    if workflow == "iflip3":
        assert {button.label for button in app.button} >= {
            "Run multi-session calibration",
            "Load saved calibration",
        }
    assert app.checkbox(key=f"{workflow}_include_licking_psth").value is False
    assert app.checkbox(key=f"{workflow}_include_running_psth").value is False
    assert app.checkbox(key=f"{workflow}_reuse_psth_output").value is False
    assert app.checkbox(key=f"{workflow}_fixed_y_axis").value is False
    assert app.multiselect(key=f"{workflow}_psth_summary_metrics").value == [
        "mean",
        "auc",
        "peak",
        "peak_latency",
    ]
    assert (
        app.selectbox(key=f"{workflow}_psth_summary_saved_run").label
        == "Saved PSTH run folder"
    )
