from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_streamlit_app_loads_without_exceptions():
    project_root = Path(__file__).resolve().parents[3]
    app = streamlit_testing.AppTest.from_file(
        str(project_root / "streamlit_app.py")
    ).run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "1. Sessions",
        "2. Batch preprocessing",
        "3. Event-aligned PSTH",
        "4. Summary statistics",
        "5. Behavioral GLM",
    ]
    assert {button.label for button in app.button} >= {
        "Validate sessions",
        "Save manifest",
        "Preview preprocessing command",
        "Run preprocessing",
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
    assert "Signal" not in {selectbox.label for selectbox in app.selectbox}


@pytest.mark.parametrize(
    ("workflow", "expected_signal"),
    [("fluopulse", "tau"), ("iflip3", "mpet")],
)
def test_streamlit_lifetime_modes_load(workflow, expected_signal):
    project_root = Path(__file__).resolve().parents[3]
    app = streamlit_testing.AppTest.from_file(
        str(project_root / "streamlit_app.py")
    ).run(timeout=30)
    app.radio[0].set_value(workflow)
    app.run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "1. Sessions",
        "2. Preprocess and align",
        "3. Event-aligned PSTH",
        "4. Summary statistics",
        "5. Lifetime GLM",
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
