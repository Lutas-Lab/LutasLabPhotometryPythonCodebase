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
        "4. Behavioral GLM",
    ]
    assert {button.label for button in app.button} >= {
        "Validate sessions",
        "Save manifest",
        "Preview preprocessing command",
        "Run preprocessing",
        "Preview PSTH command",
        "Run PSTH and plots",
        "Preview GLM command",
        "Run behavioral GLM",
    }


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
        "2. Align and export",
        "3. Event-aligned PSTH",
        "4. Lifetime GLM",
    ]
    assert app.selectbox(key=f"{workflow}_signal").value == expected_signal
    assert {button.label for button in app.button} >= {
        "Validate sessions",
        "Save manifest",
        "Run alignment and export",
        "Run PSTH and heatmaps",
        "Run lifetime GLM",
    }
