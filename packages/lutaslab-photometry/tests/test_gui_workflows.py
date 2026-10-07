import csv
import io
import sys
from pathlib import Path

import pytest

from lutaslab_photometry.gui_workflows import (
    build_behavior_glm_command,
    build_lifetime_command,
    build_preprocess_command,
    build_psth_command,
    manifest_csv_text,
    normalize_manifest_rows,
    run_command,
    write_manifest,
)


def sample_rows():
    return [
        {
            "mouse": "DK130",
            "date": "250912",
            "run": "1",
            "group": "Astrocyte",
            "condition": "Naive",
            "channel": "2",
        }
    ]


def test_normalize_manifest_rows_preserves_analysis_fields():
    rows = normalize_manifest_rows(sample_rows())
    assert rows == [
        {
            "mouse": "DK130",
            "date": "250912",
            "run": 1,
            "group": "Astrocyte",
            "condition": "Naive",
            "channel": 2,
        }
    ]


@pytest.mark.parametrize("channel", ["0", "3", "bad"])
def test_normalize_manifest_rows_rejects_invalid_channel(channel):
    rows = sample_rows()
    rows[0]["channel"] = channel
    with pytest.raises(ValueError, match="channel must be 1 or 2"):
        normalize_manifest_rows(rows)


def test_normalize_manifest_rows_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        normalize_manifest_rows(sample_rows() * 2)


def test_manifest_csv_uses_core_manifest_columns():
    text = manifest_csv_text(sample_rows())
    sessions = list(csv.DictReader(io.StringIO(text)))
    assert sessions[0]["mouse"] == "DK130"
    assert sessions[0]["run"] == "1"
    assert sessions[0]["channel"] == "2"
    assert text.startswith("mouse,date,run")


def test_write_manifest_creates_parent_and_leaves_no_temporary_file(tmp_path):
    manifest_path = tmp_path / "analysis" / "sessions.csv"
    assert write_manifest(manifest_path, sample_rows()) == manifest_path.resolve()
    assert manifest_path.read_text(encoding="utf-8").startswith("mouse,date,run")
    assert list(manifest_path.parent.glob("*.tmp")) == []


def test_build_preprocess_command_uses_current_python_and_safe_arguments():
    command = build_preprocess_command(
        Path("project"),
        Path("analysis/sessions.csv"),
        Path("Z:/Photometry Data"),
        overwrite=True,
        post_cue_window=20,
    )
    assert command[:3] == [
        command[0],
        "-m",
        "lutaslab_photometry.cli.run_preprocess_batch",
    ]
    assert command[command.index("--data-root") + 1] == str(Path("Z:/Photometry Data"))
    assert "--overwrite" in command
    assert command[command.index("--post-cue-window") + 1] == "20.0"


def test_build_psth_command_includes_gui_choices():
    command = build_psth_command(
        ".",
        "sessions.csv",
        "data",
        "output",
        event_key="cue_onset",
        signal="licking",
        channel="2",
        window=(-5, 20),
        stratify=False,
        null_method="random_onsets",
        n_shuffles=25,
        seed=7,
        null_exclusion=3.5,
        trial_class="cue_miss",
        post_cue_window=4,
        heatmaps=True,
        heatmap_sort="ensure_latency",
        heatmap_sort_window=(0, 12),
        heatmap_sort_direction="ascending",
        heatmap_unmatched="exclude",
        heatmap_cmap="viridis",
    )
    assert command[command.index("--event-key") + 1] == "cue_onset"
    assert command[command.index("--signal") + 1] == "licking"
    assert command[command.index("--channel") + 1] == "2"
    assert "--no-stratify" in command
    assert command[command.index("--n-shuffles") + 1] == "25"
    assert command[command.index("--null-exclusion") + 1] == "3.5"
    assert command[command.index("--trial-class") + 1] == "cue_miss"
    assert command[command.index("--post-cue-window") + 1] == "4.0"
    assert "--heatmaps" in command
    assert command[command.index("--heatmap-sort") + 1] == "ensure_latency"
    assert command[command.index("--heatmap-sort-window") + 1 :][:2] == ["0.0", "12.0"]
    assert command[command.index("--heatmap-sort-direction") + 1] == "ascending"
    assert command[command.index("--heatmap-unmatched") + 1] == "exclude"
    assert command[command.index("--heatmap-cmap") + 1] == "viridis"


def test_build_behavior_glm_command_uses_zero_horizon_nested_model():
    command = build_behavior_glm_command(
        ".",
        "sessions.csv",
        "data",
        "output",
        channel="2",
        photometry_source="dff",
        history=8,
        lag_step=1,
        folds=4,
        inner_folds=2,
    )
    assert command[:3] == [
        command[0],
        "-m",
        "lutaslab_photometry.cli.run_forecasting",
    ]
    assert command[command.index("--target") + 1] == "photometry"
    assert command[command.index("--horizons") + 1] == "0.0"
    assert command[command.index("--history") + 1] == "8.0"
    assert command[command.index("--photometry-source") + 1] == "dff"
    assert command[command.index("--inner-folds") + 1] == "2"


def test_build_lifetime_command_includes_sensor_specific_options():
    command = build_lifetime_command(
        ".",
        "psth",
        "iflip3",
        "sessions.csv",
        "Z:/",
        "analysis/iflip3/psth",
        signal="mpet",
        event="ensure",
        heatmap_sort="first_lick_latency",
        heatmap_sort_window=(0, 12),
    )
    assert command[:3] == [
        command[0],
        "-m",
        "lutaslab_photometry.cli.run_lifetime_workflow",
    ]
    assert command[3] == "psth"
    assert command[command.index("--workflow") + 1] == "iflip3"
    assert command[command.index("--signal") + 1] == "mpet"
    assert command[command.index("--event") + 1] == "ensure"
    assert command[command.index("--heatmap-sort-window") + 1 :][:2] == [
        "0.0",
        "12.0",
    ]


def test_run_command_streams_and_captures_output(tmp_path):
    streamed = []
    return_code, output = run_command(
        [sys.executable, "-c", "print('first'); print('second')"],
        tmp_path,
        streamed.append,
    )
    assert return_code == 0
    assert output.splitlines() == ["first", "second"]
    assert "".join(streamed) == output
