import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from lutaslab_photometry.gui_workflows import (
    build_behavior_glm_command,
    build_lifetime_command,
    build_preprocess_command,
    build_psth_command,
    build_saved_psth_statistics_command,
    manifest_csv_text,
    next_analysis_run_directory,
    normalize_manifest_rows,
    run_command,
    safe_analysis_run_label,
    suggest_psth_run_label,
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


def test_psth_run_label_is_readable_and_windows_safe():
    assert (
        suggest_psth_run_label(
            "long_lifetime_fraction",
            "ensure",
            (-5, 20),
            condition_filter="BLA input",
        )
        == "BLA_input__long_lifetime_fraction__ensure__m5to20s"
    )
    assert safe_analysis_run_label(" custom: label? ") == "custom_label"


def test_next_analysis_run_directory_avoids_existing_run(tmp_path):
    now = datetime(2026, 10, 26, 14, 32, 10)
    first = next_analysis_run_directory(tmp_path, "BLA", now=now)
    first.mkdir()

    second = next_analysis_run_directory(tmp_path, "BLA", now=now)

    assert first.name == "2026-10-26_143210__BLA"
    assert second.name == "2026-10-26_143210__BLA__r02"


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
        photometry_signal="fitted405",
        channel="2",
        window=(-5, 20),
        y_limits=(-2, 3),
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
        first_event_only=True,
        allow_partial_windows=True,
        group_filter="Control",
        condition_filter="Ensure",
    )
    assert command[command.index("--event-key") + 1] == "cue_onset"
    assert command[command.index("--signal") + 1] == "licking"
    assert command[command.index("--photometry-signal") + 1] == "fitted405"
    assert command[command.index("--channel") + 1] == "2"
    assert "--no-stratify" in command
    assert command[command.index("--n-shuffles") + 1] == "25"
    assert command[command.index("--null-exclusion") + 1] == "3.5"
    assert command[command.index("--trial-class") + 1] == "cue_miss"
    assert command[command.index("--post-cue-window") + 1] == "4.0"
    assert command[command.index("--y-limits") + 1 :][:2] == ["-2.0", "3.0"]
    assert "--heatmaps" in command
    assert command[command.index("--heatmap-sort") + 1] == "ensure_latency"
    assert command[command.index("--heatmap-sort-window") + 1 :][:2] == ["0.0", "12.0"]
    assert command[command.index("--heatmap-sort-direction") + 1] == "ascending"
    assert command[command.index("--heatmap-unmatched") + 1] == "exclude"
    assert command[command.index("--heatmap-cmap") + 1] == "viridis"
    assert "--first-event-only" in command
    assert "--allow-partial-windows" in command
    assert command[command.index("--group") + 1] == "Control"
    assert command[command.index("--condition") + 1] == "Ensure"


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
        group_filter="Control",
        condition_filter="Ensure",
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
    assert command[command.index("--group") + 1] == "Control"
    assert command[command.index("--condition") + 1] == "Ensure"


def test_build_saved_psth_statistics_command_includes_metric_controls():
    command = build_saved_psth_statistics_command(
        ".",
        "analysis/psth/run",
        "analysis/psth/run/summary",
        response_window=(0.5, 3.0),
        peak_smoothing=0.2,
        metrics=("mean", "positive_auc", "trough"),
        test="wilcoxon",
        bootstrap_resamples=1000,
        bootstrap_seed=12,
    )

    assert command[2] == "lutaslab_photometry.cli.run_saved_psth_statistics"
    assert command[command.index("--response-window") + 1 :][:2] == ["0.5", "3.0"]
    assert command[command.index("--metrics") + 1 : command.index("--test")] == [
        "mean",
        "positive_auc",
        "trough",
    ]
    assert command[command.index("--test") + 1] == "wilcoxon"
    assert command[command.index("--bootstrap-resamples") + 1] == "1000"
    assert command[command.index("--bootstrap-seed") + 1] == "12"


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
        y_limits=(-50, 150),
        heatmap_sort="first_lick_latency",
        heatmap_sort_window=(0, 12),
        first_event_only=True,
        allow_partial_windows=True,
        group_filter="Experimental",
        condition_filter="PostEnsure",
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
    assert command[command.index("--y-limits") + 1 :][:2] == ["-50.0", "150.0"]
    assert command[command.index("--heatmap-sort-window") + 1 :][:2] == [
        "0.0",
        "12.0",
    ]
    assert "--first-event-only" in command
    assert "--allow-partial-windows" in command
    assert command[command.index("--group") + 1] == "Experimental"
    assert command[command.index("--condition") + 1] == "PostEnsure"


def test_build_lifetime_preprocess_command_writes_beside_raw_data():
    command = build_lifetime_command(
        ".",
        "preprocess",
        "fluopulse",
        "sessions.csv",
        "Z:/",
        overwrite=True,
    )
    assert "--output-dir" not in command
    assert "--overwrite" in command


def test_build_iflip3_preprocess_command_includes_fit_controls():
    settings = {"tau1": {"mode": "fixed", "value": 0.7, "lower": 0.03, "upper": 5.0}}
    command = build_lifetime_command(
        ".",
        "preprocess",
        "iflip3",
        "sessions.csv",
        "Z:/",
        iflip3_fit_settings=settings,
    )
    encoded = command[command.index("--iflip3-fit-settings") + 1]
    assert json.loads(encoded) == settings


def test_build_iflip3_preprocess_command_accepts_saved_calibration():
    command = build_lifetime_command(
        ".",
        "preprocess",
        "iflip3",
        "sessions.csv",
        "Z:/",
        iflip3_calibration="analysis/iflip3/calibrations/shared.json",
    )
    assert command[command.index("--iflip3-calibration") + 1] == str(
        Path("analysis/iflip3/calibrations/shared.json")
    )


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
