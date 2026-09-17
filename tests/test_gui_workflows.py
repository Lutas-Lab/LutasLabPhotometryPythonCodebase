import csv
import io
from pathlib import Path

import pytest

from src.gui_workflows import (
    build_preprocess_command,
    build_psth_command,
    manifest_csv_text,
    normalize_manifest_rows,
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


def test_build_preprocess_command_uses_current_python_and_safe_arguments():
    command = build_preprocess_command(
        Path("project"),
        Path("analysis/sessions.csv"),
        Path("Z:/Photometry Data"),
        overwrite=True,
        post_cue_window=20,
    )
    assert command[1].endswith("scripts\\run_preprocess_batch.py") or command[1].endswith(
        "scripts/run_preprocess_batch.py"
    )
    assert command[command.index("--data-root") + 1] == str(Path("Z:/Photometry Data"))
    assert "--overwrite" in command
    assert command[command.index("--post-cue-window") + 1] == "20.0"


def test_build_psth_command_includes_gui_choices():
    command = build_psth_command(
        ".",
        "sessions.csv",
        "data",
        "output",
        event_key="lick_bout_onset",
        signal="licking",
        channel="2",
        window=(-5, 20),
        stratify=False,
        null_method="random_onsets",
        n_shuffles=25,
        seed=7,
    )
    assert command[command.index("--event-key") + 1] == "lick_bout_onset"
    assert command[command.index("--signal") + 1] == "licking"
    assert command[command.index("--channel") + 1] == "2"
    assert "--no-stratify" in command
    assert command[command.index("--n-shuffles") + 1] == "25"
