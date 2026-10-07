import csv

import numpy as np
import pytest
from lutaslab_core.session import AlignedSession, ContinuousSignal, EventSeries

import lutaslab_photometry.lifetime_workflows as workflows
from lutaslab_photometry.lifetime_workflows import (
    lifetime_manifest_columns,
    normalize_lifetime_manifest_rows,
    write_lifetime_manifest,
)


def test_fluopulse_manifest_allows_convention_based_path_discovery():
    rows = normalize_lifetime_manifest_rows(
        "fluopulse",
        [{"mouse": "SC81", "date": "260923", "run": "1"}],
    )
    assert rows[0]["run"] == 1
    assert rows[0]["doric_path"] == ""
    assert lifetime_manifest_columns("fluopulse")[-3:] == (
        "doric_path",
        "nidaq_path",
        "running_path",
    )


def test_iflip3_manifest_requires_explicit_matched_background():
    with pytest.raises(ValueError, match="matched background_path"):
        normalize_lifetime_manifest_rows(
            "iflip3",
            [{"mouse": "AL164", "date": "260923", "run": 4}],
        )


def test_write_iflip3_manifest_preserves_background_provenance(tmp_path):
    destination = tmp_path / "iflip3" / "sessions.csv"
    write_lifetime_manifest(
        "iflip3",
        destination,
        [
            {
                "mouse": "AL164",
                "date": "260923",
                "run": 4,
                "background_path": "backgrounds/high_power.iFLiP3",
            }
        ],
    )
    with destination.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["background_path"] == "backgrounds/high_power.iFLiP3"


def test_lifetime_psth_writes_mouse_and_pooled_heatmaps(monkeypatch, tmp_path):
    time = np.arange(0.0, 40.0, 0.1)

    def fake_load(workflow, row, data_root):
        del workflow, data_root
        session = AlignedSession(
            session_id=f"{row['mouse']}_{row['run']}",
            continuous={
                "tau": ContinuousSignal(time, np.sin(time / 4), "ns")
            },
            events={
                "ensure": EventSeries(np.array([10.0, 20.0, 30.0])),
                "visual_cue": EventSeries(np.array([9.0, 19.0, 29.0])),
                "licks": EventSeries(np.array([10.5, 20.5, 30.5])),
            },
        )
        return session, {"source_path": "synthetic.doric"}

    monkeypatch.setattr(workflows, "load_lifetime_session", fake_load)
    outputs = workflows.run_lifetime_psth(
        "fluopulse",
        [
            {
                "mouse": "M1",
                "date": "260101",
                "run": 1,
                "group": "control",
                "condition": "test",
            },
            {
                "mouse": "M2",
                "date": "260101",
                "run": 1,
                "group": "control",
                "condition": "test",
            },
        ],
        tmp_path,
        tmp_path / "output",
        signal="tau",
        event="ensure",
        window=(-2, 3),
        dt=0.1,
        baseline=(-2, 0),
        heatmap_sort="first_lick_latency",
        heatmap_sort_window=(0, 2),
    )
    names = {path.name for path in outputs}
    assert "group_ensure_mouse_means_heatmap.png" in names
    assert "group_ensure_pooled_trials_heatmap.png" in names
    assert "heatmap_trial_order.csv" in names
