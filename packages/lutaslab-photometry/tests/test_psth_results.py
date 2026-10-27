import csv
import json

import numpy as np
import pytest

from lutaslab_photometry.psth_results import (
    load_saved_psth_timecourses,
    save_psth_timecourse_tables,
    save_saved_psth_statistics,
)


def _result(condition, offset=0.0, *, signal="mpet"):
    time = np.array([-1.0, 0.0, 1.0, 2.0])
    sessions = []
    for index, mouse in enumerate(("M1", "M2"), start=1):
        trace = offset * index + index + np.array([0.0, 1.0, 2.0, 1.0])
        sessions.append(
            {
                "mouse": mouse,
                "date": f"26010{index}",
                "run": 1,
                "group": "G",
                "condition": condition,
                "session_id": f"{mouse}_{condition}",
                "trials": np.vstack((trace, trace + 0.2)),
            }
        )
    return {
        "time": time,
        "workflow": "iflip3",
        "signal_key": signal,
        "signal_type": "photometry",
        "event_key": "ensure",
        "normalization": "subtract",
        "units": "ps",
        "session_results": sessions,
    }


def test_timecourse_exports_roundtrip_and_create_prism_table(tmp_path):
    outputs = save_psth_timecourse_tables(_result("A"), tmp_path)

    assert {path.name for path in outputs} >= {
        "psth_session_timecourses.csv",
        "psth_mouse_timecourses.csv",
        "psth_group_timecourse.csv",
        "psth_prism_timecourses.csv",
        "psth_timecourses.npz",
        "psth_timecourse_metadata.json",
    }
    loaded = load_saved_psth_timecourses(tmp_path)
    assert loaded["identity"]["signal"] == "mpet"
    assert len(loaded["session_records"]) == 2
    assert len(loaded["mouse_records"]) == 2
    np.testing.assert_allclose(loaded["time"], [-1.0, 0.0, 1.0, 2.0])
    with (tmp_path / "psth_prism_timecourses.csv").open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert header == ["time_s", "G__A__M1", "G__A__M2"]


def test_timecourse_loader_preserves_blank_partial_window_samples_as_nan(tmp_path):
    result = _result("A")
    result["session_results"][0]["trials"][:, -1] = np.nan
    save_psth_timecourse_tables(result, tmp_path)

    loaded = load_saved_psth_timecourses(tmp_path)

    first_session = next(
        trace
        for record, trace in zip(
            loaded["session_records"], loaded["session_matrix"], strict=True
        )
        if record["session_id"] == "M1_A"
    )
    assert np.isnan(first_session[-1])
    np.testing.assert_allclose(first_session[:-1], [1.1, 2.1, 3.1])


def test_mouse_timecourse_weights_sessions_equally_not_trials(tmp_path):
    result = _result("A")
    first = result["session_results"][0]
    first["trials"] = np.zeros((100, 4))
    second = {**first, "date": "260110", "session_id": "M1_second"}
    second["trials"] = np.full((1, 4), 10.0)
    result["session_results"] = [first, second]

    save_psth_timecourse_tables(result, tmp_path)
    loaded = load_saved_psth_timecourses(tmp_path)

    assert len(loaded["mouse_records"]) == 1
    np.testing.assert_allclose(loaded["mouse_matrix"][0], 5.0)


def test_saved_statistics_combine_conditions_and_write_metric_prism_files(tmp_path):
    source = tmp_path / "source"
    save_psth_timecourse_tables(_result("A"), source / "A")
    save_psth_timecourse_tables(_result("B", offset=2.0), source / "B")
    progress = []

    outputs = save_saved_psth_statistics(
        source,
        tmp_path / "summary",
        response_window=(0.0, 2.0),
        peak_smoothing=0.0,
        metrics=("mean", "auc", "peak", "trough"),
        bootstrap_resamples=100,
        formats=("png",),
        progress_callback=lambda current, total, label: progress.append(
            (current, total, label)
        ),
    )

    output_names = {path.name for path in outputs}
    assert {
        "psth_session_metrics.csv",
        "psth_mouse_metrics.csv",
        "psth_group_summary.csv",
        "psth_statistical_tests.csv",
        "psth_prism_wide.csv",
        "prism_mean.csv",
        "prism_auc.csv",
        "prism_peak.csv",
        "prism_trough.csv",
        "psth_statistics_metadata.json",
    }.issubset(output_names)
    with (tmp_path / "summary" / "prism_mean.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ["mouse", "G__A", "G__B"]
    assert [row["mouse"] for row in rows] == ["M1", "M2"]
    with (tmp_path / "summary" / "psth_mouse_metrics.csv").open(
        encoding="utf-8"
    ) as stream:
        metric_rows = list(csv.DictReader(stream))
    m1_a_mean = next(
        row
        for row in metric_rows
        if row["mouse"] == "M1" and row["condition"] == "A" and row["metric"] == "mean"
    )
    assert float(m1_a_mean["value"]) == pytest.approx((2.1 + 3.1 + 2.1) / 3)
    metadata = json.loads(
        (tmp_path / "summary" / "psth_statistics_metadata.json").read_text()
    )
    assert metadata["hierarchy"].endswith("mice are the inferential N")
    assert progress[-1] == (4, 4, "PSTH summary statistics complete")


def test_run_root_uses_canonical_export_without_mixing_behavior_subfolders(tmp_path):
    save_psth_timecourse_tables(_result("A", signal="mpet"), tmp_path)
    save_psth_timecourse_tables(
        _result("A", signal="licking"), tmp_path / "licking"
    )

    loaded = load_saved_psth_timecourses(tmp_path)

    assert loaded["identity"]["signal"] == "mpet"
    assert loaded["source_files"] == [
        str(tmp_path / "psth_session_timecourses.csv")
    ]


def test_saved_statistics_reject_incompatible_signals(tmp_path):
    save_psth_timecourse_tables(_result("A", signal="mpet"), tmp_path / "mpet")
    save_psth_timecourse_tables(_result("A", signal="raw_intensity"), tmp_path / "raw")

    with pytest.raises(ValueError, match="incompatible PSTH"):
        load_saved_psth_timecourses(tmp_path)
