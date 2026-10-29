import csv
import json

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from lutaslab_core.session import (
    AlignedSession,
    ContinuousSignal,
    EventSeries,
    IntervalSeries,
)

from lutaslab_photometry.session_trace import (
    SessionTraceData,
    load_conventional_session_trace,
    load_lifetime_session_trace,
    plot_session_trace,
    save_session_trace,
    transform_trace,
)


def _trace_data():
    time = np.linspace(0.0, 20.0, 201)
    return SessionTraceData(
        workflow="iflip3",
        session_id="M1_260101_run001",
        signal="mpet",
        units="ps",
        time=time,
        values=np.sin(time),
        cue_starts=np.array([2.0]),
        cue_ends=np.array([4.0]),
        ensure_starts=np.array([5.0, 10.0]),
        ensure_ends=np.array([5.5, 12.0]),
        lick_times=np.array([5.2, 5.4, 11.0]),
        source_path="processed.npz",
    )


def test_plot_uses_measured_cue_and_ensure_durations():
    figure, _, _ = plot_session_trace(
        _trace_data(), overview_window=(0.0, 20.0), zoom_window=(4.0, 13.0)
    )

    # One Cue and two Ensure duration patches appear in each panel.
    assert len(figure.axes[0].patches) == 3
    widths = []
    for patch in figure.axes[0].patches:
        display_vertices = patch.get_transform().transform(patch.get_path().vertices)
        data_vertices = figure.axes[0].transData.inverted().transform(display_vertices)
        widths.append(data_vertices[:, 0].max() - data_vertices[:, 0].min())
    widths.sort()
    np.testing.assert_allclose(widths, [0.5, 2.0, 2.0])
    plt.close(figure)


def test_plot_falls_back_to_onset_line_when_offset_is_missing():
    data = _trace_data()
    data = SessionTraceData(
        **{
            **data.__dict__,
            "ensure_starts": np.array([5.0]),
            "ensure_ends": np.array([np.nan]),
        }
    )

    figure, _, _ = plot_session_trace(
        data, overview_window=(0.0, 20.0), zoom_window=(4.0, 8.0)
    )

    assert any(
        len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [5.0, 5.0])
        for line in figure.axes[0].lines
    )
    plt.close(figure)


def test_full_session_transform_does_not_depend_on_display_window():
    values = np.array([1.0, 2.0, 3.0])

    transformed, parameters = transform_trace(values, "zscore")

    np.testing.assert_allclose(transformed, [-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    assert parameters[0] == 2.0


def test_save_session_trace_writes_figure_values_events_and_metadata(tmp_path):
    outputs = save_session_trace(
        _trace_data(),
        tmp_path,
        overview_window=(0.0, 20.0),
        zoom_window=(4.0, 13.0),
        transform="subtract_mean",
        formats=("png",),
    )

    assert {path.name for path in outputs} == {
        "session_trace.png",
        "session_trace_values.csv",
        "session_trace_events.csv",
        "session_trace_metadata.json",
    }
    with (tmp_path / "session_trace_events.csv").open(encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    assert [row["event"] for row in events].count("ensure") == 2
    metadata = json.loads((tmp_path / "session_trace_metadata.json").read_text())
    assert metadata["zoom_window"] == [4.0, 13.0]
    assert metadata["transform"] == "subtract_mean"


def test_conventional_loader_uses_recorded_event_offsets(monkeypatch, tmp_path):
    session = {
        "photo_time_465_ch2": np.array([0.0, 1.0, 2.0]),
        "dff_ch2": np.array([0.0, 1.0, 0.0]),
        "cue_onset": np.array([0.25]),
        "cue_offset": np.array([0.75]),
        "solenoid_onset": np.array([1.0]),
        "solenoid_offset": np.array([1.5]),
        "lick_times": np.array([1.25]),
    }
    monkeypatch.setattr(
        "lutaslab_photometry.session_trace.load_session", lambda path: session
    )
    row = {"mouse": "M1", "date": "260101", "run": 1, "channel": 2}

    data = load_conventional_session_trace(row, tmp_path, signal="dff")

    np.testing.assert_allclose(data.ensure_ends - data.ensure_starts, [0.5])
    assert data.session_id.endswith("ch2")


def test_lifetime_loader_uses_saved_two_second_ensure_interval(monkeypatch):
    session = AlignedSession(
        session_id="M1_260101_run001",
        continuous={"mpet": ContinuousSignal([0.0, 1.0, 2.0], [1.0, 2.0, 1.0], "ps")},
        events={
            "licks": EventSeries([1.5], "licks"),
            "ensure": EventSeries([0.0], "ensure"),
        },
        intervals={"ensure": IntervalSeries([0.0], [2.0], "ensure")},
        metadata={},
    )
    monkeypatch.setattr(
        "lutaslab_photometry.session_trace.load_processed_lifetime_session",
        lambda workflow, row, root: (session, {"processed_path": "processed.npz"}),
    )

    data = load_lifetime_session_trace(
        "iflip3", {"mouse": "M1", "date": "260101", "run": 1}, ".", "mpet"
    )

    np.testing.assert_allclose(data.ensure_ends - data.ensure_starts, [2.0])
