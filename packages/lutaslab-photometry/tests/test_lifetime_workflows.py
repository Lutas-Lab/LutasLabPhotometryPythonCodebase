import csv
import types

import numpy as np
import pytest
from lutaslab_core.session import AlignedSession, ContinuousSignal, EventSeries

import lutaslab_photometry.lifetime_workflows as workflows
from lutaslab_photometry.lifetime_workflows import (
    discover_lifetime_paths,
    lifetime_manifest_columns,
    lifetime_processed_path,
    load_processed_lifetime_session,
    normalize_lifetime_manifest_rows,
    preprocess_lifetime_sessions,
    preview_iflip3_fit,
    write_lifetime_manifest,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


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


def test_iflip3_manifest_allows_missing_background():
    rows = normalize_lifetime_manifest_rows(
        "iflip3",
        [{"mouse": "AL164", "date": "260923", "run": 4}],
    )
    assert rows[0]["background_path"] == ""


def test_iflip3_fit_preview_returns_quality_and_time_resolved_components(monkeypatch):
    from iflip3.models import periodic_exgaussian_basis

    lifetime_time = np.arange(126, dtype=float) * 0.1
    short = periodic_exgaussian_basis(lifetime_time, 0.7, 1.0, 0.14, 12.5)
    long = periodic_exgaussian_basis(lifetime_time, 2.6, 1.0, 0.14, 12.5)
    scales = np.linspace(0.8, 1.2, 12)
    curves = (
        short[:, None] * (2500.0 * scales)[None, :]
        + long[:, None] * (5000.0 / scales)[None, :]
        + 2.0
    )
    header = types.SimpleNamespace(
        pulse_interval_ns=12.5,
        get_path=lambda path: 1.0,
    )
    recording = types.SimpleNamespace(
        lifetime_time=lifetime_time,
        header=header,
    )
    monkeypatch.setattr(
        workflows,
        "_iflip3_paths",
        lambda row, root: {"source_path": "recording", "background_path": None},
    )
    monkeypatch.setattr("iflip3.read_iflip3", lambda path: recording)
    monkeypatch.setattr(
        "iflip3.calculate_mpet",
        lambda *args, **kwargs: (None, types.SimpleNamespace(corrected=curves)),
    )
    settings = {
        "t0": {"mode": "fixed", "value": 1.0},
        "sigma": {"mode": "fixed", "value": 0.14},
        "tau1": {"mode": "bounded", "value": 0.6, "lower": 0.4, "upper": 1.0},
        "tau2": {"mode": "bounded", "value": 2.4, "lower": 1.5, "upper": 3.5},
    }

    preview = preview_iflip3_fit({}, ".", settings)

    assert preview.success
    np.testing.assert_allclose(preview.lifetimes, [0.7, 2.6], atol=0.04)
    assert preview.r_squared > 0.999
    assert preview.long_lifetime_fraction.shape == (12,)
    assert preview.fit_rmse_by_sample.shape == (12,)


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


def test_discover_fluopulse_paths_accepts_descriptive_suffixes(tmp_path):
    doric = _touch(
        tmp_path
        / "FLIM FLIP"
        / "SC81"
        / "SC81_260923"
        / "260923_SC81_run1_0000_EnsureTask.doric"
    )
    nidaq = _touch(
        tmp_path
        / "Photometry"
        / "SC81"
        / "SC81_260923"
        / "SC81-260923-001-nidaq.mat"
    )
    choices = discover_lifetime_paths("fluopulse", "SC81", "260923", 1, tmp_path)
    assert choices["doric_path"] == [doric]
    assert choices["nidaq_path"] == [nidaq]


def test_discover_fluopulse_paths_falls_back_to_session_folder_choices(tmp_path):
    folder = tmp_path / "FLIM FLIP" / "SC81" / "SC81_260923"
    descriptive = _touch(folder / "SC81_260923_15secFoodReward_40trials.doric")
    choices = discover_lifetime_paths("fluopulse", "SC81", "260923", 2, tmp_path)
    assert choices["doric_path"] == [descriptive]


def test_discover_iflip_paths_returns_recording_and_background_choices(tmp_path):
    folder = tmp_path / "FLIM FLIP" / "AL164" / "AL164_260923"
    recording = _touch(folder / "AL164_260923_004_EnsureTask.iFLiP3")
    background = _touch(folder / "AL164_260923_BG_high-power.iFLiP3")
    other_run = _touch(folder / "AL164_260923_005_EnsureTask.iFLiP3")
    choices = discover_lifetime_paths("iflip3", "AL164", "260923", 4, tmp_path)
    assert choices["iflip_path"] == [recording]
    assert choices["background_path"] == [background]
    assert other_run not in choices["iflip_path"]


@pytest.mark.parametrize(
    ("workflow", "recording_field", "recording_name", "background"),
    [
        ("fluopulse", "doric_path", "recording.doric", ""),
        ("iflip3", "iflip_path", "recording.iFLiP3", ""),
    ],
)
def test_lifetime_paths_leave_missing_nidaq_optional(
    workflow,
    recording_field,
    recording_name,
    background,
    tmp_path,
):
    recording = _touch(tmp_path / recording_name)
    row = {
        "mouse": "M1",
        "date": "260101",
        "run": 1,
        recording_field: str(recording),
        "nidaq_path": "",
        "running_path": "",
        "background_path": background,
    }
    if workflow == "iflip3":
        resolved = workflows._iflip3_paths(row, tmp_path)
        assert resolved["background_path"] is None
    else:
        resolved = workflows._fluopulse_paths(row, tmp_path)

    assert resolved["nidaq_path"] is None
    assert resolved["running_path"] is None


def test_lifetime_preprocessing_saves_beside_raw_and_roundtrips(monkeypatch, tmp_path):
    raw = _touch(tmp_path / "FLIM FLIP" / "M1" / "M1_260101" / "recording.doric")
    nidaq = _touch(tmp_path / "Photometry" / "M1" / "M1_260101" / "nidaq.mat")
    row = {
        "mouse": "M1",
        "date": "260101",
        "run": 1,
        "doric_path": str(raw),
        "nidaq_path": str(nidaq),
        "running_path": "",
    }
    time = np.arange(0.0, 3.0, 0.1)
    session = AlignedSession(
        session_id="M1_260101_run001",
        continuous={"tau": ContinuousSignal(time, np.sin(time), "ns")},
        events={
            "licks": EventSeries(np.array([1.0, 1.2])),
            "ensure": EventSeries(np.array([1.5])),
            "visual_cue": EventSeries(np.array([0.5])),
        },
        metadata={"clock_drift_ppm": np.float64(2.5)},
    )
    monkeypatch.setattr(
        workflows,
        "load_lifetime_session",
        lambda workflow, manifest_row, data_root: (
            session,
            {"source_path": str(raw), "nidaq_path": str(nidaq), "running_path": ""},
        ),
    )

    outputs = preprocess_lifetime_sessions("fluopulse", [row], tmp_path)
    expected = raw.with_name("recording-processed.npz")
    assert outputs == [expected]
    assert lifetime_processed_path("fluopulse", row, tmp_path) == expected

    loaded, paths = load_processed_lifetime_session("fluopulse", row, tmp_path)
    assert loaded.session_id == session.session_id
    assert loaded.continuous["tau"].units == "ns"
    np.testing.assert_allclose(loaded.continuous["tau"].values, np.sin(time))
    np.testing.assert_allclose(loaded.events["ensure"].timestamps, [1.5])
    assert loaded.metadata["clock_drift_ppm"] == 2.5
    assert paths["processed_path"] == str(expected)

    monkeypatch.setattr(
        workflows,
        "load_lifetime_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw reload")),
    )
    assert preprocess_lifetime_sessions("fluopulse", [row], tmp_path) == [expected]
    psth_outputs = workflows.run_lifetime_psth(
        "fluopulse",
        [row],
        tmp_path,
        tmp_path / "psth",
        signal="tau",
        event="ensure",
        window=(-0.2, 0.2),
        dt=0.1,
        normalization="none",
        heatmaps=False,
    )
    assert any(path.name == "fluopulse_tau_ensure_psth.png" for path in psth_outputs)


def test_processed_lifetime_file_is_required_for_downstream_analysis(tmp_path):
    raw = _touch(tmp_path / "FLIM FLIP" / "M1" / "M1_260101" / "recording.doric")
    row = {
        "mouse": "M1",
        "date": "260101",
        "run": 1,
        "doric_path": str(raw),
        "nidaq_path": "nidaq.mat",
        "running_path": "",
    }
    with pytest.raises(FileNotFoundError, match="Run lifetime preprocessing"):
        load_processed_lifetime_session("fluopulse", row, tmp_path)


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

    monkeypatch.setattr(workflows, "load_processed_lifetime_session", fake_load)
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


def test_lifetime_bout_offset_first_event_keeps_partial_tail(monkeypatch, tmp_path):
    time = np.arange(0.0, 12.1, 0.1)

    def fake_load(workflow, row, data_root):
        del workflow, data_root
        session = AlignedSession(
            session_id="M1_1",
            continuous={"tau": ContinuousSignal(time, time.copy(), "ns")},
            events={
                "ensure": EventSeries(np.array([2.0, 8.0])),
                "visual_cue": EventSeries(np.array([])),
                "licks": EventSeries(
                    np.array([2.0, 2.2, 2.4, 2.6, 2.8, 8.0, 8.2, 8.4, 8.6, 8.8])
                ),
            },
        )
        return session, {"source_path": "synthetic.doric"}

    monkeypatch.setattr(workflows, "load_processed_lifetime_session", fake_load)
    outputs = workflows.run_lifetime_psth(
        "fluopulse",
        [{"mouse": "M1", "date": "260101", "run": 1}],
        tmp_path,
        tmp_path / "partial",
        signal="tau",
        event="lick_bout_offset",
        window=(-1.0, 12.0),
        dt=0.1,
        baseline=(-1.0, 0.0),
        first_event_only=True,
        allow_partial_windows=True,
        heatmaps=True,
    )
    metadata = next(path for path in outputs if path.name == "analysis_metadata.json")
    text = metadata.read_text(encoding="utf-8")
    assert '"first_event_only": true' in text
    assert '"allow_partial_windows": true' in text
