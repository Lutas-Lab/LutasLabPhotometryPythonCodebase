import csv
import json
import types
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from lutaslab_core.session import (
    AlignedSession,
    ContinuousSignal,
    EventSeries,
    IntervalSeries,
)

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


def test_next_iflip3_calibration_path_is_descriptive_and_unique(tmp_path):
    calibration = types.SimpleNamespace(
        pulse_interval=12.5,
        session_qc=tuple({"session_id": f"S{index}"} for index in range(3)),
    )
    now = datetime(2026, 10, 26, 19, 15, tzinfo=UTC)
    first = workflows.next_iflip3_calibration_path(
        tmp_path / "shared.json", calibration, now=now
    )
    first.write_text("{}", encoding="utf-8")

    second = workflows.next_iflip3_calibration_path(first, calibration, now=now)

    assert first.name == "iflip3_80mhz_3sessions__2026-10-26_191500Z.json"
    assert second.name == "iflip3_80mhz_3sessions__2026-10-26_191500Z__r02.json"


def test_iflip3_residual_summary_reports_scale_free_percentages():
    summary = workflows.iflip3_residual_summary(
        np.array([0.0, 10.0, 5.0]),
        np.array([0.0, 1.0, -1.0]),
    )

    assert summary["max_absolute_residual_percent_peak"] == pytest.approx(10.0)
    assert summary["rmse_percent_peak"] == pytest.approx(8.1649658)
    assert summary["absolute_residual_area_percent"] == pytest.approx(13.3333333)


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


def test_multi_session_iflip3_calibration_roundtrips_and_fixes_background(
    monkeypatch, tmp_path
):
    from iflip3 import periodic_exgaussian_basis

    time = np.arange(0.4, 12.31, 0.1)
    tau = (0.7, 2.6)

    def corrected(row, root, *, spc_range, afterpulse_ratio):
        del root, spc_range, afterpulse_ratio
        scale = float(row["run"])
        session_t0 = {1: 1.0, 2: 1.05, 3: 3.0, 4: 1.4}[int(row["run"])]
        short = periodic_exgaussian_basis(time, tau[0], session_t0, 0.14, 12.5)
        long = periodic_exgaussian_basis(time, tau[1], session_t0, 0.14, 12.5)
        aggregate = (1000.0 * scale) * short + (2500.0 / scale) * long
        curves = np.column_stack([aggregate * 0.45, aggregate * 0.55])
        return (
            time,
            curves,
            12.5,
            10.0,
            True,
            {
                "source_path": Path(f"recording-{row['run']}.iFLiP3"),
                "background_path": Path("background.iFLiP3"),
            },
        )

    monkeypatch.setattr(workflows, "_load_corrected_iflip3_curves", corrected)
    rows = [
        {"mouse": "M1", "date": "260101", "run": 1},
        {"mouse": "M2", "date": "260102", "run": 2},
        {"mouse": "M3", "date": "260103", "run": 3},
    ]
    calibration = workflows.calibrate_iflip3_sessions(rows, tmp_path)
    destination = workflows.save_iflip3_calibration(
        calibration, tmp_path / "shared.json"
    )
    loaded = workflows.load_iflip3_calibration(destination)

    assert loaded.success
    np.testing.assert_allclose(loaded.lifetimes, tau, atol=0.25)
    assert loaded.t0 == pytest.approx(1.05, abs=0.2)
    assert loaded.irf_sigma == pytest.approx(0.14, abs=0.12)
    assert loaded.to_dict()["model"]["residual_background_counts_per_bin"] == 0.0
    assert loaded.to_dict()["model"]["shared_parameters"] == [
        "tau1",
        "tau2",
        "irf_sigma",
    ]
    assert len(loaded.session_qc) == 3
    assert loaded.session_qc[0]["t0_ns"] == pytest.approx(1.0, abs=0.2)
    assert loaded.session_qc[1]["t0_ns"] == pytest.approx(1.05, abs=0.2)
    assert loaded.session_qc[2]["t0_ns"] == pytest.approx(3.0, abs=0.2)
    assert loaded.session_qc[0]["max_absolute_residual_percent_peak"] < 3.0
    assert [item["t0_offset_flag"] for item in loaded.session_qc] == [
        False,
        False,
        True,
    ]
    decomposition = workflows.apply_iflip3_calibration(rows[0], tmp_path, loaded)
    assert decomposition.background == 0.0
    assert decomposition.t0 == pytest.approx(loaded.session_qc[0]["t0_ns"])
    np.testing.assert_allclose(
        decomposition.fitted_decay,
        decomposition.short_component + decomposition.long_component,
    )
    np.testing.assert_allclose(
        decomposition.fitted_decay,
        decomposition.aggregate_decay,
        rtol=0.03,
    )
    assert decomposition.r_squared > 0.999
    np.testing.assert_allclose(decomposition.lifetimes, loaded.lifetimes)
    assert decomposition.short_component_counts.shape == (2,)
    assert decomposition.long_component_counts.shape == (2,)
    new_session = workflows.apply_iflip3_calibration(
        {"mouse": "M4", "date": "260104", "run": 4}, tmp_path, loaded
    )
    assert new_session.t0 == pytest.approx(1.4, abs=0.2)
    assert "estimated from this session" in new_session.message

    legacy_payload = loaded.to_dict()
    legacy_payload["calibration_schema_version"] = "1.0"
    legacy_payload["model"]["t0_ns"] = loaded.t0
    legacy_payload["model"].pop("reference_t0_ns")
    legacy_payload["model"].pop("session_specific_parameters")
    legacy_payload["model"]["shared_parameters"] = [
        "tau1",
        "tau2",
        "t0",
        "irf_sigma",
    ]
    for item in legacy_payload["fit"]["session_qc"]:
        item.pop("t0_ns")
        item.pop("t0_offset_from_median_ns")
        item.pop("t0_offset_flag")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    legacy = workflows.load_iflip3_calibration(legacy_path)
    legacy_preview = workflows.apply_iflip3_calibration(rows[0], tmp_path, legacy)
    assert legacy.schema_version == "1.0"
    assert legacy_preview.t0 == loaded.t0


def test_iflip3_pulse_interval_comparison_accepts_small_sync_rate_variation():
    assert workflows._pulse_intervals_compatible(12.5, 12.51)
    assert not workflows._pulse_intervals_compatible(12.5, 25.0)


def test_iflip3_calibration_reports_material_pulse_rate_difference(monkeypatch):
    time = np.arange(0.4, 12.31, 0.1)

    def corrected(row, root, *, spc_range, afterpulse_ratio):
        del root, spc_range, afterpulse_ratio
        interval = 12.5 if int(row["run"]) == 1 else 25.0
        return (
            time,
            np.ones((time.size, 2)),
            interval,
            10.0,
            True,
            {"source_path": Path("recording.iFLiP3"), "background_path": None},
        )

    monkeypatch.setattr(workflows, "_load_corrected_iflip3_curves", corrected)
    rows = [
        {"mouse": "M1", "date": "260101", "run": 1},
        {"mouse": "M1", "date": "260101", "run": 2},
    ]
    with pytest.raises(ValueError, match=r"25 ns \(40 MHz\).+12.5 ns \(80 MHz\)"):
        workflows.calibrate_iflip3_sessions(rows, ".")


def test_lifetime_processed_file_embeds_structured_provenance():
    path = Path("packages/lutaslab-photometry/tests/_lifetime-provenance.npz")
    session = AlignedSession(
        session_id="M1_260101_run001",
        continuous={
            "tau": ContinuousSignal(np.array([0.0, 1.0]), np.array([2.0, 2.1]), "ns")
        },
        events={"ensure": EventSeries(np.array([0.5]))},
        metadata={},
    )
    parameters = workflows.lifetime_preprocessing_parameters("fluopulse")
    try:
        workflows._save_processed_lifetime_session(
            "fluopulse",
            session,
            {"source_path": "recording.doric"},
            path,
            parameters,
        )
        with np.load(path, allow_pickle=False) as data:
            provenance = json.loads(str(data["provenance_json"].item()))
    finally:
        path.unlink(missing_ok=True)

    assert provenance["workflow"] == "fluopulse"
    assert provenance["parameters"] == parameters
    assert "commit" in provenance["code"]
    assert "dirty" in provenance["code"]


def test_processed_file_replacement_retries_transient_lock(monkeypatch, tmp_path):
    source = tmp_path / "temporary.npz"
    destination = tmp_path / "session-processed.npz"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    real_replace = workflows.os.replace
    attempts = []
    delays = []

    def transient_lock(current_source, current_destination):
        attempts.append((current_source, current_destination))
        if len(attempts) < 3:
            raise PermissionError(13, "file is in use", str(current_destination))
        real_replace(current_source, current_destination)

    monkeypatch.setattr(workflows.os, "replace", transient_lock)
    monkeypatch.setattr(workflows.time, "sleep", delays.append)

    workflows._replace_with_retry(
        source,
        destination,
        attempts=3,
        initial_delay=0.25,
    )

    assert destination.read_bytes() == b"new"
    assert len(attempts) == 3
    assert delays == [0.25, 0.5]


def test_lifetime_plot_values_use_picoseconds_and_clear_labels():
    signal = ContinuousSignal(
        np.array([0.0, 1.0]),
        np.array([2.4, 2.5]),
        "ns",
    )

    values, units = workflows._lifetime_plot_values("tau", signal)

    np.testing.assert_allclose(values, [2400.0, 2500.0])
    assert units == "ps"
    assert workflows._lifetime_psth_ylabel("tau", "none", units) == "τ (ps)"
    assert workflows._lifetime_psth_ylabel("tau", "subtract", units) == "Δτ (ps)"
    assert (
        workflows._lifetime_psth_ylabel("running_speed", "none", "cm/s")
        == "Running speed (cm/s)"
    )


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
        sampling_frequency=10.0,
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


def test_iflip3_binning_sensitivity_reports_improving_poisson_precision():
    from iflip3.models import periodic_exgaussian_basis

    time = np.arange(0.4, 12.31, 0.1)
    short_shape = periodic_exgaussian_basis(
        time, 0.7, 1.0, 0.14, 12.5, normalize="area"
    )
    long_shape = periodic_exgaussian_basis(time, 2.6, 1.0, 0.14, 12.5, normalize="area")
    n_samples = 200
    short_counts = np.full(n_samples, 3000.0)
    long_counts = np.full(n_samples, 6000.0)
    preview = workflows.IFLIP3FitPreview(
        lifetime_time=time,
        aggregate_decay=n_samples * (3000.0 * short_shape + 6000.0 * long_shape),
        fitted_decay=n_samples * (3000.0 * short_shape + 6000.0 * long_shape),
        short_component=n_samples * 3000.0 * short_shape,
        long_component=n_samples * 6000.0 * long_shape,
        residuals=np.zeros(time.size),
        lifetimes=np.array([0.7, 2.6]),
        t0=1.0,
        irf_sigma=0.14,
        background=0.0,
        rmse=0.0,
        r_squared=1.0,
        success=True,
        message="synthetic preview",
        short_component_counts=short_counts,
        long_component_counts=long_counts,
        long_lifetime_fraction=np.full(n_samples, 2 / 3),
        fit_rmse_by_sample=np.zeros(n_samples),
        sampling_frequency=10.0,
    )

    sensitivity = workflows.estimate_iflip3_binning_sensitivity(
        preview, n_bootstrap=100, random_seed=7
    )

    np.testing.assert_allclose(
        sensitivity.bin_durations_seconds, [0.1, 0.2, 0.5, 1, 2, 5, 10]
    )
    np.testing.assert_allclose(
        sensitivity.median_sensor_photons,
        [9_000, 18_000, 45_000, 90_000, 180_000, 450_000, 900_000],
    )
    assert sensitivity.mpet_95_half_width_ps[-1] < sensitivity.mpet_95_half_width_ps[0]
    assert (
        sensitivity.long_fraction_95_half_width[-1]
        < sensitivity.long_fraction_95_half_width[0]
    )

    with pytest.raises(ValueError, match="cannot be shorter than the native"):
        workflows.estimate_iflip3_binning_sensitivity(
            preview,
            bin_durations_seconds=(0.05, 0.1),
            n_bootstrap=20,
        )


def test_temporal_binning_sums_corrected_iflip3_curves():
    curves = np.arange(24, dtype=float).reshape(3, 8)

    binned = workflows._temporally_bin_iflip3_curves(curves, 10.0, 0.2)

    np.testing.assert_allclose(
        binned,
        curves.reshape(3, 4, 2).sum(axis=2),
    )


def test_iflip3_afterpulse_only_fit_reuses_corrected_decay_shape():
    from iflip3.models import periodic_exgaussian_basis

    lifetime_time = np.arange(126, dtype=float) * 0.1
    short = periodic_exgaussian_basis(
        lifetime_time, 0.7, 1.0, 0.14, 12.5, normalize="area"
    )
    long = periodic_exgaussian_basis(
        lifetime_time, 2.6, 1.0, 0.14, 12.5, normalize="area"
    )
    curves = short[:, None] * np.array([[2000.0, 2200.0]]) + long[
        :, None
    ] * np.array([[4000.0, 3800.0]])
    reference = workflows.IFLIP3FitPreview(
        lifetime_time=lifetime_time,
        aggregate_decay=curves.sum(axis=1),
        fitted_decay=curves.sum(axis=1),
        short_component=short,
        long_component=long,
        residuals=np.zeros(lifetime_time.size),
        lifetimes=np.array([0.7, 2.6]),
        t0=1.0,
        irf_sigma=0.14,
        background=0.0,
        rmse=0.0,
        r_squared=1.0,
        success=True,
        message="reference",
        short_component_counts=np.array([2000.0, 2200.0]),
        long_component_counts=np.array([4000.0, 3800.0]),
        long_lifetime_fraction=np.array([2 / 3, 0.6333333333]),
        fit_rmse_by_sample=np.zeros(2),
        sampling_frequency=10.0,
    )

    alternate = workflows._iflip3_preview_with_fixed_shape(
        reference,
        lifetime_time,
        curves,
        12.5,
    )

    np.testing.assert_allclose(alternate.lifetimes, reference.lifetimes)
    assert alternate.t0 == reference.t0
    assert alternate.irf_sigma == reference.irf_sigma
    np.testing.assert_allclose(
        alternate.short_component_counts,
        reference.short_component_counts,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        alternate.long_component_counts,
        reference.long_component_counts,
        rtol=1e-5,
    )


def test_iflip3_comparison_signals_are_available_to_downstream_workflows():
    assert {
        "mpet_afterpulse_only",
        "mpet_uncorrected",
        "long_lifetime_fraction_afterpulse_only",
        "short_component_counts_afterpulse_only",
        "long_component_counts_afterpulse_only",
    }.issubset(workflows.LIFETIME_SIGNALS["iflip3"])
    assert "mpet_afterpulse_only" in workflows.LIFETIME_TIME_SIGNALS
    assert "mpet_uncorrected" in workflows.LIFETIME_TIME_SIGNALS


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
        tmp_path / "Photometry" / "SC81" / "SC81_260923" / "SC81-260923-001-nidaq.mat"
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
        intervals={
            "visual_cue": IntervalSeries(np.array([0.5]), np.array([0.8]))
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
    np.testing.assert_allclose(loaded.intervals["visual_cue"].ends, [0.8])
    assert loaded.metadata["clock_drift_ppm"] == 2.5
    assert loaded.metadata["provenance"]["workflow"] == "fluopulse"
    assert "commit" in loaded.metadata["provenance"]["code"]
    assert loaded.metadata["provenance"]["parameters"]["nidaq_threshold"] == 1.5
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
        y_limits=(-10.0, 20.0),
        heatmaps=False,
    )
    assert any(path.name == "fluopulse_tau_ensure_psth.png" for path in psth_outputs)
    assert {
        "psth_session_timecourses.csv",
        "psth_mouse_timecourses.csv",
        "psth_group_timecourse.csv",
        "psth_prism_timecourses.csv",
        "psth_timecourses.npz",
        "psth_timecourse_metadata.json",
    }.issubset({path.name for path in psth_outputs})
    psth_metadata = json.loads(
        (tmp_path / "psth" / "analysis_metadata.json").read_text()
    )
    assert psth_metadata["units"] == "ps"
    assert psth_metadata["ylabel"] == "τ (ps)"
    assert psth_metadata["y_limits"] == [-10.0, 20.0]
    assert psth_metadata["run_label"] == "psth"
    assert psth_metadata["heatmaps"]["enabled"] is False
    assert psth_metadata["sessions"][0]["session_id"] == session.session_id
    assert "generated_utc" in psth_metadata
    assert "code_commit" in psth_metadata


def test_iflip3_preprocessing_applies_saved_calibration(monkeypatch, tmp_path):
    raw = _touch(tmp_path / "FLIM FLIP" / "M1" / "M1_260101" / "recording.iFLiP3")
    row = {
        "mouse": "M1",
        "date": "260101",
        "run": 1,
        "iflip_path": str(raw),
        "nidaq_path": "",
        "running_path": "",
        "background_path": "",
    }
    calibration = workflows.IFLIP3Calibration(
        lifetimes=np.array([0.7, 2.6]),
        t0=1.0,
        irf_sigma=0.14,
        lifetime_time=np.arange(0.4, 12.31, 0.1),
        pulse_interval=12.5,
        spc_range=(0.4, 12.3),
        afterpulse_ratio=0.0,
        session_qc=({"session_id": "M1_260101_run001", "t0_ns": 1.0},),
        success=True,
        message="synthetic calibration",
        cost=0.0,
        nfev=1,
        created_utc="2026-10-27T12:00:00+00:00",
    )
    time = np.arange(0.0, 3.0, 0.1)
    fraction = np.linspace(0.2, 0.8, time.size)
    session = AlignedSession(
        session_id="M1_260101_run001",
        continuous={
            "mpet": ContinuousSignal(time, np.full(time.size, 2.0), "ns"),
            "long_lifetime_fraction": ContinuousSignal(time, fraction, "fraction"),
        },
        events={"ensure": EventSeries(np.array([1.5]))},
    )
    received = {}

    def fake_load(
        workflow,
        manifest_row,
        data_root,
        *,
        iflip3_fit_settings=None,
        iflip3_calibration=None,
        iflip3_temporal_bin_seconds=None,
    ):
        del manifest_row, data_root
        received.update(
            workflow=workflow,
            fit_settings=iflip3_fit_settings,
            calibration=iflip3_calibration,
            temporal_bin_seconds=iflip3_temporal_bin_seconds,
        )
        return session, {"source_path": str(raw)}

    monkeypatch.setattr(workflows, "load_lifetime_session", fake_load)

    progress = []
    outputs = preprocess_lifetime_sessions(
        "iflip3",
        [row],
        tmp_path,
        iflip3_calibration=calibration,
        iflip3_temporal_bin_seconds=1.0,
        progress_callback=lambda current, total, label: progress.append(
            (current, total, label)
        ),
    )

    assert received == {
        "workflow": "iflip3",
        "fit_settings": None,
        "calibration": calibration,
        "temporal_bin_seconds": 1.0,
    }
    loaded, _ = load_processed_lifetime_session("iflip3", row, tmp_path)
    np.testing.assert_allclose(
        loaded.continuous["long_lifetime_fraction"].values, fraction
    )
    assert outputs == [raw.with_name("recording-processed.npz")]
    assert (
        loaded.metadata["provenance"]["parameters"]["lifetime_fit"]["mode"]
        == "shared_multi_session_calibration"
    )
    assert loaded.metadata["provenance"]["parameters"]["temporal_bin_seconds"] == 1.0
    assert progress == [
        (0, 1, "Starting iflip3 preprocessing"),
        (0, 1, "Processing M1 260101 run 1"),
        (1, 1, "Processed M1 260101 run 1"),
    ]


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


def test_lifetime_glm_saves_mouse_level_kernels_in_picoseconds(monkeypatch, tmp_path):
    time = np.arange(0.0, 35.0, 0.1)
    rows = [
        {"mouse": "M1", "date": "260101", "run": 1},
        {"mouse": "M1", "date": "260102", "run": 1},
        {"mouse": "M2", "date": "260103", "run": 1},
    ]

    def fake_load(workflow, row, data_root):
        del workflow, data_root
        ensure = np.array([10.0, 20.0, 30.0])
        licks = ensure + 0.5
        values = 2.4 + 0.01 * np.sin(time) + 0.001 * int(row["run"])
        return (
            AlignedSession(
                session_id=f"{row['mouse']}_{row['date']}_{row['run']}",
                continuous={"tau": ContinuousSignal(time, values, "ns")},
                events={
                    "licks": EventSeries(licks),
                    "ensure": EventSeries(ensure),
                },
                metadata={},
            ),
            {},
        )

    monkeypatch.setattr(workflows, "load_processed_lifetime_session", fake_load)
    monkeypatch.setattr(
        workflows,
        "fit_ridge",
        lambda design, response, alpha, penalize: np.zeros(design.shape[1]),
    )

    progress = []
    outputs = workflows.run_lifetime_glm(
        "fluopulse",
        rows,
        tmp_path,
        tmp_path / "glm",
        signal="tau",
        lick_kernel_seconds=2.0,
        ensure_kernel_seconds=3.0,
        progress_callback=lambda current, total, label: progress.append(
            (current, total, label)
        ),
    )

    output_names = {path.name for path in outputs}
    assert "glm_session_kernels.csv" in output_names
    assert "glm_mouse_kernels.csv" in output_names
    assert "glm_group_kernels.csv" in output_names
    assert "glm_heldout_predictions.csv" in output_names
    assert "glm_session_baselines.csv" in output_names
    assert "glm_baseline_factor_effects.csv" in output_names
    assert "glm_baseline_coefficients.csv" in output_names
    assert "fluopulse_tau_glm_performance.svg" in output_names
    assert "fluopulse_tau_glm_session_baselines.svg" in output_names
    predictions = list(
        csv.DictReader(
            (tmp_path / "glm" / "glm_heldout_predictions.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert predictions
    first_prediction = predictions[0]
    assert float(first_prediction["residual"]) == pytest.approx(
        float(first_prediction["actual"]) - float(first_prediction["predicted"])
    )
    assert float(first_prediction["centered_residual"]) == pytest.approx(
        float(first_prediction["actual_centered"])
        - float(first_prediction["predicted_centered"])
    )
    baseline_rows = list(
        csv.DictReader(
            (tmp_path / "glm" / "glm_session_baselines.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert len(baseline_rows) == len(rows)
    expected_median_ps = float(np.median(2.4 + 0.01 * np.sin(time) + 0.001) * 1000)
    assert float(baseline_rows[0]["median"]) == pytest.approx(expected_median_ps)
    metadata = json.loads((tmp_path / "glm" / "glm_metadata.json").read_text())
    assert metadata["units"] == "ps"
    assert metadata["ylabel"] == "Δτ (ps)"
    assert metadata["ensure_model"] == "one shared kernel across all Ensure deliveries"
    assert "median-centered" in metadata["response_centering"]
    assert metadata["baseline_model"]["n_sessions"] == len(rows)
    assert metadata["ensure_basis_peak_seconds"]
    with np.load(tmp_path / "glm" / "glm_kernels.npz") as kernel_data:
        assert kernel_data["ensure_basis_values"].shape[1] == len(
            metadata["ensure_basis_peak_seconds"]
        )
    kernel_rows = list(
        csv.DictReader(
            (tmp_path / "glm" / "glm_group_kernels.csv").open(encoding="utf-8")
        )
    )
    assert {row["kernel"] for row in kernel_rows} == {"lick", "ensure"}
    assert progress[-1][0] == progress[-1][1]
    assert progress[-1][2] == "Lifetime GLM complete"
    assert any("Cross-validation: outer fold" in item[2] for item in progress)


def test_lifetime_baseline_model_reports_condition_beyond_mouse():
    rows = []
    for mouse_index, mouse in enumerate(("M1", "M2", "M3")):
        for condition, effect in (("fed", 0.0), ("fasted", 25.0)):
            rows.append(
                {
                    "mouse": mouse,
                    "group": "BLA",
                    "condition": condition,
                    "median": 1000.0 + 50.0 * mouse_index + effect,
                }
            )

    factors, coefficients, summary = workflows._lifetime_baseline_fixed_effects(rows)

    condition = next(row for row in factors if row["factor"] == "condition")
    assert condition["identifiable_beyond_other_factors"] is True
    assert condition["partial_r2"] > 0.9
    assert summary["n_sessions"] == 6
    assert any(row["term"].startswith("condition[") for row in coefficients)


def test_lifetime_baseline_model_flags_mouse_placement_alias():
    rows = [
        {
            "mouse": mouse,
            "group": group,
            "condition": condition,
            "median": baseline + effect,
        }
        for mouse, group, baseline in (("M1", "BLA", 1000.0), ("M2", "NAc", 1100.0))
        for condition, effect in (("fed", 0.0), ("fasted", 20.0))
    ]

    factors, _, summary = workflows._lifetime_baseline_fixed_effects(rows)

    placement = next(row for row in factors if row["factor"] == "group")
    assert placement["identifiable_beyond_other_factors"] is False
    assert np.isnan(placement["partial_r2"])
    assert summary["rank_deficient"] is True


def test_lifetime_preprocessing_error_identifies_session_and_paths(
    monkeypatch, tmp_path
):
    row = {
        "mouse": "AL166",
        "date": "261013",
        "run": 4,
        "iflip_path": "FLIM FLIP/AL166/session.iFLiP3",
        "nidaq_path": "Photometry/AL166/session-nidaq.mat",
        "running_path": "",
        "background_path": "",
    }
    monkeypatch.setattr(
        workflows,
        "load_lifetime_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Source pulse interval after index 64 is 0.3 s")
        ),
    )

    with pytest.raises(RuntimeError) as error:
        preprocess_lifetime_sessions("iflip3", [row], tmp_path)

    message = str(error.value)
    assert "AL166 261013 run 4" in message
    assert row["iflip_path"] in message
    assert row["nidaq_path"] in message
    assert "index 64" in message


def test_lifetime_psth_writes_mouse_and_pooled_heatmaps(monkeypatch, tmp_path):
    time = np.arange(0.0, 40.0, 0.1)

    def fake_load(workflow, row, data_root):
        del workflow, data_root
        session = AlignedSession(
            session_id=f"{row['mouse']}_{row['run']}",
            continuous={"tau": ContinuousSignal(time, np.sin(time / 4), "ns")},
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


def test_lifetime_psth_supports_derived_alignment_cue_classes_and_nulls(
    monkeypatch, tmp_path
):
    time = np.arange(0.0, 40.0, 0.1)

    def fake_load(workflow, row, data_root):
        del workflow, row, data_root
        return (
            AlignedSession(
                session_id="M1_260101_run001",
                continuous={"tau": ContinuousSignal(time, np.sin(time / 3), "ns")},
                events={
                    "ensure": EventSeries(np.array([10.0, 20.0, 30.0])),
                    "visual_cue": EventSeries(np.array([8.0, 18.0, 28.0])),
                    "licks": EventSeries(
                        np.array([8.5, 10.8, 18.5, 21.2, 31.5])
                    ),
                },
                intervals={
                    "visual_cue": IntervalSeries(
                        np.array([8.0, 18.0, 28.0]),
                        np.array([9.0, 19.0, 29.0]),
                    )
                },
            ),
            {"source_path": "synthetic.doric"},
        )

    monkeypatch.setattr(workflows, "load_processed_lifetime_session", fake_load)
    outputs = workflows.run_lifetime_psth(
        "fluopulse",
        [{"mouse": "M1", "date": "260101", "run": 1}],
        tmp_path,
        tmp_path / "derived",
        signal="tau",
        event="first_lick_after_ensure",
        derived_event_window=(0.5, 3.0),
        window=(-1.0, 1.0),
        dt=0.1,
        normalization="subtract",
        null_method="random_onsets",
        n_shuffles=5,
        random_seed=4,
        heatmap_sort="first_bout_latency",
        heatmap_sort_window=(0.0, 2.0),
    )

    names = {path.name for path in outputs}
    assert "null_diagnostics_summary.csv" in names
    metadata = json.loads(
        (tmp_path / "derived" / "analysis_metadata.json").read_text()
    )
    assert metadata["derived_event_window"] == [0.5, 3.0]
    assert metadata["null_method"] == "random_onsets"

    cue_outputs = workflows.run_lifetime_psth(
        "fluopulse",
        [{"mouse": "M1", "date": "260101", "run": 1}],
        tmp_path,
        tmp_path / "cue_class",
        signal="tau",
        event="visual_cue",
        trial_class="cue_only",
        window=(-1.0, 1.0),
        dt=0.1,
        normalization="none",
        heatmaps=False,
    )
    assert any(path.name == "fluopulse_tau_visual_cue_psth.png" for path in cue_outputs)
    cue_metadata = json.loads(
        (tmp_path / "cue_class" / "analysis_metadata.json").read_text()
    )
    assert cue_metadata["trial_class"] == "cue_only"


@pytest.mark.parametrize(
    ("signal", "expected_name", "expected_units"),
    [
        ("licking", "fluopulse_licking_ensure_psth.png", "licks/s"),
        ("running_speed", "fluopulse_running_speed_ensure_psth.png", "cm/s"),
    ],
)
def test_lifetime_psth_supports_behavior_signals_and_progress(
    monkeypatch, tmp_path, signal, expected_name, expected_units
):
    time = np.arange(0.0, 30.0, 0.1)

    def fake_load(workflow, row, data_root):
        del workflow, data_root
        return (
            AlignedSession(
                session_id=f"{row['mouse']}_{row['run']}",
                continuous={
                    "tau": ContinuousSignal(time, np.sin(time), "ns"),
                    "running_speed": ContinuousSignal(time, time / 10, "cm/s"),
                },
                events={
                    "ensure": EventSeries(np.array([10.0, 20.0])),
                    "licks": EventSeries(np.array([10.2, 10.4, 20.2])),
                },
            ),
            {"source_path": "synthetic.doric"},
        )

    monkeypatch.setattr(workflows, "load_processed_lifetime_session", fake_load)
    progress = []
    outputs = workflows.run_lifetime_psth(
        "fluopulse",
        [{"mouse": "M1", "date": "260101", "run": 1}],
        tmp_path,
        tmp_path / signal,
        signal=signal,
        event="ensure",
        window=(-1.0, 1.0),
        dt=0.1,
        normalization="zscore",
        heatmaps=False,
        progress_callback=lambda current, total, label: progress.append(
            (current, total, label)
        ),
    )

    assert expected_name in {path.name for path in outputs}
    assert progress == [(1, 1, "M1 260101 run 1")]
    metadata = json.loads((tmp_path / signal / "analysis_metadata.json").read_text())
    assert metadata["normalization"] == "none"
    assert metadata["units"] == expected_units


def test_lifetime_running_skips_sessions_without_running_data(monkeypatch, tmp_path):
    time = np.arange(0.0, 30.0, 0.1)

    def fake_load(workflow, row, data_root):
        del workflow, data_root
        continuous = {"tau": ContinuousSignal(time, np.sin(time), "ns")}
        if row["mouse"] == "M2":
            continuous["running_speed"] = ContinuousSignal(time, time / 10, "cm/s")
        return (
            AlignedSession(
                session_id=f"{row['mouse']}_{row['run']}",
                continuous=continuous,
                events={"ensure": EventSeries(np.array([10.0, 20.0]))},
            ),
            {"source_path": "synthetic.doric"},
        )

    monkeypatch.setattr(workflows, "load_processed_lifetime_session", fake_load)
    rows = [
        {"mouse": "M1", "date": "260101", "run": 1},
        {"mouse": "M2", "date": "260101", "run": 1},
    ]
    with pytest.warns(UserWarning, match="running data are unavailable"):
        outputs = workflows.run_lifetime_psth(
            "fluopulse",
            rows,
            tmp_path,
            tmp_path / "running",
            signal="running_speed",
            event="ensure",
            window=(-1.0, 1.0),
            dt=0.1,
            normalization="none",
            heatmaps=False,
        )

    assert "fluopulse_running_speed_ensure_psth.png" in {path.name for path in outputs}


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
