import csv
import json
from pathlib import Path

import numpy as np
import pytest
from lutaslab_core.functional_trials import FunctionalTrialDataset
from lutaslab_core.session import AlignedSession, ContinuousSignal, EventSeries

from lutaslab_photometry.functional_mixed import (
    build_flmm_dataset,
    flmm_formula,
    flmm_runtime_status,
    plot_flmm_coefficients,
    plot_flmm_fitted_responses,
    plot_flmm_mouse_deviations,
    prepare_flmm_analysis,
    run_prepared_flmm,
)
from lutaslab_photometry.session_manifest import processed_session_path


def _write_session(root, row, offset):
    path = processed_session_path(root, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(0.0, 12.01, 0.1)
    np.savez_compressed(
        path,
        mouse=row["mouse"],
        date=row["date"],
        run=row["run"],
        processed_schema_version="1.0",
        photo_time_465_ch1=time,
        photometry_465_ch1=time + offset,
        dff_ch1=np.sin(time) + offset,
        locomotion_time=time,
        processed_locomotion=np.zeros_like(time),
        cue_onset=np.array([0.2, 5.0, 10.0]),
        lick_times=np.array([5.2]),
    )


def test_builds_complete_conventional_dataset_and_tracks_exclusions(tmp_path):
    rows = [
        {"mouse": "M1", "date": "260101", "run": 1, "condition": "a"},
        {"mouse": "M2", "date": "260101", "run": 1, "condition": "a"},
        {"mouse": "M3", "date": "260101", "run": 1, "condition": "b"},
        {"mouse": "M4", "date": "260101", "run": 1, "condition": "b"},
    ]
    for index, row in enumerate(rows):
        _write_session(tmp_path, row, float(index))

    dataset = build_flmm_dataset(
        "conventional",
        rows,
        tmp_path,
        signal="dff",
        event="cue_onset",
        window=(-1.0, 1.0),
        dt=0.1,
        normalization="none",
    )

    assert dataset.n_trials == 8
    assert dataset.n_mice == 4
    assert len(dataset.exclusions) == 4
    assert flmm_formula(dataset, "condition").startswith("outcome ~ condition")
    for session in np.unique(dataset.session):
        selected = dataset.session == session
        assert np.mean(dataset.scalar_covariates["trial_z"][selected]) == pytest.approx(0)


@pytest.mark.parametrize(
    ("workflow", "signal"),
    [("fluopulse", "tau"), ("iflip3", "mpet")],
)
def test_builds_same_contract_for_lifetime_workflows(monkeypatch, workflow, signal):
    time = np.arange(0.0, 12.01, 0.1)

    def fake_loader(selected_workflow, row, data_root):
        assert selected_workflow == workflow
        offset = 0.0 if row["mouse"] == "M1" else 1.0
        session = AlignedSession(
            session_id=f"{row['mouse']}_{row['date']}_run{row['run']}",
            continuous={
                signal: ContinuousSignal(time, np.sin(time) + offset, "ns")
            },
            events={"ensure": EventSeries(np.array([0.2, 5.0, 10.0]))},
        )
        return session, {}

    monkeypatch.setattr(
        "lutaslab_photometry.functional_mixed.load_processed_lifetime_session",
        fake_loader,
    )
    rows = [
        {"mouse": "M1", "date": "260101", "run": 1, "condition": "a"},
        {"mouse": "M2", "date": "260101", "run": 1, "condition": "b"},
    ]

    dataset = build_flmm_dataset(
        workflow,
        rows,
        "unused",
        signal=signal,
        event="ensure",
        window=(-1.0, 1.0),
        dt=0.1,
        normalization="none",
    )

    assert dataset.n_trials == 4
    assert dataset.n_mice == 2
    assert dataset.metadata["workflow"] == workflow
    assert dataset.metadata["signal"] == signal


def test_preparation_writes_auditable_wide_table(tmp_path):
    rows = [
        {"mouse": "M1", "date": "260101", "run": 1, "group": "x"},
        {"mouse": "M2", "date": "260101", "run": 1, "group": "x"},
        {"mouse": "M3", "date": "260101", "run": 1, "group": "y"},
        {"mouse": "M4", "date": "260101", "run": 1, "group": "y"},
    ]
    for index, row in enumerate(rows):
        _write_session(tmp_path, row, float(index))
    dataset = build_flmm_dataset(
        "conventional",
        rows,
        tmp_path,
        signal="raw465",
        event="cue_onset",
        window=(-1.0, 1.0),
        normalization="subtract",
        baseline=(-1.0, 0.0),
    )

    output = tmp_path / "output"
    paths = prepare_flmm_analysis(dataset, output, effect="group")

    assert all(path.is_file() for path in paths)
    with (output / "flmm_input.csv").open(newline="", encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == dataset.n_trials
    assert "outcome.1" in table[0]
    specification = json.loads((output / "flmm_specification.json").read_text())
    assert specification["formula"] == "outcome ~ group + trial_z + (1 | mouse)"
    assert specification["reference_levels"]["group"] == "x"


def test_preparation_removes_stale_fit_results(tmp_path):
    rows = [
        {"mouse": "M1", "date": "260101", "run": 1, "group": "x"},
        {"mouse": "M2", "date": "260101", "run": 1, "group": "x"},
        {"mouse": "M3", "date": "260101", "run": 1, "group": "y"},
        {"mouse": "M4", "date": "260101", "run": 1, "group": "y"},
    ]
    for index, row in enumerate(rows):
        _write_session(tmp_path, row, float(index))
    dataset = build_flmm_dataset(
        "conventional",
        rows,
        tmp_path,
        signal="dff",
        event="cue_onset",
        window=(-1.0, 1.0),
        normalization="none",
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "flmm_coefficients.csv").write_text(
        "term,time,estimate\nold effect,0,1\n", encoding="utf-8"
    )
    (output / "flmm_old_effect.png").write_bytes(b"old")
    (output / "flmm_fit.rds").write_bytes(b"old")

    prepare_flmm_analysis(dataset, output, effect="group")

    assert not (output / "flmm_coefficients.csv").exists()
    assert not (output / "flmm_old_effect.png").exists()
    assert not (output / "flmm_fit.rds").exists()


def test_missing_r_runtime_preserves_prepared_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lutaslab_photometry.functional_mixed._find_rscript", lambda rscript=None: None
    )
    assert flmm_runtime_status()["ready"] is False
    with pytest.raises(RuntimeError, match="requires Rscript"):
        run_prepared_flmm(tmp_path)


def test_runtime_status_uses_discovered_rscript(monkeypatch):
    class Result:
        returncode = 0
        stdout = "1.0.1"

    monkeypatch.setattr(
        "lutaslab_photometry.functional_mixed._find_rscript",
        lambda rscript=None: r"C:\Program Files\R\bin\Rscript.exe",
    )
    monkeypatch.setattr(
        "lutaslab_photometry.functional_mixed.subprocess.run",
        lambda *args, **kwargs: Result(),
    )

    status = flmm_runtime_status()

    assert status == {
        "ready": True,
        "rscript": r"C:\Program Files\R\bin\Rscript.exe",
        "fastfmm_version": "1.0.1",
    }


def test_python_plotter_reads_backend_contract(tmp_path):
    result = tmp_path / "flmm_coefficients.csv"
    result.write_text(
        "term,time,estimate,pointwise_lower,pointwise_upper,joint_lower,joint_upper\n"
        "conditionb,-1,0.1,0.0,0.2,-0.1,0.3\n"
        "conditionb,0,0.2,0.1,0.3,0.0,0.4\n",
        encoding="utf-8",
    )

    paths = plot_flmm_coefficients(result, tmp_path)

    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert all(path.is_file() for path in paths)


def test_reconstructs_condition_responses_from_intercept_and_contrast(tmp_path):
    result = tmp_path / "flmm_coefficients.csv"
    result.write_text(
        "term,time,estimate,pointwise_lower,pointwise_upper,joint_lower,joint_upper\n"
        "(Intercept),-1,1.0,0.8,1.2,0.7,1.3\n"
        "(Intercept),0,2.0,1.8,2.2,1.7,2.3\n"
        "conditiontrained,-1,0.5,0.2,0.8,0.1,0.9\n"
        "conditiontrained,0,-0.25,-0.5,0.0,-0.6,0.1\n",
        encoding="utf-8",
    )
    specification = tmp_path / "flmm_specification.json"
    specification.write_text(
        json.dumps(
            {
                "effect": "condition",
                "levels": {"condition": ["naive", "trained"]},
                "reference_levels": {"condition": "naive"},
            }
        ),
        encoding="utf-8",
    )

    paths = plot_flmm_fitted_responses(result, specification, tmp_path)

    assert {path.suffix for path in paths} == {".csv", ".png", ".svg"}
    with (tmp_path / "flmm_fitted_responses.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    trained = [row for row in rows if row["curve"] == "Condition: trained"]
    assert [float(row["estimate"]) for row in trained] == [1.5, 1.75]


def test_plots_mouse_random_effect_deviations(tmp_path):
    random_effects = tmp_path / "flmm_mouse_random_effects.csv"
    random_effects.write_text(
        "mouse,time,deviation\nM1,-1,0.1\nM1,0,0.2\nM2,-1,-0.1\nM2,0,-0.2\n",
        encoding="utf-8",
    )

    paths = plot_flmm_mouse_deviations(random_effects, tmp_path)

    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert all(path.is_file() for path in paths)


def test_progress_model_requires_trial_order_variation():
    dataset = FunctionalTrialDataset(
        time=np.array([-1.0, 0.0]),
        outcome=np.ones((2, 2)),
        mouse=np.array(["M1", "M2"]),
        session=np.array(["S1", "S2"]),
        trial_index=np.array([1, 1]),
        alignment_time=np.array([2.0, 2.0]),
        scalar_covariates={
            "group": np.array(["a", "a"]),
            "condition": np.array(["a", "a"]),
            "trial_z": np.zeros(2),
            "session_z": np.zeros(2),
        },
    )

    with pytest.raises(ValueError, match="variation in within-session trial order"):
        flmm_formula(dataset, "progress")


def test_packaged_r_backend_requests_and_exports_random_effects():
    backend = (
        Path(__file__).parents[1]
        / "src"
        / "lutaslab_photometry"
        / "r"
        / "fit_flmm.R"
    ).read_text(encoding="utf-8")

    assert "REs = TRUE" in backend
    assert 'file.path(output_dir, "flmm_mouse_random_effects.csv")' in backend
