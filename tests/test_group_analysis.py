from pathlib import Path
import shutil
import unittest

import numpy as np

from src.group_analysis import (
    compute_manifest_psth,
    compute_manifest_psth_strata,
    extract_perievent_trials,
    generate_null_onsets,
    normalize_trials,
)
from src.session_manifest import processed_session_path


def _write_processed_session(data_root, info, value):
    path = processed_session_path(data_root, info)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.linspace(0.0, 20.0, 201)
    np.savez_compressed(
        path,
        mouse=info["mouse"],
        date=info["date"],
        run=info["run"],
        processed_schema_version="1.0",
        photo_time_465_ch1=time,
        photometry_465_ch1=np.full_like(time, value),
        dff_ch1=np.full_like(time, value),
        photo_time_465_ch2=time,
        photometry_465_ch2=np.full_like(time, value * 10),
        dff_ch2=np.full_like(time, value * 10),
        locomotion_time=time,
        processed_locomotion=np.zeros_like(time),
        cue_onset=np.array([10.0]),
    )


class GroupAnalysisTests(unittest.TestCase):
    def test_null_onsets_are_reproducible_and_session_bounded(self):
        events = np.array([4.0, 8.0])
        first = generate_null_onsets(
            events,
            (1.0, 11.0),
            n_shuffles=20,
            method="random_onsets",
            exclusion=0.5,
            rng=np.random.default_rng(12),
        )
        second = generate_null_onsets(
            events,
            (1.0, 11.0),
            n_shuffles=20,
            method="random_onsets",
            exclusion=0.5,
            rng=np.random.default_rng(12),
        )
        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all((first >= 1.0) & (first <= 11.0)))
        self.assertTrue(
            np.all(np.abs(first[:, :, None] - events[None, None, :]) >= 0.5)
        )

        shifted = generate_null_onsets(
            events,
            (1.0, 11.0),
            n_shuffles=20,
            method="circular_shift",
            rng=np.random.default_rng(12),
        )
        np.testing.assert_allclose(np.mod(shifted[:, 1] - shifted[:, 0], 10.0), 4.0)

    def test_extract_and_normalize_trials(self):
        time = np.arange(0.0, 10.1, 0.1)
        signal = time.copy()
        peri_time, trials, valid = extract_perievent_trials(
            time, signal, [0.5, 5.0, 9.8], window=(-1.0, 1.0), dt=0.1
        )
        np.testing.assert_array_equal(valid, [1])
        np.testing.assert_allclose(trials[0], 5.0 + peri_time)

        normalized = normalize_trials(
            peri_time, trials, normalization="subtract", baseline=(-1.0, 0.0)
        )
        self.assertAlmostEqual(float(np.mean(normalized[0, peri_time < 0])), 0.0)

    def test_hierarchy_averages_sessions_then_mice(self):
        sessions = [
            {"mouse": "M1", "date": "260101", "run": 1},
            {"mouse": "M1", "date": "260102", "run": 1},
            {"mouse": "M2", "date": "260101", "run": 1},
        ]
        data_root = Path("tests/_group_analysis_data")
        try:
            for info, value in zip(sessions, (1.0, 3.0, 5.0)):
                _write_processed_session(data_root, info, value)

            results = compute_manifest_psth(
                sessions,
                data_root,
                window=(-1.0, 1.0),
                dt=0.1,
                normalization="none",
                null_method="random_onsets",
                n_shuffles=12,
                random_seed=7,
            )
        finally:
            shutil.rmtree(data_root, ignore_errors=True)

        np.testing.assert_allclose(results["mouse_results"]["M1"]["mean"], 2.0)
        np.testing.assert_allclose(results["mouse_results"]["M2"]["mean"], 5.0)
        np.testing.assert_allclose(results["group_mean"], 3.5)
        np.testing.assert_allclose(results["group_sem"], 1.5)
        np.testing.assert_allclose(results["group_null_mean"], 3.5)
        self.assertEqual(results["group_null_matrix"].shape[0], 12)
        self.assertEqual(results["n_mice"], 2)

    def test_manifest_selects_photoreceiver_channel_per_session(self):
        sessions = [
            {"mouse": "M1", "date": "260101", "run": 1, "channel": "1"},
            {"mouse": "M2", "date": "260101", "run": 1, "channel": "2"},
        ]
        data_root = Path("tests/_group_analysis_data")
        try:
            for info, value in zip(sessions, (1.0, 2.0)):
                _write_processed_session(data_root, info, value)
            results = compute_manifest_psth(
                sessions,
                data_root,
                channel="manifest",
                window=(-1.0, 1.0),
                dt=0.1,
                normalization="none",
            )
        finally:
            shutil.rmtree(data_root, ignore_errors=True)

        np.testing.assert_allclose(results["mouse_results"]["M1"]["mean"], 1.0)
        np.testing.assert_allclose(results["mouse_results"]["M2"]["mean"], 20.0)
        self.assertEqual([row["channel"] for row in results["session_results"]], [1, 2])

    def test_manifest_strata_keep_conditions_separate(self):
        sessions = [
            {
                "mouse": mouse,
                "date": "260101",
                "run": run,
                "group": "G",
                "condition": condition,
                "channel": "1",
            }
            for mouse, run, condition in (
                ("M1", 1, "Naive"),
                ("M1", 2, "Trained"),
                ("M2", 1, "Naive"),
                ("M2", 2, "Trained"),
            )
        ]
        data_root = Path("tests/_group_analysis_data")
        try:
            for info, value in zip(sessions, (1.0, 2.0, 3.0, 4.0)):
                _write_processed_session(data_root, info, value)
            results = compute_manifest_psth_strata(
                sessions,
                data_root,
                window=(-1.0, 1.0),
                dt=0.1,
                normalization="none",
            )
        finally:
            shutil.rmtree(data_root, ignore_errors=True)

        self.assertEqual(set(results), {("G", "Naive"), ("G", "Trained")})
        np.testing.assert_allclose(results[("G", "Naive")]["group_mean"], 2.0)
        np.testing.assert_allclose(results[("G", "Trained")]["group_mean"], 3.0)
        self.assertEqual(results[("G", "Naive")]["mouse_names"], ["M1", "M2"])
