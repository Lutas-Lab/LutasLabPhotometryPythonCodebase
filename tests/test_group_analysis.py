from pathlib import Path
import shutil
import unittest

import numpy as np

from src.group_analysis import (
    compute_manifest_psth,
    extract_perievent_trials,
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
        locomotion_time=time,
        processed_locomotion=np.zeros_like(time),
        cue_onset=np.array([10.0]),
    )


class GroupAnalysisTests(unittest.TestCase):
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
            )
        finally:
            shutil.rmtree(data_root, ignore_errors=True)

        np.testing.assert_allclose(results["mouse_results"]["M1"]["mean"], 2.0)
        np.testing.assert_allclose(results["mouse_results"]["M2"]["mean"], 5.0)
        np.testing.assert_allclose(results["group_mean"], 3.5)
        np.testing.assert_allclose(results["group_sem"], 1.5)
        self.assertEqual(results["n_mice"], 2)
