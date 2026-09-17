import importlib.util
from pathlib import Path
import shutil
import unittest

import numpy as np

from src.psth_statistics import (
    analyze_manifest_metrics,
    compute_response_metrics,
    prism_wide_rows,
    run_pairwise_tests,
    summarize_group_metrics,
    summarize_mouse_metrics,
)
from src.session_manifest import processed_session_path


SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


def _write_session(root, info, amplitude):
    path = processed_session_path(root, info)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(0.0, 30.01, 0.1)
    signal = np.zeros_like(time)
    events = np.array([10.0, 20.0])
    for event in events:
        response = (time >= event) & (time <= event + 2.0)
        signal[response] += amplitude
    np.savez_compressed(
        path,
        mouse=info["mouse"],
        date=info["date"],
        run=info["run"],
        processed_schema_version="1.0",
        photo_time_465_ch1=time,
        photometry_465_ch1=signal,
        dff_ch1=signal,
        locomotion_time=time,
        processed_locomotion=np.zeros_like(time),
        cue_onset=events,
    )


class PsthStatisticsTests(unittest.TestCase):
    def test_response_metrics_include_amplitude_auc_and_latency(self):
        time = np.array([-1.0, 0.0, 1.0, 2.0])
        traces = np.array([[0.0, 0.0, 1.0, 2.0]])
        result = compute_response_metrics(
            time,
            traces,
            response_window=(0.0, 2.0),
            peak_smoothing=0.0,
        )
        self.assertAlmostEqual(result["mean"][0], 1.0)
        self.assertAlmostEqual(result["auc"][0], 2.0)
        self.assertAlmostEqual(result["peak"][0], 2.0)
        self.assertAlmostEqual(result["peak_latency"][0], 2.0)
        self.assertAlmostEqual(result["trough"][0], 0.0)

    def test_hierarchy_averages_sessions_before_mice(self):
        session_rows = [
            {
                "mouse": "M1",
                "group": "control",
                "condition": "rewarded",
                "metric": "mean",
                "value": 1.0,
                "n_trials": 2,
            },
            {
                "mouse": "M1",
                "group": "control",
                "condition": "rewarded",
                "metric": "mean",
                "value": 3.0,
                "n_trials": 2,
            },
            {
                "mouse": "M2",
                "group": "control",
                "condition": "rewarded",
                "metric": "mean",
                "value": 6.0,
                "n_trials": 8,
            },
        ]
        mouse_rows = summarize_mouse_metrics(session_rows)
        values = {row["mouse"]: row["value"] for row in mouse_rows}
        self.assertEqual(values, {"M1": 2.0, "M2": 6.0})
        group_rows = summarize_group_metrics(mouse_rows)
        self.assertEqual(group_rows[0]["mean"], 4.0)
        self.assertEqual(group_rows[0]["n_mice"], 2)

    def test_manifest_analysis_and_prism_export(self):
        root = Path("tests/_psth_statistics_data")
        sessions = [
            {
                "mouse": "M1",
                "date": "260101",
                "run": 1,
                "group": "control",
                "condition": "rewarded",
            },
            {
                "mouse": "M2",
                "date": "260101",
                "run": 1,
                "group": "control",
                "condition": "rewarded",
            },
        ]
        try:
            for session, amplitude in zip(sessions, (1.0, 3.0)):
                _write_session(root, session, amplitude)
            results = analyze_manifest_metrics(
                sessions,
                root,
                window=(-2.0, 3.0),
                baseline=(-2.0, 0.0),
                response_window=(0.0, 2.0),
                dt=0.1,
                normalization="subtract",
                metrics=("mean", "auc"),
                null_method="random_onsets",
                n_shuffles=10,
                random_seed=4,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

        means = [
            row["mean"]
            for row in results["group_rows"]
            if row["metric"] == "mean"
        ]
        self.assertEqual(len(results["trial_rows"]), 8)
        self.assertAlmostEqual(means[0], 2.0)
        self.assertEqual(len(results["shuffle_rows"]), 2)
        prism = prism_wide_rows(results["mouse_rows"])
        self.assertIn("control__rewarded__mean", prism[0])

    @unittest.skipUnless(SCIPY_AVAILABLE, "SciPy is not installed")
    def test_pairwise_conditions_use_matched_mice(self):
        rows = []
        for mouse, first, second in (
            ("M1", 1.0, 2.0),
            ("M2", 2.0, 4.0),
            ("M3", 3.0, 6.0),
        ):
            for condition, value in (("A", first), ("B", second)):
                rows.append(
                    {
                        "mouse": mouse,
                        "group": "control",
                        "condition": condition,
                        "metric": "mean",
                        "value": value,
                    }
                )
        tests = run_pairwise_tests(rows, test="auto")
        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0]["test"], "paired_t")
        self.assertEqual(tests[0]["n_pairs"], 3)
        self.assertLess(tests[0]["p_value"], 0.2)
        self.assertIn("p_adjusted_holm", tests[0])


if __name__ == "__main__":
    unittest.main()
