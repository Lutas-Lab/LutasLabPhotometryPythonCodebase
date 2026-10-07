import importlib.util
import shutil
import unittest
from pathlib import Path

import numpy as np

from lutaslab_photometry.psth_statistics import (
    analyze_manifest_metrics,
    compute_response_metrics,
    prism_wide_rows,
    run_pairwise_tests,
    summarize_group_metrics,
    summarize_mouse_metrics,
    summarize_shuffle_tests,
)
from lutaslab_photometry.session_manifest import processed_session_path

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
        cue_offset=events + 0.5,
        lick_times=np.array([10.25, 10.75, 20.25]),
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

    def test_shuffle_p_value_uses_plus_one_correction(self):
        mouse_rows = [
            {
                "mouse": "M1",
                "group": "control",
                "condition": "rewarded",
                "metric": "mean",
                "value": 1_000.0,
            }
        ]
        null_sessions = [
            {
                "mouse": "M1",
                "group": "control",
                "condition": "rewarded",
                "metric": "mean",
                "values": np.arange(500, dtype=float),
            }
        ]

        row = summarize_shuffle_tests(mouse_rows, null_sessions)[0]

        self.assertEqual(row["n_shuffles"], 500)
        self.assertEqual(row["empirical_p_two_sided"], 1 / 501)
        self.assertGreater(row["empirical_p_two_sided"], 0.0)

    def test_shuffle_p_value_excludes_nonfinite_null_values(self):
        mouse_rows = [
            {
                "mouse": "M1",
                "group": "control",
                "condition": "rewarded",
                "metric": "mean",
                "value": 10.0,
            }
        ]
        null_sessions = [
            {
                "mouse": "M1",
                "group": "control",
                "condition": "rewarded",
                "metric": "mean",
                "values": np.array([0.0, 1.0, 2.0, np.nan, np.inf]),
            }
        ]

        row = summarize_shuffle_tests(mouse_rows, null_sessions)[0]

        self.assertEqual(row["n_shuffles"], 3)
        self.assertEqual(row["empirical_p_two_sided"], 1 / 4)

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
            for session, amplitude in zip(sessions, (1.0, 3.0), strict=True):
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

    def test_manifest_metrics_preserve_original_event_indices_after_filtering(self):
        root = Path("tests/_psth_statistics_data")
        session = {"mouse": "M1", "date": "260101", "run": 1}
        try:
            _write_session(root, session, 1.0)
            results = analyze_manifest_metrics(
                [session],
                root,
                trial_class="cue_only",
                post_cue_window=2.0,
                window=(-2.0, 3.0),
                response_window=(0.0, 2.0),
                dt=0.1,
                normalization="none",
                metrics=("mean",),
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

        self.assertEqual(len(results["trial_rows"]), 1)
        self.assertEqual(results["trial_rows"][0]["event_index"], 1)
        self.assertEqual(results["trial_rows"][0]["trial_class"], "cue_only")

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
        tests = run_pairwise_tests(
            rows,
            test="auto",
            n_bootstrap=500,
            bootstrap_seed=17,
        )
        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0]["test"], "paired_t")
        self.assertEqual(tests[0]["n_pairs"], 3)
        self.assertLess(tests[0]["p_value"], 0.2)
        self.assertIn("p_adjusted_holm", tests[0])
        self.assertEqual(tests[0]["holm_family"], "all_pairwise_comparisons_in_run")
        self.assertEqual(tests[0]["holm_family_size"], 1)
        self.assertEqual(tests[0]["bootstrap_resamples"], 500)
        self.assertEqual(tests[0]["bootstrap_valid_difference_resamples"], 500)
        self.assertGreater(tests[0]["bootstrap_valid_effect_resamples"], 0)
        self.assertTrue(np.isfinite(tests[0]["difference_ci_lower_bootstrap"]))
        self.assertTrue(np.isfinite(tests[0]["effect_size_ci_upper_bootstrap"]))

        repeated = run_pairwise_tests(
            rows,
            test="auto",
            n_bootstrap=500,
            bootstrap_seed=17,
        )
        self.assertEqual(
            tests[0]["effect_size_ci_lower_bootstrap"],
            repeated[0]["effect_size_ci_lower_bootstrap"],
        )

    @unittest.skipUnless(SCIPY_AVAILABLE, "SciPy is not installed")
    def test_auto_uses_welch_for_independent_groups_and_one_holm_family(self):
        rows = []
        for metric, scale in (("mean", 1.0), ("auc", 2.0)):
            for mouse, group, value in (
                ("M1", "control", 1.0),
                ("M2", "control", 2.0),
                ("M3", "treated", 4.0),
                ("M4", "treated", 7.0),
            ):
                rows.append(
                    {
                        "mouse": mouse,
                        "group": group,
                        "condition": "rewarded",
                        "metric": metric,
                        "value": scale * value,
                    }
                )

        tests = run_pairwise_tests(rows, n_bootstrap=100, bootstrap_seed=9)

        self.assertEqual(len(tests), 2)
        self.assertEqual({row["test"] for row in tests}, {"welch"})
        self.assertEqual({row["effect_size_name"] for row in tests}, {"hedges_g"})
        self.assertEqual({row["holm_family_size"] for row in tests}, {2})


if __name__ == "__main__":
    unittest.main()
