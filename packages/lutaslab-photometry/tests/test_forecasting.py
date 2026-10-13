import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lutaslab_photometry.cli.run_forecasting import (
    _plot_group_performance,
    _plot_predictor_kernels,
)
from lutaslab_photometry.forecasting import (
    build_forecast_dataset,
    fit_forecast_models,
    make_forward_folds,
    summarize_forecast_kernels,
)

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


def _signals(time, photometry=None, locomotion=None):
    time = np.asarray(time, dtype=float)
    if photometry is None:
        photometry = time.copy()
    if locomotion is None:
        locomotion = np.zeros_like(time)
    return {
        "time": time,
        "photometry": np.asarray(photometry, dtype=float),
        "locomotion": np.asarray(locomotion, dtype=float),
        "licking": np.zeros_like(time),
        "cue": np.zeros_like(time),
        "solenoid": np.zeros_like(time),
    }


class ForecastingTests(unittest.TestCase):
    def test_zero_horizon_excludes_current_target_from_predictors(self):
        time = np.arange(0.0, 20.0, 0.1)
        signals = {
            "time": time,
            "photometry": np.sin(time),
            "locomotion": np.cos(time),
            "licking": np.zeros_like(time),
            "cue": np.zeros_like(time),
            "solenoid": np.zeros_like(time),
        }
        dataset = build_forecast_dataset(
            signals,
            target="photometry",
            horizon=0.0,
            history=0.5,
            lag_step=0.1,
        )
        assert "photometry[t-0s]" not in dataset["feature_names"]
        assert "photometry[t-0.1s]" in dataset["feature_names"]
        assert "locomotion[t-0s]" in dataset["feature_names"]
        locomotion_index = dataset["feature_names"].index("locomotion[t-0s]")
        assert dataset["feature_predictors"][locomotion_index] == "locomotion"
        assert dataset["feature_lags"][locomotion_index] == 0.0

    def test_dataset_uses_only_present_and_past_features(self):
        signals = _signals(np.arange(20, dtype=float))
        dataset = build_forecast_dataset(
            signals,
            target="photometry",
            horizon=2.0,
            history=2.0,
            lag_step=1.0,
        )

        photometry_columns = dataset["feature_groups"]["photometry"]
        np.testing.assert_allclose(dataset["X"][0, photometry_columns], [2.0, 1.0, 0.0])
        self.assertEqual(dataset["anchor_time"][0], 2.0)
        self.assertEqual(dataset["target_time"][0], 4.0)
        self.assertEqual(dataset["y"][0], 4.0)

    def test_forward_folds_train_strictly_before_gapped_test(self):
        time = np.arange(100, dtype=float)
        folds = make_forward_folds(
            time,
            n_folds=5,
            gap_seconds=5.0,
            initial_train_fraction=0.5,
        )
        for train_indices, test_indices in folds:
            self.assertLessEqual(time[train_indices[-1]], time[test_indices[0]] - 6.0)
            self.assertLess(train_indices[-1], test_indices[0])

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_cross_modal_model_forecasts_synthetic_future_photometry(self):
        rng = np.random.default_rng(4)
        time = np.arange(600, dtype=float)
        locomotion = rng.normal(size=len(time))
        photometry = rng.normal(scale=0.01, size=len(time))
        photometry[1:] += 2.0 * locomotion[:-1]
        dataset = build_forecast_dataset(
            _signals(time, photometry=photometry, locomotion=locomotion),
            target="photometry",
            horizon=1.0,
            history=0.0,
            lag_step=1.0,
        )
        fitted = fit_forecast_models(
            dataset,
            n_folds=3,
            alpha=0.01,
            initial_train_fraction=0.5,
        )

        self.assertGreater(fitted["models"]["cross_modal"]["metrics"]["r2"], 0.95)
        self.assertGreater(fitted["models"]["combined"]["metrics"]["r2"], 0.95)
        self.assertLess(fitted["models"]["history_only"]["metrics"]["r2"], 0.1)
        self.assertIn("constant_baseline", fitted["models"])
        combined = fitted["models"]["combined"]
        self.assertEqual(
            len(combined["final_coefficients"]), len(combined["feature_names"])
        )
        self.assertIn("locomotion", combined["feature_predictors"])
        locomotion_index = combined["feature_predictors"].index("locomotion")
        self.assertAlmostEqual(
            combined["final_coefficients"][locomotion_index], 2.0, places=2
        )

    def test_kernel_summary_averages_sessions_before_mice(self):
        rows = []
        for mouse, values in (("M1", (1.0, 3.0)), ("M2", (5.0,))):
            for run, value in enumerate(values, start=1):
                rows.append(
                    {
                        "mouse": mouse,
                        "run": run,
                        "horizon": 0.0,
                        "model": "combined",
                        "predictor": "licking",
                        "lag_s": 0.5,
                        "coefficient": value,
                    }
                )

        mouse_rows, group_rows = summarize_forecast_kernels(rows)

        by_mouse = {row["mouse"]: row["coefficient"] for row in mouse_rows}
        assert by_mouse == {"M1": 2.0, "M2": 5.0}
        assert group_rows[0]["mean"] == 3.5
        assert group_rows[0]["n_mice"] == 2

    def test_gui_glm_figures_support_single_horizon_and_predictor_kernels(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        metric_mouse_rows = []
        metric_group_rows = []
        for model_index, model in enumerate(
            ("constant_baseline", "history_only", "cross_modal", "combined")
        ):
            values = []
            for mouse_index, mouse in enumerate(("M1", "M2")):
                value = 0.1 * model_index + 0.01 * mouse_index
                values.append(value)
                metric_mouse_rows.append(
                    {
                        "mouse": mouse,
                        "horizon": 0.0,
                        "model": model,
                        "value": value,
                    }
                )
            metric_group_rows.append(
                {
                    "horizon": 0.0,
                    "model": model,
                    "mean": float(np.mean(values)),
                    "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))),
                }
            )

        performance_paths = _plot_group_performance(
            metric_mouse_rows,
            metric_group_rows,
            "photometry",
            "r2",
            tmp_path / "performance",
            ("png",),
            100,
        )

        kernel_session_rows = []
        for mouse_index, mouse in enumerate(("M1", "M2")):
            for predictor in (
                "photometry",
                "locomotion",
                "licking",
                "cue",
                "solenoid",
            ):
                for lag_s in (0.0, 0.5, 1.0):
                    kernel_session_rows.append(
                        {
                            "mouse": mouse,
                            "horizon": 0.0,
                            "model": "combined",
                            "predictor": predictor,
                            "lag_s": lag_s,
                            "coefficient": lag_s + mouse_index,
                        }
                    )
        mouse_kernels, group_kernels = summarize_forecast_kernels(
            kernel_session_rows
        )
        kernel_paths = _plot_predictor_kernels(
            mouse_kernels,
            group_kernels,
            target="photometry",
            photometry_source="dff",
            output_base=tmp_path / "kernels",
            formats=("png",),
            dpi=100,
        )

        assert performance_paths == [tmp_path / "performance.png"]
        assert kernel_paths == [tmp_path / "kernels.png"]
        assert all(path.is_file() for path in (*performance_paths, *kernel_paths))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_photometry_forecasts_synthetic_future_licks(self):
        rng = np.random.default_rng(9)
        time = np.arange(800, dtype=float)
        photometry = rng.normal(size=len(time))
        signals = _signals(time, photometry=photometry)
        signals["licking"][1:] = (photometry[:-1] > 0).astype(float)
        dataset = build_forecast_dataset(
            signals,
            target="lick_binary",
            horizon=1.0,
            history=0.0,
            lag_step=1.0,
            target_window=1.0,
        )
        fitted = fit_forecast_models(
            dataset,
            n_folds=3,
            alpha=0.1,
            initial_train_fraction=0.5,
        )

        cross_score = fitted["models"]["cross_modal"]["metrics"]["average_precision"]
        history_score = fitted["models"]["history_only"]["metrics"][
            "average_precision"
        ]
        self.assertGreater(cross_score, 0.95)
        self.assertGreater(cross_score, history_score + 0.3)
        cross_metrics = fitted["models"]["cross_modal"]["metrics"]
        self.assertIn("roc_auc", cross_metrics)
        self.assertIn("positive_prevalence", cross_metrics)
        self.assertAlmostEqual(
            cross_metrics["average_precision_gain_over_prevalence"],
            cross_metrics["average_precision"] - cross_metrics["positive_prevalence"],
        )
        baseline = fitted["models"]["constant_baseline"]["prediction"]
        for train_indices, test_indices in fitted["folds"]:
            np.testing.assert_allclose(
                baseline[test_indices], np.mean(dataset["y"][train_indices])
            )

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_nested_alpha_selection_stays_inside_outer_training_folds(self):
        rng = np.random.default_rng(12)
        time = np.arange(800, dtype=float)
        locomotion = rng.normal(size=len(time))
        photometry = rng.normal(scale=0.2, size=len(time))
        photometry[1:] += locomotion[:-1]
        dataset = build_forecast_dataset(
            _signals(time, photometry=photometry, locomotion=locomotion),
            target="photometry",
            horizon=1.0,
            history=0.0,
            lag_step=1.0,
        )

        fitted = fit_forecast_models(
            dataset,
            n_folds=3,
            alpha=None,
            candidate_alphas=(0.01, 1.0),
            n_inner_folds=2,
            initial_train_fraction=0.5,
        )

        for model_name in ("history_only", "cross_modal", "combined"):
            result = fitted["models"][model_name]
            self.assertEqual(result["alpha_selection"], "nested_forward_cv")
            self.assertEqual(len(result["selected_alphas"]), 3)
            self.assertEqual(len(result["inner_alpha_scores"]), 3)
            self.assertTrue(set(result["selected_alphas"]).issubset({0.01, 1.0}))
