import importlib.util
import unittest

import numpy as np

from src.forecasting import (
    build_forecast_dataset,
    fit_forecast_models,
    make_forward_folds,
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
