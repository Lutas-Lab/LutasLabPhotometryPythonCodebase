import numpy as np
import unittest

from src.nemos_analysis import (
    _apply_temporal_basis,
    make_gapped_folds,
    make_time_gapped_folds,
)


class TemporalDesignTests(unittest.TestCase):
    def test_causal_design_never_uses_future_predictor_samples(self):
        signal = np.zeros(30)
        signal[10] = 1.0
        lags = np.arange(-4, 1)
        basis_values = np.column_stack([lags, lags**2, np.ones(len(lags))])
        X = _apply_temporal_basis(signal, lags, basis_values)

        np.testing.assert_allclose(X[10], basis_values[-1])
        np.testing.assert_allclose(X[14], basis_values[0])
        np.testing.assert_allclose(X[9], 0.0)
        self.assertTrue(np.isnan(X[:4]).all())
        self.assertTrue(np.isfinite(X[-1]).all())

    def test_asymmetric_design_anchors_each_lag_to_response_time(self):
        signal = np.zeros(40)
        signal[20] = 1.0
        lags = np.arange(-2, 6)
        basis_values = np.column_stack([lags, np.ones(len(lags))])
        X = _apply_temporal_basis(signal, lags, basis_values)

        for response_index in range(15, 23):
            lag = 20 - response_index
            np.testing.assert_allclose(X[response_index], basis_values[lag - lags[0]])

        np.testing.assert_allclose(X[14], 0.0)
        np.testing.assert_allclose(X[23], 0.0)

    def test_fold_count_is_validated(self):
        with self.assertRaisesRegex(ValueError, "n_folds"):
            make_gapped_folds(3, n_folds=4)

    def test_time_gap_uses_elapsed_time_across_missing_samples(self):
        time = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
        folds = make_time_gapped_folds(time, n_folds=2, gap_seconds=2.0)
        train, test = folds[0]
        np.testing.assert_array_equal(test, [0, 1, 2])
        np.testing.assert_array_equal(train, [3, 4, 5])
