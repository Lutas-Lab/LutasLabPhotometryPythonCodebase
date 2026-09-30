import numpy as np

from lutaslab_core.glm import (
    apply_lag_basis,
    blocked_folds,
    convolve_basis,
    event_times_to_counts,
    fit_grouped_ridge_cv,
    lagged_basis_matrix,
    lagged_signal_matrix,
    predict_lagged_signal,
    raised_cosine_basis,
    reconstruct_kernel,
)


def test_basis_convolution_and_kernel_reconstruction():
    basis = raised_cosine_basis((0.0, 1.0), count=3, dt=0.1)
    events = np.zeros(30)
    events[5] = 1
    design = convolve_basis(events, basis.values)
    assert design.shape == (30, 3)
    coefficients = np.array([1.0, -0.5, 0.25])
    np.testing.assert_allclose(reconstruct_kernel(basis.values, coefficients), basis.values @ coefficients)


def test_signed_lag_basis_marks_unobserved_edges():
    signal = np.arange(10, dtype=float)
    design = apply_lag_basis(signal, [-1, 0, 1], np.eye(3))
    assert np.all(np.isnan(design[0]))
    assert np.all(np.isnan(design[-1]))


def test_raw_lag_matrix_matches_matlab_shift_convention():
    signal = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    design, lags = lagged_signal_matrix(signal, pre_samples=1, post_samples=2)
    np.testing.assert_array_equal(lags, [-1, 0, 1, 2])
    np.testing.assert_array_equal(design[:, 0], [1, 0, 2, 0, 0])
    np.testing.assert_array_equal(design[:, 1], signal)
    np.testing.assert_array_equal(design[:, 2], [0, 0, 1, 0, 2])
    np.testing.assert_array_equal(design[:, 3], [0, 0, 0, 1, 0])


def test_lagged_prediction_matches_materialized_design():
    signal = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    design, _ = lagged_signal_matrix(signal, pre_samples=1, post_samples=2)
    kernel = np.array([0.5, 1.0, -0.25, 0.75])
    expected = 2.0 + design @ kernel
    np.testing.assert_allclose(
        predict_lagged_signal(signal, kernel, pre_samples=1, intercept=2.0),
        expected,
    )


def test_lagged_basis_matches_materialized_raw_lags():
    signal = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    raw, lags = lagged_signal_matrix(signal, pre_samples=1, post_samples=2)
    basis = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.25, 0.75]])
    np.testing.assert_allclose(lagged_basis_matrix(signal, lags, basis), raw @ basis)


def test_event_times_to_counts_uses_common_timebase():
    time = np.arange(0.0, 1.0, 0.1)
    counts = event_times_to_counts([0.02, 0.08, 0.51], time)
    assert counts.sum() == 3
    assert counts[0] == 1
    assert counts[1] == 1
    assert counts[5] == 1


def test_grouped_ridge_cv_recovers_predictive_model():
    rng = np.random.default_rng(4)
    design = np.column_stack([np.ones(120), rng.normal(size=(120, 2))])
    truth = np.array([0.5, 1.5, -0.75])
    response = design @ truth + rng.normal(scale=0.05, size=120)
    groups = np.repeat(["m1", "m2", "m3"], 40)
    result = fit_grouped_ridge_cv(
        design,
        response,
        groups,
        np.array([0.0, 0.1, 1.0]),
        penalize=np.array([0.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(result.coefficients, truth, atol=0.05)
    assert np.nanmax(result.mean_scores) > 0.98


def test_blocked_folds_apply_temporal_gap():
    folds = blocked_folds(20, n_folds=4, gap_samples=2)
    for train, test in folds:
        assert np.all(np.abs(train[:, None] - test[None, :]) > 2)
