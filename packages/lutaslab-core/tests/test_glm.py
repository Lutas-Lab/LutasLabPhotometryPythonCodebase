import numpy as np

from lutaslab_core.glm import (
    apply_lag_basis,
    blocked_folds,
    build_trialwise_basis_design,
    convolve_basis,
    event_times_to_counts,
    fit_grouped_ridge_cv,
    fit_ridge,
    fit_ridge_path,
    lagged_basis_matrix,
    lagged_signal_matrix,
    predict_lagged_signal,
    raised_cosine_basis,
    reconstruct_kernel,
    zero_split_raised_cosine_basis,
)


def test_basis_convolution_and_kernel_reconstruction():
    basis = raised_cosine_basis((0.0, 1.0), count=3, dt=0.1)
    events = np.zeros(30)
    events[5] = 1
    design = convolve_basis(events, basis.values)
    assert design.shape == (30, 3)
    coefficients = np.array([1.0, -0.5, 0.25])
    np.testing.assert_allclose(
        reconstruct_kernel(basis.values, coefficients), basis.values @ coefficients
    )


def test_zero_split_basis_has_no_columns_crossing_time_zero():
    basis = zero_split_raised_cosine_basis((-5.0, 20.0), count=14, dt=0.1)
    negative = basis.lag_times < 0
    nonnegative = ~negative
    support_before = np.any(basis.values[negative] != 0, axis=0)
    support_after = np.any(basis.values[nonnegative] != 0, axis=0)

    assert basis.values.shape[1] == 14
    assert np.any(support_before)
    assert np.any(support_after)
    assert not np.any(support_before & support_after)
    np.testing.assert_array_equal(basis.values[nonnegative, :3], 0.0)
    np.testing.assert_array_equal(basis.values[negative, 3:], 0.0)
    np.testing.assert_array_equal(basis.values[basis.lag_times == 0], 0.0)


def test_zero_split_basis_delays_one_sided_peak_beyond_onset():
    basis = zero_split_raised_cosine_basis((0.0, 8.0), count=8, dt=0.1)
    onset = np.flatnonzero(basis.lag_times == 0)

    np.testing.assert_array_equal(basis.values[onset], 0.0)
    first_peak = basis.lag_times[np.argmax(basis.values[:, 0])]
    assert 0 < first_peak < 0.5


def test_zero_split_basis_is_denser_early_than_late():
    basis = zero_split_raised_cosine_basis((0.0, 20.0), count=11, dt=0.01)
    peaks = np.asarray(
        [basis.lag_times[np.argmax(column)] for column in basis.values.T]
    )

    assert np.all(np.diff(peaks) > 0)
    assert peaks[0] < 0.5
    assert np.diff(peaks)[-1] > 5 * np.diff(peaks)[0]


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


def test_ridge_path_matches_independent_ridge_fits():
    rng = np.random.default_rng(17)
    design = np.column_stack([np.ones(80), rng.normal(size=(80, 4))])
    response = rng.normal(size=80)
    alphas = np.array([0.0, 0.01, 1.0, 100.0])
    penalty = np.array([0.0, 1.0, 1.0, 1.0, 1.0])

    expected = np.vstack(
        [fit_ridge(design, response, alpha, penalize=penalty) for alpha in alphas]
    )
    actual = fit_ridge_path(design, response, alphas, penalize=penalty)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_blocked_folds_apply_temporal_gap():
    folds = blocked_folds(20, n_folds=4, gap_samples=2)
    for train, test in folds:
        assert np.all(np.abs(train[:, None] - test[None, :]) > 2)


def test_trialwise_design_resets_convolution_between_trials():
    basis = raised_cosine_basis((0.0, 0.4), count=3, dt=0.1)
    events = np.zeros((2, 10))
    events[0, -1] = 1.0
    events[1, 2] = 1.0
    design = build_trialwise_basis_design(
        {"reward": events},
        {"reward": basis},
        0.1,
        trial_covariates={"trained": np.array([0.0, 1.0])},
    )
    assert design.matrix.shape == (20, 5)
    np.testing.assert_array_equal(design.matrix[:10, 0], 1.0)
    np.testing.assert_array_equal(design.matrix[:10, 1], 0.0)
    np.testing.assert_array_equal(design.matrix[10:, 1], 1.0)
    np.testing.assert_array_equal(
        design.matrix[10, design.column_slices["reward"]], 0.0
    )
