import numpy as np

from src.figure5_reanalysis import contiguous_trial_blocks, fit_nested_blocked_ridge


def test_contiguous_trial_blocks_preserve_order():
    np.testing.assert_array_equal(
        contiguous_trial_blocks(10, 3),
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
    )


def test_nested_blocked_ridge_returns_out_of_fold_predictions():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(120, 2))
    matrix = np.column_stack([np.ones(120), x])
    response = 0.25 + x @ np.array([1.2, -0.8]) + rng.normal(scale=0.05, size=120)
    groups = np.repeat(np.arange(5), 24)
    result = fit_nested_blocked_ridge(
        matrix,
        response,
        groups,
        np.array([0.01, 0.1, 1.0]),
        penalty_weights=np.array([0.0, 1.0, 1.0]),
    )
    assert np.all(np.isfinite(result.prediction))
    assert result.heldout_r2 > 0.98
    assert result.fold_r2.shape == (5,)
    assert len(result.outer_coefficients) == 5
