import numpy as np

from iflip3.preprocessing import bin_curves, correct_lifetime_data, mpet_from_corrected


def test_mpet_known_curve():
    data = np.array([[0.0, 1.0], [1.0, 1.0], [3.0, 2.0]])
    result = mpet_from_corrected(data, np.array([0.0, 1.0, 2.0]), (0.0, 2.0), 0.5)
    np.testing.assert_allclose(result, [1.25, 0.75])


def test_zero_background_leaves_data_unchanged():
    data = np.arange(12, dtype=float).reshape(3, 4)
    result = correct_lifetime_data(
        data,
        None,
        dead_time_seconds=25e-9,
        sampling_frequency=10.0,
        afterpulse_ratio=0.0,
    )
    np.testing.assert_allclose(result.corrected, data)


def test_bin_curves_sums_complete_groups():
    data = np.arange(14).reshape(2, 7)
    result = bin_curves(data, 3)
    np.testing.assert_array_equal(result, [[3, 12], [24, 33]])
