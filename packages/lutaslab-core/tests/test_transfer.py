import numpy as np
import pytest

from lutaslab_core.transfer import (
    apply_gamma_transfer,
    fit_nonnegative_gamma_transfer,
    gamma_cascade_kernel,
)


def test_gamma_kernel_has_unit_area_on_fine_grid():
    lag = np.arange(0.0, 100.0, 0.01)
    kernel = gamma_cascade_kernel(lag, 4.0, order=2, delay_seconds=0.5)
    np.testing.assert_allclose(kernel.sum() * 0.01, 1.0, rtol=1e-4)
    assert np.all(kernel[lag < 0.5] == 0.0)


def test_nonnegative_gamma_grid_recovers_synthetic_transfer():
    time = np.arange(-5.0, 30.0, 0.05)
    drive = np.zeros(time.size)
    drive[(time >= 0.0) & (time < 1.0)] = 1.0
    response = 2.5 * apply_gamma_transfer(drive, 0.05, 5.0)
    fit = fit_nonnegative_gamma_transfer(
        time,
        drive,
        response,
        [2.0, 5.0, 10.0],
        fit_window=(0.0, 25.0),
        baseline_window=(-5.0, 0.0),
    )
    assert fit.tau_seconds == 5.0
    np.testing.assert_allclose(fit.gain, 2.5)
    assert fit.r2 > 0.999


def test_nonnegative_gamma_grid_rejects_fit_window_outside_time():
    time = np.arange(10.0)
    with pytest.raises(ValueError, match="fit_window does not overlap time"):
        fit_nonnegative_gamma_transfer(
            time,
            np.ones(time.size),
            np.ones(time.size),
            [1.0],
            fit_window=(20.0, 30.0),
        )
