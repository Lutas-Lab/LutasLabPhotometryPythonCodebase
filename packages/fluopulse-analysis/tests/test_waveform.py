import numpy as np

from fluopulse_analysis.waveform import (
    _convolved_basis,
    fit_irf_convolved_double_exponential,
    fit_irf_convolved_single_exponential,
    subtract_terminal_baseline,
)


def test_terminal_baseline_subtraction():
    values = np.r_[np.ones(8) * 4, np.ones(2) * 2]
    corrected = subtract_terminal_baseline(values, fraction=0.2)
    np.testing.assert_allclose(corrected[-2:], 0)


def test_convolved_fit_recovers_synthetic_lifetime():
    time = np.arange(300) * 0.1
    irf = np.exp(-0.5 * ((time - 2.0) / 0.25) ** 2)
    truth_tau = 2.7
    waveform = 1.3 + 80 * _convolved_basis(time, irf, truth_tau, 0.3)
    result = fit_irf_convolved_single_exponential(
        time,
        waveform,
        irf,
        fit_window_ns=(0, 20),
    )
    assert result.success
    np.testing.assert_allclose(result.tau_ns, truth_tau, rtol=2e-3)
    assert result.r_square > 0.9999


def test_double_fit_recovers_synthetic_components():
    time = np.arange(400) * 0.1
    irf = np.exp(-0.5 * ((time - 2.0) / 0.25) ** 2)
    short, long, fraction = 0.6, 2.8, 0.35
    mixture = (1 - fraction) * _convolved_basis(time, irf, short, 0.1)
    mixture += fraction * _convolved_basis(time, irf, long, 0.1)
    waveform = 0.8 + 75 * mixture
    result = fit_irf_convolved_double_exponential(time, waveform, irf)
    assert result.success
    np.testing.assert_allclose(result.tau_short_ns, short, rtol=0.02)
    np.testing.assert_allclose(result.tau_long_ns, long, rtol=0.02)
    np.testing.assert_allclose(result.long_fraction, fraction, atol=0.02)
