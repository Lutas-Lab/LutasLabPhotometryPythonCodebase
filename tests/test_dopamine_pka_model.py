import numpy as np

from src.dopamine_pka_model import (
    causal_biexponential_kernel,
    fit_dopamine_to_pka,
    predict_pka_from_dopamine,
)


def test_causal_biexponential_kernel_is_nonnegative_and_unit_area():
    time = np.arange(0.0, 30.0, 0.02)
    kernel = causal_biexponential_kernel(time, 0.4, 4.0, 0.2)
    assert np.all(kernel >= 0)
    np.testing.assert_allclose(kernel.sum() * 0.02, 1.0)
    assert np.all(kernel[time < 0.2] == 0)


def test_fit_dopamine_to_pka_recovers_synthetic_timecourse():
    time = np.arange(-5.0, 25.0, 0.05)
    dopamine = np.exp(-0.5 * ((time - 3.0) / 1.2) ** 2)
    pka = predict_pka_from_dopamine(
        time,
        dopamine,
        rise_tau_seconds=0.6,
        decay_tau_seconds=4.5,
        delay_seconds=0.35,
        gain=1.8,
        baseline=0.15,
    )
    fit = fit_dopamine_to_pka(time, dopamine, pka)
    assert fit.success
    assert fit.r2 > 0.999
    assert fit.rise_tau_seconds < fit.decay_tau_seconds
    np.testing.assert_allclose(fit.prediction, pka, atol=1e-3)
