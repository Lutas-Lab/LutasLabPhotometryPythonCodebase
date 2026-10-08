import numpy as np
import pytest
from iflip3.fitting import fit_decay, fit_global
from iflip3.models import design_matrix, periodic_exgaussian_basis


def test_single_decay_recovers_lifetime():
    time = np.arange(126) * 0.1
    truth = dict(tau=2.2, t0=1.0, sigma=0.15, amplitude=5000.0, background=2.0)
    curve = (
        truth["amplitude"]
        * periodic_exgaussian_basis(time, truth["tau"], truth["t0"], truth["sigma"], 12.5)
        + truth["background"]
    )
    result = fit_decay(
        time,
        curve,
        n_components=1,
        pulse_interval=12.5,
        initial={"tau1": 1.8, "t0": 0.9, "sigma": 0.13},
        weighting="none",
    )
    assert result.success
    np.testing.assert_allclose(result.lifetimes, [truth["tau"]], rtol=2e-3)
    np.testing.assert_allclose(result.t0, truth["t0"], atol=2e-3)


def test_single_decay_honors_custom_bounds_and_fixed_parameters():
    time = np.arange(126) * 0.1
    curve = (
        2500.0 * periodic_exgaussian_basis(time, 0.8, 1.0, 0.14, 12.5)
        + 5000.0 * periodic_exgaussian_basis(time, 2.8, 1.0, 0.14, 12.5)
        + 2.0
    )
    result = fit_decay(
        time,
        curve,
        n_components=2,
        pulse_interval=12.5,
        initial={"tau1": 0.7, "tau2": 2.4, "t0": 1.0, "sigma": 0.14},
        fixed={"t0": 1.0, "sigma": 0.14},
        bounds={"tau1": (0.6, 1.0), "tau2": (2.0, 3.2)},
        weighting="none",
    )
    assert result.success
    np.testing.assert_allclose(result.lifetimes, [0.8, 2.8], atol=0.02)
    assert result.t0 == pytest.approx(1.0)
    assert result.irf_sigma == pytest.approx(0.14)


def test_single_decay_rejects_invalid_custom_bounds():
    time = np.arange(20, dtype=float) * 0.1
    with pytest.raises(ValueError, match="Lower bound"):
        fit_decay(
            time,
            np.ones_like(time),
            bounds={"tau1": (2.0, 1.0)},
            weighting="none",
        )


def test_global_fit_recovers_shared_lifetimes():
    rng = np.random.default_rng(4)
    time = np.arange(126) * 0.1
    lifetimes = np.array([0.7, 2.6])
    basis = design_matrix(time, lifetimes, 1.0, 0.14, 12.5, normalize="area")
    coefficients = np.column_stack([rng.uniform(500, 1500, 18), rng.uniform(1500, 3500, 18)])
    curves = rng.poisson(np.maximum(basis @ coefficients.T, 0.0))
    result = fit_global(
        time,
        curves,
        initial_lifetimes=[0.6, 2.3],
        initial_t0=1.0,
        initial_sigma=0.14,
        shared_t0=True,
        shared_sigma=True,
        weighting="poisson",
        max_nfev=80,
    )
    assert result.success
    np.testing.assert_allclose(result.lifetimes, lifetimes, atol=0.35)
    assert result.recordings[0].component_fractions.shape == (18, 2)
