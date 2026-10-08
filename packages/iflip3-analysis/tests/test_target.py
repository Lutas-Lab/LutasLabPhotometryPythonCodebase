import numpy as np
from iflip3.models import design_matrix
from iflip3.target import fit_target, lifetime_window


def test_lifetime_window_excludes_edge_artifacts():
    time = np.arange(126) * 0.1
    mask = lifetime_window(time, (0.4, 12.3))
    assert time[mask][0] == 0.4
    assert np.isclose(time[mask][-1], 12.3)
    assert mask.sum() == 120


def test_target_analysis_recovers_component_counts_with_fixed_background():
    time = np.arange(4, 124) * 0.1
    lifetimes = np.array([0.5, 1.8])
    basis = design_matrix(
        time,
        lifetimes,
        t0=1.0,
        irf_sigma=0.15,
        pulse_interval=12.5,
        normalize="area",
    )
    truth = np.array([[1_000.0, 3_000.0], [2_500.0, 1_500.0]])
    background = 12.0
    curves = basis @ truth.T + background

    result = fit_target(
        time,
        curves,
        lifetimes=lifetimes,
        t0=1.0,
        irf_sigma=0.15,
        residual_background_per_bin=background,
    )

    np.testing.assert_allclose(result.component_counts, truth, rtol=1e-10)
    np.testing.assert_allclose(result.fitted, curves, rtol=1e-10)
    np.testing.assert_allclose(result.component_fractions.sum(axis=1), 1.0)


def test_target_poisson_weighting_rejects_negative_curves():
    time = np.arange(4, 124) * 0.1
    curves = np.ones((time.size, 2))
    curves[-1, 0] = -1
    try:
        fit_target(
            time,
            curves,
            lifetimes=[0.5, 1.8],
            t0=1.0,
            irf_sigma=0.15,
            weighting="poisson",
        )
    except ValueError as error:
        assert "negative" in str(error)
    else:
        raise AssertionError("Expected negative Poisson input to be rejected")
