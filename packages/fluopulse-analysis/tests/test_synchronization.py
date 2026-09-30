import numpy as np

from fluopulse_analysis.synchronization import fit_clock_alignment


def test_clock_alignment_recovers_offset_and_drift():
    source = np.arange(0.01, 20, 1 / 30)
    target = -0.35 + 1.000012 * source
    alignment = fit_clock_alignment(source, target)
    np.testing.assert_allclose(alignment.scale, 1.000012)
    np.testing.assert_allclose(alignment.intercept_seconds, -0.35)
    assert alignment.rms_residual_seconds < 1e-12
