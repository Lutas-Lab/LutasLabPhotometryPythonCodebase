import numpy as np

from lutaslab_core.synchronization import fit_clock_alignment


def test_clock_alignment_round_trip_and_diagnostics():
    source = np.arange(20, dtype=float) / 5
    target = 0.25 + 1.00002 * source
    alignment = fit_clock_alignment(source, target)
    np.testing.assert_allclose(alignment.source_to_target(source), target)
    np.testing.assert_allclose(alignment.target_to_source(target), source, atol=1e-12)
    assert alignment.matched_pulses == source.size
    assert np.isclose(alignment.drift_ppm, 20)
    assert alignment.rms_residual_seconds < 1e-12
