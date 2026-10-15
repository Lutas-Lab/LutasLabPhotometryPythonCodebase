import numpy as np
from iflip3.synchronization import external_marker_mask, fit_clock_alignment


def test_external_marker_one_is_matlab_bit_three():
    marks = np.array([0, 1, 2, 4, 8, 12], dtype=np.uint32)
    np.testing.assert_array_equal(
        external_marker_mask(marks, marker=1),
        [False, False, False, True, False, True],
    )


def test_clock_alignment_recovers_offset_and_drift():
    iflip = np.arange(0.15, 10.0, 0.2)
    truth_scale = 1.000009
    truth_intercept = -0.24
    nidaq = truth_intercept + truth_scale * iflip
    alignment = fit_clock_alignment(iflip, nidaq)
    assert alignment.matched_pulses == iflip.size
    np.testing.assert_allclose(alignment.scale, truth_scale)
    np.testing.assert_allclose(alignment.intercept_seconds, truth_intercept)
    np.testing.assert_allclose(alignment.nidaq_to_iflip(nidaq), iflip)
    assert alignment.max_residual_seconds < 1e-12


def test_iflip_alignment_tolerates_100_300_ms_marker_quantization():
    iflip = np.arange(30, dtype=float) * 0.2 + 0.05
    iflip[8:12] += 0.1
    nidaq = np.arange(30, dtype=float) * 0.2 + 0.4

    alignment = fit_clock_alignment(iflip, nidaq)

    assert alignment.matched_pulses == 30
    assert alignment.unmatched_iflip_marker_indices.size == 0
    assert alignment.unmatched_nidaq_pulse_indices.size == 0
    assert alignment.max_residual_seconds < 0.1
