import numpy as np
import pytest

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


def test_robust_alignment_accepts_balanced_quantization_jitter():
    source = np.arange(20, dtype=float) * 0.2
    source[5:8] += 0.1
    target = 0.25 + np.arange(20, dtype=float) * 0.2

    alignment = fit_clock_alignment(
        source,
        target,
        allow_unmatched_pulses=True,
        interval_jitter_fraction=0.5,
        max_rms_residual_fraction=0.35,
        max_residual_fraction=0.6,
    )

    assert alignment.matched_pulses == 20
    assert alignment.unmatched_source_indices.size == 0
    assert alignment.unmatched_target_indices.size == 0
    assert alignment.max_residual_seconds < 0.1


def test_robust_alignment_skips_one_missing_source_pulse():
    complete = np.arange(20, dtype=float) * 0.2
    source = np.delete(complete, 8)
    target = 0.25 + complete

    alignment = fit_clock_alignment(
        source,
        target,
        allow_unmatched_pulses=True,
        interval_jitter_fraction=0.5,
    )

    assert alignment.matched_pulses == 19
    np.testing.assert_array_equal(alignment.unmatched_source_indices, [])
    np.testing.assert_array_equal(alignment.unmatched_target_indices, [8])
    np.testing.assert_allclose(alignment.source_to_target(source), 0.25 + source)


def test_robust_alignment_skips_duplicate_source_pulse():
    complete = np.arange(20, dtype=float) * 0.2
    source = np.insert(complete, 9, complete[8] + 0.01)
    target = 0.25 + complete

    alignment = fit_clock_alignment(
        source,
        target,
        allow_unmatched_pulses=True,
        interval_jitter_fraction=0.5,
    )

    assert alignment.matched_pulses == 20
    np.testing.assert_array_equal(alignment.unmatched_source_indices, [9])
    assert alignment.unmatched_target_indices.size == 0


def test_robust_alignment_rejects_sustained_interval_corruption():
    source = np.arange(20, dtype=float) * 0.25
    target = np.arange(20, dtype=float) * 0.2

    with pytest.raises(ValueError, match="drift is implausible"):
        fit_clock_alignment(
            source,
            target,
            allow_unmatched_pulses=True,
            interval_jitter_fraction=0.5,
            max_abs_drift_ppm=5000,
        )
