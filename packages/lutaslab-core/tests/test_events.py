import numpy as np

from lutaslab_core.events import find_lick_bouts, find_ttl_pulses


def test_ttl_detection_includes_boundary_pulses_and_filters_width():
    timestamps = np.arange(8, dtype=float) / 10
    signal = np.array([5, 5, 0, 0, 5, 5, 5, 0], dtype=float)
    pulses = find_ttl_pulses(signal, timestamps, threshold=1.5, min_width_seconds=0.25)
    np.testing.assert_array_equal(pulses.rising_indices, [4])
    np.testing.assert_array_equal(pulses.falling_indices, [7])
    np.testing.assert_allclose(pulses.durations, [0.3])


def test_ttl_detection_returns_typed_empty_arrays():
    pulses = find_ttl_pulses(np.zeros(5), np.arange(5, dtype=float))
    assert pulses.onset_times.dtype == float
    assert pulses.rising_indices.dtype == int


def test_ttl_detection_can_match_legacy_complete_internal_pulses_only():
    signal = np.array([5, 0, 0, 5, 0, 5], dtype=float)
    pulses = find_ttl_pulses(
        signal,
        np.arange(signal.size, dtype=float),
        include_boundary_pulses=False,
    )
    np.testing.assert_array_equal(pulses.rising_indices, [3])
    np.testing.assert_array_equal(pulses.falling_indices, [4])


def test_lick_bouts_groups_and_filters():
    bouts = find_lick_bouts(
        np.array([0.0, 0.2, 0.4, 2.0, 2.2]),
        max_interlick_gap_seconds=0.5,
        min_licks=3,
    )
    np.testing.assert_allclose(bouts.onset_times, [0.0])
    np.testing.assert_allclose(bouts.offset_times, [0.4])
    np.testing.assert_array_equal(bouts.lick_counts, [3])
