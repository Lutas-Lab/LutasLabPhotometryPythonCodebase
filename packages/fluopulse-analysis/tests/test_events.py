import numpy as np
from fluopulse_analysis.events import (
    extract_perievent,
    find_lick_bouts,
    find_ttl_pulses,
)


def test_find_ttl_pulses_and_width_filter():
    time = np.arange(12) / 10
    signal = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], dtype=float)
    pulses = find_ttl_pulses(
        signal,
        time,
        threshold=0.5,
        min_width_seconds=0.15,
        max_width_seconds=0.25,
    )
    np.testing.assert_allclose(pulses.onset_times, [0.1])
    np.testing.assert_allclose(pulses.durations, [0.2])


def test_find_ttl_pulses_handles_constant_low_signal():
    pulses = find_ttl_pulses(np.zeros(20), np.arange(20) / 1000, threshold=0.5)
    assert pulses.onset_times.size == 0


def test_find_lick_bouts_filters_short_groups():
    licks = np.array([1.0, 1.2, 1.4, 4.0, 4.1, 7.0])
    bouts = find_lick_bouts(
        licks,
        max_interlick_gap_seconds=0.5,
        min_licks=2,
    )
    np.testing.assert_allclose(bouts.onset_times, [1.0, 4.0])
    np.testing.assert_allclose(bouts.offset_times, [1.4, 4.1])
    np.testing.assert_array_equal(bouts.lick_counts, [3, 2])


def test_extract_perievent_interpolates_and_nan_pads_edges():
    time = np.arange(0.0, 10.1, 0.1)
    relative_time, trials = extract_perievent(
        time,
        2.0 * time,
        np.array([0.2, 5.0]),
        window_seconds=(-1.0, 1.0),
        sample_interval_seconds=0.1,
    )
    assert trials.shape == (2, relative_time.size)
    assert np.isnan(trials[0, 0])
    np.testing.assert_allclose(trials[1], 2.0 * (5.0 + relative_time))
