import numpy as np

from src.figure5_bout_analysis import extract_bout_features


def test_extract_bout_features_splits_and_filters_bouts():
    events = np.zeros(30)
    events[[1, 3, 5, 20, 21]] = 1
    features = extract_bout_features(
        events,
        sample_rate_hz=10,
        max_interlick_gap_seconds=0.5,
        min_licks=3,
    )
    assert features.bout_count == 1
    assert features.onset_events[1] == 1
    assert features.size_events[1] == 3
    np.testing.assert_array_equal(np.flatnonzero(features.occupancy), np.arange(1, 6))
    assert np.all(features.rate[1:6] > 0)


def test_extract_bout_features_handles_no_licks():
    features = extract_bout_features(np.zeros(12), sample_rate_hz=50)
    assert features.bout_count == 0
    assert not np.any(features.onset_events)
