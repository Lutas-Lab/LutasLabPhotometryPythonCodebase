import numpy as np

from lutaslab_photometry.heatmap_ordering import order_heatmap_trials


def session_result():
    return {
        "trials": np.asarray([[1, 2], [3, 4], [5, 6]], dtype=float),
        "alignment_times": np.asarray([10.0, 30.0, 50.0]),
        "behavioral_events": {
            "solenoid_onset": np.asarray([13.0, 31.0]),
            "lick_times": np.asarray([10.5, 11.0, 34.0, 34.5, 49.0]),
            "lick_bout_onset": np.asarray([10.5, 34.0, 55.0]),
            "lick_bout_duration": np.asarray([1.0, 2.0, 3.0]),
            "lick_bout_lick_count": np.asarray([3, 7, 5]),
            "cue_onset": np.asarray([8.0, 28.0, 45.0]),
        },
    }


def test_orders_ensure_latency_and_keeps_unmatched_at_bottom():
    ordered = order_heatmap_trials(
        session_result(),
        [-1.0, 1.0],
        sort="ensure_latency",
        sort_window=(0, 10),
    )
    np.testing.assert_array_equal(ordered["order"], [1, 0, 2])
    np.testing.assert_allclose(ordered["ordered_values"][:2], [1.0, 3.0])
    assert np.isnan(ordered["ordered_values"][-1])


def test_can_exclude_unmatched_behavioral_events():
    ordered = order_heatmap_trials(
        session_result(),
        [-1.0, 1.0],
        sort="ensure_latency",
        sort_window=(0, 10),
        unmatched="exclude",
    )
    np.testing.assert_array_equal(ordered["order"], [1, 0])
    assert ordered["trials"].shape == (2, 2)


def test_bout_size_uses_first_bout_in_matching_window():
    ordered = order_heatmap_trials(
        session_result(),
        [-1.0, 1.0],
        sort="bout_size",
        sort_window=(0, 10),
        direction="descending",
    )
    np.testing.assert_array_equal(ordered["order"], [1, 2, 0])
    np.testing.assert_allclose(ordered["ordered_values"], [7, 5, 3])


def test_cue_to_bout_latency_uses_most_recent_prior_cue():
    ordered = order_heatmap_trials(
        session_result(),
        [-1.0, 1.0],
        sort="cue_to_bout_latency",
        sort_window=(-10, 0),
    )
    np.testing.assert_allclose(ordered["values"], [2, 2, 5])
    np.testing.assert_array_equal(ordered["order"], [0, 1, 2])
