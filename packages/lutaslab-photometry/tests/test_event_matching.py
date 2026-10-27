import numpy as np

from lutaslab_photometry.event_matching import first_events_after_references


def test_first_events_after_references_matches_once_without_crossing_next_reference():
    matched, reference_indices = first_events_after_references(
        np.array([10.0, 15.0, 30.0]),
        np.array([9.0, 12.0, 16.0, 18.0, 55.0]),
        (0.0, 20.0),
    )

    np.testing.assert_allclose(matched, [12.0, 16.0])
    np.testing.assert_array_equal(reference_indices, [0, 1])


def test_first_events_after_references_honors_lower_latency_bound():
    matched, _ = first_events_after_references(
        np.array([10.0]), np.array([10.2, 11.5]), (1.0, 3.0)
    )
    np.testing.assert_allclose(matched, [11.5])
