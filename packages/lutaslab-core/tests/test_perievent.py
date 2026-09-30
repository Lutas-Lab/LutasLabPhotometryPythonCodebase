import numpy as np

from lutaslab_core.perievent import (
    extract_perievent_event_rate,
    extract_perievent_trials,
    generate_null_onsets,
    normalize_trials,
    summarize_trials,
)


def test_extract_continuous_trials_tracks_original_event_indices():
    time = np.arange(0.0, 10.1, 0.1)
    relative, trials, valid = extract_perievent_trials(
        time,
        time,
        [0.5, 5.0, 9.8],
        window=(-1.0, 1.0),
        dt=0.1,
    )
    np.testing.assert_array_equal(valid, [1])
    np.testing.assert_allclose(trials[0], 5.0 + relative)


def test_extract_event_rate_can_filter_incomplete_windows():
    relative, trials, valid = extract_perievent_event_rate(
        [9.25, 10.25, 10.75, 19.0],
        [0.5, 10.0, 19.5],
        (0.0, 20.0),
        window=(-1.0, 1.0),
        dt=0.5,
    )
    np.testing.assert_allclose(relative, [-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_array_equal(valid, [1])
    np.testing.assert_allclose(trials[0], [2.0, 0.0, 2.0, 2.0])


def test_extract_continuous_trials_can_nan_pad_recording_edges():
    time = np.arange(0.0, 2.1, 0.1)
    relative, trials, valid = extract_perievent_trials(
        time,
        2 * time,
        [0.2, 1.0],
        window=(-0.5, 0.5),
        dt=0.1,
        require_complete=False,
    )
    np.testing.assert_array_equal(valid, [0, 1])
    assert np.isnan(trials[0, 0])
    np.testing.assert_allclose(trials[1], 2 * (1.0 + relative))


def test_normalize_and_summarize_trials():
    time = np.array([-1.0, -0.5, 0.0, 0.5])
    trials = np.array([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]])
    normalized = normalize_trials(time, trials, "subtract", (-1.0, 0.0))
    np.testing.assert_allclose(normalized[:, :2].mean(axis=1), 0.0)
    mean, sem = summarize_trials(normalized)
    assert mean.shape == sem.shape == time.shape


def test_null_onsets_are_reproducible():
    first = generate_null_onsets(
        [4.0, 8.0],
        (1.0, 11.0),
        n_shuffles=10,
        exclusion=0.5,
        rng=np.random.default_rng(12),
    )
    second = generate_null_onsets(
        [4.0, 8.0],
        (1.0, 11.0),
        n_shuffles=10,
        exclusion=0.5,
        rng=np.random.default_rng(12),
    )
    np.testing.assert_allclose(first, second)
