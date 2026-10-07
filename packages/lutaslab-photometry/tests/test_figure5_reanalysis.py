from pathlib import Path

import numpy as np

from lutaslab_photometry.figure5_reanalysis import (
    contiguous_trial_blocks,
    fit_nested_blocked_ridge,
    load_raw_figure5_trials,
)


def test_contiguous_trial_blocks_preserve_order():
    np.testing.assert_array_equal(
        contiguous_trial_blocks(10, 3),
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
    )


def test_nested_blocked_ridge_returns_out_of_fold_predictions():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(120, 2))
    matrix = np.column_stack([np.ones(120), x])
    response = 0.25 + x @ np.array([1.2, -0.8]) + rng.normal(scale=0.05, size=120)
    groups = np.repeat(np.arange(5), 24)
    result = fit_nested_blocked_ridge(
        matrix,
        response,
        groups,
        np.array([0.01, 0.1, 1.0]),
        penalty_weights=np.array([0.0, 1.0, 1.0]),
    )
    assert np.all(np.isfinite(result.prediction))
    assert result.heldout_r2 > 0.98
    assert result.fold_r2.shape == (5,)
    assert len(result.outer_coefficients) == 5


def test_load_raw_figure5_trials_uses_manifest_channel_and_fixed_cue_window(
    monkeypatch,
):
    expected_path = Path("synthetic-processed.npz")
    manifest_session = {"mouse": "M1", "date": "250101", "run": 2, "channel": "2"}

    time = np.arange(0.0, 80.0, 0.02)
    channel_1 = np.sin(time)
    channel_2 = np.cos(time / 3.0) + 0.01 * time
    session = {
        "photo_time_465_ch1": time,
        "photometry_465_ch1": channel_1,
        "dff_ch1": channel_1,
        "photo_time_465_ch2": time,
        "photometry_465_ch2": channel_2,
        "dff_ch2": channel_2,
        "cue_onset": np.array([10.0, 40.0]),
        # The first trial's detected cue is truncated, but selection must still
        # use the programmed 8--10 second interval from cue onset.
        "cue_offset": np.array([16.0, 48.0]),
        "lick_times": np.array([17.0, 50.0004]),
        "solenoid_onset": np.array([18.1, 48.1]),
    }

    def fake_load_session(path):
        assert Path(path) == expected_path
        return session

    monkeypatch.setattr(
        "lutaslab_photometry.figure5_reanalysis.load_session_manifest", lambda _: [manifest_session]
    )
    monkeypatch.setattr(
        "lutaslab_photometry.figure5_reanalysis._raw_processed_session_path",
        lambda *_: expected_path,
    )
    monkeypatch.setattr("lutaslab_photometry.figure5_reanalysis.load_session", fake_load_session)
    trials = load_raw_figure5_trials("manifest.csv", "processed")

    # The 7-second lick does not qualify despite the truncated detected cue;
    # the 10.0004-second lick qualifies through half-bin boundary tolerance.
    np.testing.assert_array_equal(trials.trial_numbers, [2])
    assert trials.photometry.shape == (1, 1000)
    assert trials.sample_rate_hz == 50.0
    assert trials.cue_events[0, 250] == 1.0
    assert trials.ensure_events[0, 655] == 1.0
    assert trials.lick_events[0, 750] == 1.0

    baseline = channel_2[(time >= 35.0) & (time < 40.0)]
    expected_first = (
        channel_2[np.searchsorted(time, 35.0)] - baseline.mean()
    ) / baseline.std(ddof=1)
    np.testing.assert_allclose(trials.photometry[0, 0], expected_first)
