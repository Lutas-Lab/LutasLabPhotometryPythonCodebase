import numpy as np

from lutaslab_photometry.pynapple_utils import session_to_core


def test_session_to_core_preserves_photometry_and_behavior():
    session = {
        "photo_time_465_ch1": np.array([0.0, 0.1]),
        "dff_ch1": np.array([1.0, 2.0]),
        "photo_time_465_ch2": np.array([0.0, 0.1]),
        "dff_ch2": np.array([3.0, 4.0]),
        "locomotion_time": np.array([0.0, 0.1]),
        "processed_locomotion": np.array([0.0, 1.0]),
        "lick_times": np.array([0.05]),
        "cue_onset": np.array([0.02]),
        "cue_offset": np.array([0.04]),
        "solenoid_onset": np.array([0.06]),
        "solenoid_offset": np.array([0.07]),
        "lick_bout_onset": np.array([0.05]),
        "lick_bout_offset": np.array([0.09]),
        "metadata": {"mouse": "M1"},
    }
    common = session_to_core(session, "M1-260101-001")
    assert common.session_id == "M1-260101-001"
    assert set(common.continuous) == {"dff_ch1", "dff_ch2", "locomotion"}
    assert set(common.events) == {"licks"}
    assert set(common.intervals) == {"visual_cues", "solenoid", "lick_bouts"}
    assert common.metadata["sensor"] == "fiber_photometry"


def test_session_to_core_omits_running_that_failed_quality_control():
    session = {
        "photo_time_465_ch1": np.array([0.0, 0.1]),
        "dff_ch1": np.array([1.0, 2.0]),
        "photo_time_465_ch2": np.array([0.0, 0.1]),
        "dff_ch2": np.array([3.0, 4.0]),
        "locomotion_time": np.array([], dtype=float),
        "processed_locomotion": np.array([], dtype=float),
        "lick_times": np.array([0.05]),
        "cue_onset": np.array([0.02]),
        "cue_offset": np.array([0.04]),
        "solenoid_onset": np.array([0.06]),
        "solenoid_offset": np.array([0.07]),
        "lick_bout_onset": np.array([0.05]),
        "lick_bout_offset": np.array([0.09]),
        "running_data_valid": False,
        "running_data_status": "invalid_flat_zero",
    }

    common = session_to_core(session)

    assert "locomotion" not in common.continuous
    assert common.metadata["running_data_valid"] is False
    assert common.metadata["running_data_status"] == "invalid_flat_zero"
