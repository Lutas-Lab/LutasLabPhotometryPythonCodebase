import unittest

import numpy as np

from src.trial_classification import (
    classify_cue_licking,
    classify_session_cue_licking,
)


class TrialClassificationTests(unittest.TestCase):
    def test_post_cue_window_changes_trial_class_without_preprocessing(self):
        session = {
            "cue_onset": np.array([0.0, 10.0]),
            "cue_offset": np.array([1.0, 11.0]),
            "lick_times": np.array([0.5, 2.5, 10.5]),
        }

        short = classify_session_cue_licking(session, post_cue_window=1.0)
        long = classify_session_cue_licking(session, post_cue_window=2.0)

        np.testing.assert_array_equal(short["cue_only"], [True, True])
        np.testing.assert_array_equal(long["cue_and_post"], [True, False])
        np.testing.assert_array_equal(long["cue_only"], [False, True])

    def test_rejects_invalid_timestamps_and_window(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            classify_cue_licking([0.0], [1.0, 2.0], [], 2.0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            classify_cue_licking([0.0], [1.0], [], -1.0)
        with self.assertRaisesRegex(ValueError, "saved timestamps"):
            classify_session_cue_licking({"cue_onset": [0.0]}, 2.0)


if __name__ == "__main__":
    unittest.main()
