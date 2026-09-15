import numpy as np
import unittest

from src.preprocess import (
    align_reference_to_experimental,
    classify_cue_licking,
    find_ttl_pulses,
    irls_dff,
    preprocess_locomotion,
    preprocess_photometry,
    preprocess_session,
)


class PreprocessTests(unittest.TestCase):
    def test_empty_ttl_edges_are_integer_arrays(self):
        rising, falling = find_ttl_pulses(np.zeros(20))
        self.assertEqual(rising.dtype.kind, "i")
        self.assertEqual(falling.dtype.kind, "i")
        self.assertEqual(rising.size, 0)
        self.assertEqual(falling.size, 0)

    def test_short_photometry_pulse_is_rejected(self):
        ttl = np.array([0, 0, 5, 5, 5, 5, 0, 0], dtype=float)
        with self.assertRaisesRegex(ValueError, "too short"):
            preprocess_photometry(
                np.arange(8, dtype=float),
                ttl,
                np.zeros(8),
                np.arange(8, dtype=float),
                edge=3,
            )

    def test_missing_locomotion_ttl_is_reported(self):
        with self.assertRaisesRegex(ValueError, "No valid locomotion TTL"):
            preprocess_locomotion(np.ones(4), np.zeros(20), np.arange(20, dtype=float))

    def test_large_locomotion_mismatch_is_rejected(self):
        ttl = np.tile([0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0], 5)
        with self.assertRaisesRegex(ValueError, "exceeds 1%"):
            preprocess_locomotion(np.ones(10), ttl, np.arange(len(ttl), dtype=float))

    def test_alignment_rejects_material_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "beyond the reference"):
            align_reference_to_experimental(
                np.array([1.0, 2.0, 3.0]), np.ones(3), np.array([-1.0, 1.0, 2.0])
            )

    def test_irls_rejects_zero_fitted_reference(self):
        with self.assertRaisesRegex(ValueError, "near-zero"):
            irls_dff(np.zeros(10), np.zeros(10))

    def test_cue_classification_uses_actual_offset(self):
        result = classify_cue_licking(
            cue_onset=[0.0, 10.0], cue_offset=[5.0, 18.0], lick_times=[4.0, 19.0]
        )
        cue_lick, post_cue_lick, cue_only, post_only, both, miss = result
        np.testing.assert_array_equal(cue_lick, [True, False])
        np.testing.assert_array_equal(post_cue_lick, [False, True])
        np.testing.assert_array_equal(cue_only, [True, False])
        np.testing.assert_array_equal(post_only, [False, True])
        self.assertFalse(np.any(both))
        self.assertFalse(np.any(miss))

    def test_synthetic_session_runs_end_to_end(self):
        n_samples = 500
        timestamps = np.arange(n_samples, dtype=float) / 100.0
        ttl_465 = np.zeros(n_samples)
        ttl_405 = np.zeros(n_samples)
        for start in range(10, 480, 20):
            ttl_465[start : start + 8] = 5.0
        for start in range(20, 490, 20):
            ttl_405[start : start + 8] = 5.0

        locomotion_ttl = np.zeros(n_samples)
        locomotion_starts = list(range(5, 495, 10))
        for start in locomotion_starts:
            locomotion_ttl[start : start + 5] = 5.0

        visual_cue = np.zeros(n_samples)
        visual_cue[100:108] = 5.0
        visual_cue[120:128] = 5.0
        licking = np.zeros(n_samples)
        licking[150:155] = 5.0
        solenoid = np.zeros(n_samples)
        solenoid[200:208] = 5.0

        base = 2.0 + 0.001 * np.arange(n_samples)
        session = {
            "mouse": "TEST",
            "date": "260101",
            "run": 1,
            "photometry_path": "synthetic.mat",
            "raw_photometry_ch1": base,
            "raw_photometry_ch2": 1.2 * base,
            "visual_cue": visual_cue,
            "licking": licking,
            "solenoid_opening": solenoid,
            "ttl_465": ttl_465,
            "ttl_405": ttl_405,
            "timestamps": timestamps,
            "fs": 100.0,
            "locomotion_ttlpulses": locomotion_ttl,
            "locomotion": np.linspace(0.0, 1.0, len(locomotion_starts)),
        }

        processed = preprocess_session(session)
        self.assertEqual(processed["processed_schema_version"], "1.0")
        self.assertEqual(processed["n_465_pulses"], len(range(10, 480, 20)))
        self.assertTrue(np.all(np.isfinite(processed["dff_ch1"])))
