from pathlib import Path
import unittest

import numpy as np

from src.save_sessiondata import load_session, save_session


class SessionPersistenceTests(unittest.TestCase):
    def test_round_trip_records_schema_and_provenance(self):
        session = {
            "mouse": "TEST",
            "date": "260101",
            "run": 1,
            "processed_schema_version": "1.0",
            "photometry_path": "source.mat",
            "photo_time_465_ch1": np.arange(3, dtype=float),
            "photometry_465_ch1": np.ones(3),
            "dff_ch1": np.zeros(3),
            "locomotion_time": np.arange(3, dtype=float),
            "processed_locomotion": np.zeros(3),
        }
        path = Path("tests/TEST-260101-001-processed.npz")
        try:
            path = save_session(session, output_dir=path.parent)
            loaded = load_session(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(loaded["processed_schema_version"], "1.0")
        self.assertIn("processing_utc", loaded)
        self.assertIn("code_commit", loaded)
        np.testing.assert_array_equal(loaded["dff_ch1"], session["dff_ch1"])
