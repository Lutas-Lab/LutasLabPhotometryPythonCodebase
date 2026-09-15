import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
if SCIPY_AVAILABLE:
    from scipy.io import savemat
    from src.load_data import DEFAULT_CHANNEL_MAP, get_session_paths, load_session_data


@unittest.skipUnless(SCIPY_AVAILABLE, "SciPy is not installed in the local smoke-test runtime")
class LoadDataTests(unittest.TestCase):
    def test_paths_use_explicit_root(self):
        session = get_session_paths("M1", "260101", 2, Path("data-root"))
        self.assertEqual(
            session["photometry_path"],
            Path("data-root/M1/M1_260101/M1-260101-002-nidaq.mat"),
        )

    def test_mat_loader_records_configured_channel_map(self):
        photometry_path = Path("tests/synthetic-nidaq.mat")
        locomotion_path = Path("tests/synthetic-running.mat")
        try:
            savemat(
                photometry_path,
                {
                    "data": np.arange(80, dtype=float).reshape(8, 10),
                    "timestamps": np.arange(10, dtype=float),
                    "Fs": 100.0,
                },
            )
            savemat(locomotion_path, {"speed": np.arange(3, dtype=float)})
            loaded = load_session_data(
                {
                    "photometry_path": photometry_path,
                    "locomotion_path": locomotion_path,
                }
            )
        finally:
            photometry_path.unlink(missing_ok=True)
            locomotion_path.unlink(missing_ok=True)

        np.testing.assert_array_equal(
            loaded["channel_map_rows"], list(DEFAULT_CHANNEL_MAP.values())
        )
        np.testing.assert_array_equal(loaded["raw_photometry_ch1"], np.arange(10))
