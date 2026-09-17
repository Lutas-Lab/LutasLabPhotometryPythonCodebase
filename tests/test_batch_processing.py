import importlib.util
from pathlib import Path
import shutil
import unittest


SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
if SCIPY_AVAILABLE:
    from src.batch_processing import preprocess_manifest_sessions
    from src.session_manifest import processed_session_path


@unittest.skipUnless(SCIPY_AVAILABLE, "SciPy is not installed in the local smoke-test runtime")
class BatchProcessingTests(unittest.TestCase):
    def test_existing_processed_session_is_not_overwritten_by_default(self):
        data_root = Path("tests/_batch_processing_data")
        info = {"mouse": "M1", "date": "260101", "run": 1}
        path = processed_session_path(data_root, info)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existing result")

            results = preprocess_manifest_sessions([info], data_root)

            self.assertEqual(results[0]["status"], "skipped")
            self.assertEqual(path.read_bytes(), b"existing result")
        finally:
            shutil.rmtree(data_root, ignore_errors=True)
