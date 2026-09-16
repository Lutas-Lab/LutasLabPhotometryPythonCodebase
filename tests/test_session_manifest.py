from pathlib import Path
import unittest

from src.session_manifest import (
    load_session_manifest,
    processed_session_path,
    resolve_session_channel,
)


class SessionManifestTests(unittest.TestCase):
    def test_manifest_preserves_dates_and_parses_runs(self):
        path = Path("tests/_manifest_sessions.csv")
        try:
            path.write_text(
                "mouse,date,run\nM1,001234,002\nM2,260102,1\n",
                encoding="utf-8",
            )
            sessions = load_session_manifest(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(
            sessions,
            [
                {"mouse": "M1", "date": "001234", "run": 2},
                {"mouse": "M2", "date": "260102", "run": 1},
            ],
        )

    def test_manifest_rejects_duplicate_sessions(self):
        path = Path("tests/_manifest_sessions.csv")
        try:
            path.write_text(
                "mouse,date,run\nM1,260101,1\nM1,260101,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_session_manifest(path)
        finally:
            path.unlink(missing_ok=True)

    def test_manifest_preserves_optional_analysis_columns(self):
        path = Path("tests/_manifest_sessions.csv")
        try:
            path.write_text(
                "mouse,date,run,group,condition\nM1,260101,1,control,rewarded\n",
                encoding="utf-8",
            )
            sessions = load_session_manifest(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(sessions[0]["group"], "control")
        self.assertEqual(sessions[0]["condition"], "rewarded")

    def test_processed_path_uses_session_directory_convention(self):
        path = processed_session_path(
            Path("data"), {"mouse": "M1", "date": "260101", "run": 2}
        )
        self.assertEqual(
            path,
            Path("data/M1/M1_260101/M1-260101-002-processed.npz"),
        )

    def test_channel_uses_manifest_value_with_channel_one_fallback(self):
        self.assertEqual(resolve_session_channel({"mouse": "M1", "channel": "2"}), 2)
        self.assertEqual(resolve_session_channel({"mouse": "M1"}), 1)
        self.assertEqual(
            resolve_session_channel({"mouse": "M1", "channel": "2"}, "1"),
            1,
        )
        with self.assertRaisesRegex(ValueError, "must be 1 or 2"):
            resolve_session_channel({"mouse": "M1", "channel": "3"})
