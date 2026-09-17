from pathlib import Path
import importlib.util
import shutil
import unittest

import numpy as np

from src.lickbout_delivery import (
    compute_lickbout_delivery_psth,
    match_cue_lickbout_delivery_trials,
    save_delivery_sorted_heatmap,
)
from src.session_manifest import processed_session_path


class LickBoutDeliveryTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "Matplotlib is required",
    )
    def test_heatmap_labels_delivery_and_cohort(self):
        output_dir = Path("tests/_lickbout_delivery_figure")
        results = {
            "time": np.array([-1.0, 0.0, 1.0]),
            "trial_matrix": np.array([[0.0, 1.0, 0.5], [0.0, 0.5, 1.0]]),
            "trial_rows": [
                {"delivery_latency": 0.8},
                {"delivery_latency": -0.2},
            ],
            "normalization": "zscore",
            "group": "Astrocyte",
            "condition": "Trained",
            "n_mice": 2,
            "minimum_delivery_latency": 0.0,
        }
        try:
            paths = save_delivery_sorted_heatmap(
                results,
                output_dir,
                formats=("svg", "png"),
                dpi=72,
                font_family="DejaVu Sans",
            )
            svg = output_dir.joinpath(
                "lick_bout_onset_delivery_sorted_heatmap.svg"
            ).read_text(encoding="utf-8")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

        self.assertEqual({path.suffix for path in paths}, {".svg", ".png"})
        self.assertIn("Astrocyte / Trained", svg)
        self.assertIn("Ensure delivery", svg)

    def test_matches_first_bout_and_delivery_within_each_cue_trial(self):
        matches = match_cue_lickbout_delivery_trials(
            cue_onsets=[10.0, 30.0, 50.0],
            lick_bout_onsets=[12.0, 15.0, 34.0, 54.0],
            delivery_onsets=[11.0, 36.0],
            recording_end=70.0,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["lick_bout_onset"], 34.0)
        self.assertEqual(matches[0]["delivery_latency"], 2.0)

    def test_computes_psth_from_the_same_paired_trials_as_heatmap(self):
        info = {
            "mouse": "M1",
            "date": "260101",
            "run": 1,
            "group": "Astrocyte",
            "condition": "Naive",
            "channel": "1",
        }
        data_root = Path("tests/_lickbout_delivery_data")
        path = processed_session_path(data_root, info)
        time = np.arange(0.0, 80.01, 0.1)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                mouse=info["mouse"],
                date=info["date"],
                run=info["run"],
                processed_schema_version="1.0",
                photo_time_465_ch1=time,
                photometry_465_ch1=time,
                dff_ch1=time,
                locomotion_time=time,
                processed_locomotion=np.zeros_like(time),
                cue_onset=np.array([20.0, 50.0]),
                lick_bout_onset=np.array([22.0, 52.0]),
                solenoid_onset=np.array([21.0, 54.0]),
            )
            results = compute_lickbout_delivery_psth(
                [info],
                data_root,
                window=(-1.0, 2.0),
                dt=0.1,
                normalization="subtract",
                baseline=(-1.0, 0.0),
            )
        finally:
            shutil.rmtree(data_root, ignore_errors=True)

        self.assertEqual(results["trial_matrix"].shape[0], 1)
        self.assertEqual(results["session_results"][0]["n_events"], 1)
        self.assertEqual(
            [row["delivery_latency"] for row in results["trial_rows"]],
            [2.0],
        )
        np.testing.assert_allclose(
            results["session_results"][0]["mean"],
            np.nanmean(results["trial_matrix"], axis=0),
        )


if __name__ == "__main__":
    unittest.main()
