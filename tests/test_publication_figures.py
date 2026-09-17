import importlib.util
from pathlib import Path
import shutil
import unittest

from src.publication_figures import (
    configure_publication_style,
    save_figure_formats,
    save_metric_figures,
)


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
if MATPLOTLIB_AVAILABLE:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
    plt = None


class PublicationFigureTests(unittest.TestCase):
    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "Matplotlib is not installed")
    def test_svg_preserves_editable_text(self):
        output = Path("tests/_publication_figure")
        configure_publication_style(font_family="DejaVu Sans")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("Editable label")
        try:
            paths = save_figure_formats(fig, output, formats=("svg", "png"), dpi=72)
            svg = output.with_suffix(".svg").read_text(encoding="utf-8")
        finally:
            plt.close(fig)
            output.with_suffix(".svg").unlink(missing_ok=True)
            output.with_suffix(".png").unlink(missing_ok=True)

        self.assertEqual(len(paths), 2)
        self.assertIn("<text", svg)
        self.assertIn("Editable label", svg)

    @unittest.skipUnless(
        MATPLOTLIB_AVAILABLE and SCIPY_AVAILABLE,
        "Matplotlib and SciPy are required",
    )
    def test_metric_figures_save_svg_and_png(self):
        output = Path("tests/_publication_metric_figures")
        rows = []
        for mouse, first, second in (("M1", 1.0, 2.0), ("M2", 1.5, 2.5)):
            for condition, value in (("A", first), ("B", second)):
                rows.append(
                    {
                        "mouse": mouse,
                        "group": "control",
                        "condition": condition,
                        "metric": "mean",
                        "value": value,
                    }
                )
        tests = [
            {
                "comparison": "condition_within_group",
                "stratum": "control",
                "level_a": "A",
                "level_b": "B",
                "metric": "mean",
                "p_adjusted_holm": 0.04,
            }
        ]
        try:
            paths = save_metric_figures(
                rows,
                tests,
                output,
                formats=("svg", "png"),
                dpi=72,
                font_family="DejaVu Sans",
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual({path.suffix for path in paths}, {".svg", ".png"})
        finally:
            shutil.rmtree(output, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
