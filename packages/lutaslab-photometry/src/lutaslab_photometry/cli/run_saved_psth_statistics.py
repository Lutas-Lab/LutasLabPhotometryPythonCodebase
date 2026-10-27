"""Summarize standardized PSTH exports without reopening processed or raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from lutaslab_photometry.psth_results import save_saved_psth_statistics
from lutaslab_photometry.psth_statistics import AVAILABLE_METRICS


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-window", type=float, nargs=2, default=(0.0, 2.0))
    parser.add_argument("--peak-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=AVAILABLE_METRICS,
        default=("mean", "auc", "peak", "peak_latency"),
    )
    parser.add_argument(
        "--test",
        choices=("auto", "paired_t", "wilcoxon", "welch", "mannwhitney"),
        default="auto",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--formats", nargs="+", choices=("svg", "png", "pdf"), default=("svg", "png")
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-family", default="Arial")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outputs = save_saved_psth_statistics(
        args.input_dir,
        args.output_dir,
        response_window=tuple(args.response_window),
        peak_smoothing=args.peak_smoothing,
        metrics=tuple(args.metrics),
        test=args.test,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        formats=tuple(args.formats),
        dpi=args.dpi,
        font_family=args.font_family,
        progress_callback=lambda current, total, label: print(
            f"PROGRESS\t{current}\t{total}\t{label}", flush=True
        ),
    )
    for path in outputs:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
