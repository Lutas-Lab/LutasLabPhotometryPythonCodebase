import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.psth_statistics import (
    AVAILABLE_METRICS,
    analyze_manifest_metrics,
    prism_wide_rows,
    run_pairwise_tests,
)
from src.publication_figures import save_metric_figures
from src.session_manifest import load_session_manifest


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract hierarchical PSTH metrics and run mouse-level statistics."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-key", default="cue_onset")
    parser.add_argument(
        "--channel",
        choices=("manifest", "1", "2"),
        default="manifest",
        help="Use each manifest row's channel, or override every session.",
    )
    parser.add_argument("--window", type=float, nargs=2, default=(-5.0, 10.0))
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--normalization",
        choices=("none", "subtract", "zscore"),
        default="zscore",
    )
    parser.add_argument("--baseline", type=float, nargs=2, default=(-5.0, 0.0))
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
    parser.add_argument(
        "--null-method",
        choices=("none", "random_onsets", "circular_shift"),
        default="none",
    )
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null-exclusion", type=float, default=0.0)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("svg", "png", "pdf"),
        default=("svg", "png"),
    )
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-mouse-points", action="store_true")
    parser.add_argument("--no-pairs", action="store_true")
    parser.add_argument("--no-statistics", action="store_true")
    return parser.parse_args()


def _write_rows(path, rows):
    if not rows:
        return False
    preferred = (
        "mouse",
        "date",
        "run",
        "group",
        "condition",
        "channel",
        "event_key",
        "trial",
        "event_index",
        "event_time",
        "metric",
        "value",
        "mean",
        "sem",
        "n_trials",
        "n_sessions",
        "n_mice",
        "comparison",
        "stratum",
        "level_a",
        "level_b",
        "test",
        "n_a",
        "n_b",
        "n_pairs",
        "mean_a",
        "mean_b",
        "difference_a_minus_b",
        "effect_size",
        "effect_size_name",
        "ci_lower",
        "ci_upper",
        "statistic",
        "p_value",
        "p_adjusted_holm",
        "observed_mean",
        "null_mean",
        "null_lower",
        "null_upper",
        "empirical_p_two_sided",
        "n_shuffles",
        "path",
    )
    available = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in available]
    fieldnames.extend(sorted(available.difference(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def main():
    args = parse_arguments()
    sessions = load_session_manifest(args.manifest)
    results = analyze_manifest_metrics(
        sessions,
        args.data_root,
        event_key=args.event_key,
        channel=args.channel,
        window=args.window,
        dt=args.dt,
        normalization=args.normalization,
        baseline=args.baseline,
        response_window=args.response_window,
        peak_smoothing=args.peak_smoothing,
        metrics=args.metrics,
        null_method=args.null_method,
        n_shuffles=args.n_shuffles,
        random_seed=args.seed,
        null_exclusion=args.null_exclusion,
    )
    results["test_rows"] = run_pairwise_tests(results["mouse_rows"], test=args.test)
    results["prism_rows"] = prism_wide_rows(results["mouse_rows"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "trial_rows": "psth_trial_metrics.csv",
        "session_rows": "psth_session_metrics.csv",
        "mouse_rows": "psth_mouse_metrics.csv",
        "group_rows": "psth_group_summary.csv",
        "test_rows": "psth_statistical_tests.csv",
        "shuffle_rows": "psth_shuffle_tests.csv",
        "prism_rows": "psth_prism_wide.csv",
    }
    saved = []
    for key, filename in outputs.items():
        path = args.output_dir / filename
        if _write_rows(path, results[key]):
            saved.append(path)

    figure_paths = save_metric_figures(
        results["mouse_rows"],
        results["test_rows"],
        args.output_dir,
        formats=args.formats,
        dpi=args.dpi,
        font_family=args.font_family,
        show_mice=not args.no_mouse_points,
        show_pairs=not args.no_pairs,
        show_statistics=not args.no_statistics,
    )
    saved.extend(figure_paths)

    metadata = {
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "event_key": args.event_key,
        "channel": args.channel,
        "window": args.window,
        "dt": args.dt,
        "normalization": args.normalization,
        "baseline": args.baseline,
        "response_window": args.response_window,
        "peak_smoothing": args.peak_smoothing,
        "metrics": args.metrics,
        "test": args.test,
        "null_method": args.null_method,
        "n_shuffles": args.n_shuffles if args.null_method != "none" else 0,
        "seed": args.seed,
        "null_exclusion": args.null_exclusion,
        "hierarchy": "trials within sessions; sessions within mice; mice are N",
        "figure_formats": args.formats,
        "font_family": args.font_family,
        "show_mouse_points": not args.no_mouse_points,
        "show_pairs": not args.no_pairs,
        "show_statistics": not args.no_statistics,
    }
    metadata_path = args.output_dir / "psth_statistics_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    saved.append(metadata_path)

    print(f"Analyzed {len(results['session_rows'])} session-metric rows.")
    print(f"Produced {len(results['mouse_rows'])} mouse-level metric rows.")
    if not results["test_rows"]:
        print("No pairwise test had enough mouse-level observations.")
    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
