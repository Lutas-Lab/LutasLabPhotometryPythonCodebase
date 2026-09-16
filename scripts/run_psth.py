import argparse
import csv
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.group_analysis import (
    compute_manifest_psth,
    compute_manifest_psth_strata,
    save_condition_comparison_figures,
    save_psth_figures,
)
from src.session_manifest import load_session_manifest


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate mouse-level and group-level peri-event photometry figures "
            "from a session manifest."
        )
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
    parser.add_argument(
        "--figures",
        choices=("individual", "group", "both"),
        default="both",
    )
    parser.add_argument(
        "--stratify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Separate PSTHs by manifest group and condition when available.",
    )
    parser.add_argument(
        "--null-method",
        choices=("none", "random_onsets", "circular_shift"),
        default="none",
        help="Optional session-local null model to compare with real event alignment.",
    )
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--null-exclusion",
        type=float,
        default=0.0,
        help="Minimum distance in seconds between null and real onsets.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("svg", "png", "pdf"),
        default=("svg", "png"),
    )
    parser.add_argument("--font-family", default="Arial")
    return parser.parse_args()


def _save_numeric_results(results, output_dir):
    result_path = output_dir / "psth_results.npz"
    np.savez_compressed(
        result_path,
        time=results["time"],
        mouse_names=np.asarray(results["mouse_names"], dtype=str),
        mouse_matrix=results["mouse_matrix"],
        group_mean=results["group_mean"],
        group_sem=results["group_sem"],
        event_key=results["event_key"],
        signal_key=results["signal_key"],
        normalization=results["normalization"],
        null_method=results["null_method"],
        n_shuffles=results["n_shuffles"],
        random_seed=results["random_seed"],
        null_exclusion=results["null_exclusion"],
        mouse_null_mean_matrix=results.get(
            "mouse_null_mean_matrix", np.empty((0, len(results["time"])))
        ),
        group_null_matrix=results.get(
            "group_null_matrix", np.empty((0, len(results["time"])))
        ),
        group_null_mean=results.get("group_null_mean", np.empty(0)),
        group_null_lower=results.get("group_null_lower", np.empty(0)),
        group_null_upper=results.get("group_null_upper", np.empty(0)),
        session_channels=np.asarray(
            [result["channel"] for result in results["session_results"]], dtype=int
        ),
    )

    summary_path = output_dir / "psth_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("mouse", "channels", "n_sessions", "n_events"),
        )
        writer.writeheader()
        for mouse in results["mouse_names"]:
            mouse_result = results["mouse_results"][mouse]
            writer.writerow(
                {
                    "mouse": mouse,
                    "channels": ";".join(
                        str(channel) for channel in mouse_result["channels"]
                    ),
                    "n_sessions": mouse_result["n_sessions"],
                    "n_events": mouse_result["n_events"],
                }
            )
    return result_path, summary_path


def _safe_path_component(value):
    value = str(value)
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )
    return safe or "all"


def main():
    args = parse_arguments()
    sessions = load_session_manifest(args.manifest)
    analysis_options = dict(
        event_key=args.event_key,
        channel=args.channel,
        window=args.window,
        dt=args.dt,
        normalization=args.normalization,
        baseline=args.baseline,
        null_method=args.null_method,
        n_shuffles=args.n_shuffles,
        random_seed=args.seed,
        null_exclusion=args.null_exclusion,
    )
    has_strata = any(
        session.get("group") or session.get("condition") for session in sessions
    )
    if args.stratify and has_strata:
        stratum_results = compute_manifest_psth_strata(
            sessions,
            args.data_root,
            **analysis_options,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        saved_figures = []
        for (group, condition), results in stratum_results.items():
            stratum_dir = (
                args.output_dir
                / _safe_path_component(group)
                / _safe_path_component(condition)
            )
            figure_paths = save_psth_figures(
                results,
                stratum_dir,
                figure_level=args.figures,
                formats=args.formats,
                dpi=args.dpi,
                font_family=args.font_family,
            )
            result_path, summary_path = _save_numeric_results(results, stratum_dir)
            saved_figures.extend(figure_paths)
            print(
                f"{group} / {condition}: {len(results['session_results'])} sessions, "
                f"{results['n_mice']} mice."
            )
            print(f"Saved numeric results: {result_path}")
            print(f"Saved summary: {summary_path}")
        comparison_paths = save_condition_comparison_figures(
            stratum_results,
            args.output_dir / "comparisons",
            formats=args.formats,
            dpi=args.dpi,
            font_family=args.font_family,
        )
        for path in saved_figures + comparison_paths:
            print(f"Saved figure: {path}")
        return

    results = compute_manifest_psth(sessions, args.data_root, **analysis_options)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = save_psth_figures(
        results,
        args.output_dir,
        figure_level=args.figures,
        formats=args.formats,
        dpi=args.dpi,
        font_family=args.font_family,
    )
    result_path, summary_path = _save_numeric_results(results, args.output_dir)

    print(f"Analyzed {len(results['session_results'])} sessions.")
    print(f"Biological units in group SEM: {results['n_mice']} mice.")
    if results["null_method"] != "none":
        print(
            f"Null comparison: {results['null_method']}, "
            f"{results['n_shuffles']} shuffles, seed {results['random_seed']}."
        )
    for path in figure_paths:
        print(f"Saved figure: {path}")
    print(f"Saved numeric results: {result_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
