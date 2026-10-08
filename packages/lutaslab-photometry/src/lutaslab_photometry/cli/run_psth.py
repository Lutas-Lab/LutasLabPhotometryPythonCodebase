import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from lutaslab_photometry.group_analysis import (
    compute_manifest_psth,
    compute_manifest_psth_strata,
    save_condition_comparison_figures,
    save_null_diagnostics,
    save_psth_figures,
    save_psth_heatmaps,
)
from lutaslab_photometry.heatmap_ordering import HEATMAP_SORT_METHODS
from lutaslab_photometry.session_manifest import load_session_manifest
from lutaslab_photometry.trial_classification import TRIAL_CLASS_KEYS


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate mouse-level and group-level peri-event response figures "
            "from a session manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-key", default="cue_onset")
    parser.add_argument(
        "--first-event-only",
        action="store_true",
        help="Use only the first selected alignment event in each session.",
    )
    parser.add_argument(
        "--allow-partial-windows",
        action="store_true",
        help="Keep recorded samples and pad unavailable window edges with NaN.",
    )
    parser.add_argument(
        "--trial-class",
        choices=TRIAL_CLASS_KEYS,
        default="all",
        help="For cue_onset analyses, include only the selected lick-response class.",
    )
    parser.add_argument(
        "--post-cue-window",
        type=float,
        default=2.0,
        help="Seconds after cue offset used to classify post-cue licking.",
    )
    parser.add_argument(
        "--signal",
        choices=("photometry", "licking"),
        default="photometry",
        help="Plot photometry or binned lick rate around the alignment events.",
    )
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
    parser.add_argument(
        "--heatmaps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save session, mouse-level, and descriptive pooled-trial heatmaps.",
    )
    parser.add_argument(
        "--heatmap-sort",
        choices=HEATMAP_SORT_METHODS,
        default="event_order",
    )
    parser.add_argument("--heatmap-sort-window", type=float, nargs=2, default=None)
    parser.add_argument(
        "--heatmap-sort-direction",
        choices=("auto", "ascending", "descending"),
        default="auto",
    )
    parser.add_argument(
        "--heatmap-unmatched",
        choices=("bottom", "exclude"),
        default="bottom",
    )
    parser.add_argument("--heatmap-cmap", default="coolwarm")
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
        signal_type=results.get("signal_type", "photometry"),
        normalization=results["normalization"],
        null_method=results["null_method"],
        n_shuffles=results["n_shuffles"],
        random_seed=results["random_seed"],
        null_exclusion=results["null_exclusion"],
        trial_class=results.get("trial_class", "all"),
        post_cue_window=results.get("post_cue_window", 2.0),
        first_event_only=results.get("first_event_only", False),
        allow_partial_windows=results.get("allow_partial_windows", False),
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
            fieldnames=(
                "mouse",
                "channels",
                "n_sessions",
                "n_events",
                "trial_class",
                "post_cue_window",
            ),
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
                    "trial_class": results.get("trial_class", "all"),
                    "post_cue_window": results.get("post_cue_window", 2.0),
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
        signal_type=args.signal,
        channel=args.channel,
        window=args.window,
        dt=args.dt,
        normalization=args.normalization,
        baseline=args.baseline,
        null_method=args.null_method,
        n_shuffles=args.n_shuffles,
        random_seed=args.seed,
        null_exclusion=args.null_exclusion,
        trial_class=args.trial_class,
        post_cue_window=args.post_cue_window,
        first_event_only=args.first_event_only,
        allow_partial_windows=args.allow_partial_windows,
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
            heatmap_paths = (
                save_psth_heatmaps(
                    results,
                    stratum_dir,
                    sort=args.heatmap_sort,
                    sort_window=args.heatmap_sort_window,
                    direction=args.heatmap_sort_direction,
                    unmatched=args.heatmap_unmatched,
                    cmap=args.heatmap_cmap,
                    formats=args.formats,
                    dpi=args.dpi,
                    font_family=args.font_family,
                )
                if args.heatmaps
                else []
            )
            diagnostic_report = (
                save_null_diagnostics(
                    results,
                    stratum_dir,
                    formats=args.formats,
                    dpi=args.dpi,
                    font_family=args.font_family,
                )
                if args.null_method != "none"
                else {"paths": [], "summary_rows": []}
            )
            result_path, summary_path = _save_numeric_results(results, stratum_dir)
            saved_figures.extend(
                figure_paths + heatmap_paths + diagnostic_report["paths"]
            )
            print(
                f"{group} / {condition}: {len(results['session_results'])} sessions, "
                f"{results['n_mice']} mice."
            )
            print(f"Saved numeric results: {result_path}")
            print(f"Saved summary: {summary_path}")
            warning_count = sum(
                row["warning_near_zero_baseline"] or row["warning_extreme_z"]
                for row in diagnostic_report["summary_rows"]
            )
            if warning_count:
                print(
                    f"WARNING: null diagnostics flagged {warning_count} session(s); "
                    f"review {stratum_dir / 'null_diagnostics'}."
                )
        comparison_paths = save_condition_comparison_figures(
            stratum_results,
            args.output_dir / "comparisons",
            formats=args.formats,
            dpi=args.dpi,
            font_family=args.font_family,
        )
        for path in saved_figures + comparison_paths:
            print(f"Saved output: {path}")
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
    if args.heatmaps:
        figure_paths.extend(
            save_psth_heatmaps(
                results,
                args.output_dir,
                sort=args.heatmap_sort,
                sort_window=args.heatmap_sort_window,
                direction=args.heatmap_sort_direction,
                unmatched=args.heatmap_unmatched,
                cmap=args.heatmap_cmap,
                formats=args.formats,
                dpi=args.dpi,
                font_family=args.font_family,
            )
        )
    diagnostic_report = (
        save_null_diagnostics(
            results,
            args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
            font_family=args.font_family,
        )
        if args.null_method != "none"
        else {"paths": [], "summary_rows": []}
    )
    figure_paths.extend(diagnostic_report["paths"])
    result_path, summary_path = _save_numeric_results(results, args.output_dir)

    print(f"Analyzed {len(results['session_results'])} sessions.")
    print(f"Biological units in group SEM: {results['n_mice']} mice.")
    if results["null_method"] != "none":
        print(
            f"Null comparison: {results['null_method']}, "
            f"{results['n_shuffles']} shuffles, seed {results['random_seed']}."
        )
    for path in figure_paths:
        print(f"Saved output: {path}")
    warning_count = sum(
        row["warning_near_zero_baseline"] or row["warning_extreme_z"]
        for row in diagnostic_report["summary_rows"]
    )
    if warning_count:
        print(
            f"WARNING: null diagnostics flagged {warning_count} session(s); "
            f"review {args.output_dir / 'null_diagnostics'}."
        )
    print(f"Saved numeric results: {result_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
